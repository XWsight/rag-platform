from __future__ import annotations

import unittest
from contextlib import contextmanager
from dataclasses import replace

from rag_system.config import Settings
from rag_system.domain import (
    AnswerClaim,
    AnswerRequest,
    Chunk,
    GeneratedAnswer,
    Route,
    SearchHit,
    WebSearchResult,
)
from rag_system.provider_errors import ProviderUnavailableError
from rag_system.rag_service import RagService


class FakeRetriever:
    def __init__(self, hits, *, error=None):
        self.hits = hits
        self.error = error
        self.queries = []

    def search(self, query: str, *, top_k: int):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.hits[:top_k]


class FakeIndexManager:
    def __init__(self, retriever):
        self.retriever = retriever

    @contextmanager
    def lease(self, index_id: str):
        if index_id != "idx":
            raise KeyError(index_id)
        yield self.retriever


class MultiIndexManager:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    @contextmanager
    def lease(self, index_id: str):
        yield self.retrievers[index_id]


class FakeChat:
    def __init__(self, answer=None, *, available=True, error=None):
        self.response = answer or GeneratedAnswer(
            (AnswerClaim("基于资料的回答。", ("L1",)),)
        )
        self.available = available
        self.error = error
        self.calls = 0

    def answer(self, question, evidence):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


class ProfileAwareChat(FakeChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models: list[str] = []

    def answer_with_model(self, question, evidence, *, model: str):
        self.models.append(model)
        return self.answer(question, evidence)


class FakeWeb:
    def __init__(self, results=(), *, available=True, error=None):
        self.results = results
        self.available = available
        self.error = error
        self.calls = 0

    def search(self, query, *, count):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results


class FakePlanner:
    def __init__(self, queries=(), *, available=True, error=None):
        self.queries = queries
        self.available = available
        self.error = error
        self.calls = 0

    def plan_queries(self, question, *, max_queries):
        self.calls += 1
        if self.error:
            raise self.error
        return self.queries[:max_queries]


class ProfileAwarePlanner(FakePlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models: list[str] = []

    def plan_queries_with_model(self, question, *, max_queries, model: str):
        self.models.append(model)
        return self.plan_queries(question, max_queries=max_queries)


def make_hit(score: float, reasons=("dense", "sparse")) -> SearchHit:
    text = "RAG 使用检索到的资料生成答案。"
    chunk = Chunk("chunk", "doc", "guide.md", text, 0, 0, len(text), "原理")
    return SearchHit(chunk, score, reasons=reasons)


class RagServiceTests(unittest.TestCase):
    def build_service(self, hit, chat=None, web=None, planner=None, retrieval_error=None):
        chat = chat or FakeChat()
        web = web or FakeWeb()
        service = RagService(
            Settings(),
            FakeIndexManager(FakeRetriever([hit] if hit else [], error=retrieval_error)),
            chat,
            web,
            query_planner=planner,
        )
        return service, chat, web

    def test_local_answer_uses_chat_and_never_calls_web(self) -> None:
        service, chat, web = self.build_service(make_hit(0.9))
        result = service.answer("idx", AnswerRequest("什么是 RAG", "s", True, True))
        self.assertEqual(result.decision.route, Route.LOCAL)
        self.assertEqual(chat.calls, 1)
        self.assertEqual(web.calls, 0)
        self.assertEqual(result.citations[0].citation_id, "L1")

    def test_answer_resolves_the_requested_model_profile_for_profile_aware_provider(self) -> None:
        chat = ProfileAwareChat()
        settings = replace(
            Settings(),
            model_profile_ids=("default", "fast"),
            model_profile_models=(("default", "glm-5.2"), ("fast", "glm-4.5-air")),
        )
        service = RagService(
            settings,
            FakeIndexManager(FakeRetriever([make_hit(0.9)])),
            chat,
            FakeWeb(),
        )

        service.answer(
            "idx",
            AnswerRequest("What is RAG?", "profile-session", allow_cloud=True, model_profile_id="fast"),
        )

        self.assertEqual(chat.models, ["glm-4.5-air"])

    def test_research_planning_uses_the_same_resolved_model_profile(self) -> None:
        planner = ProfileAwarePlanner(("RAG sources",))
        settings = replace(
            Settings(),
            model_profile_ids=("default", "fast"),
            model_profile_models=(("default", "glm-5.2"), ("fast", "glm-4.5-air")),
        )
        service = RagService(
            settings,
            FakeIndexManager(FakeRetriever([make_hit(0.9)])),
            ProfileAwareChat(),
            FakeWeb(),
            query_planner=planner,
        )

        service.answer(
            "idx",
            AnswerRequest(
                "What is RAG?", "research-session", allow_cloud=True, deep_research=True,
                model_profile_id="fast",
            ),
        )

        self.assertEqual(planner.models, ["glm-4.5-air"])

    def test_multiple_indexes_are_fused_before_one_answer_is_generated(self) -> None:
        first_chunk = Chunk("chunk-a", "doc-a", "alpha.md", "Alpha evidence", 0, 0, 14)
        second_chunk = Chunk("chunk-b", "doc-b", "beta.md", "Beta evidence", 0, 0, 13)
        manager = MultiIndexManager({
            "idx-a": FakeRetriever([
                SearchHit(first_chunk, 0.9, lexical_score=0.8, reasons=("dense", "sparse"))
            ]),
            "idx-b": FakeRetriever([
                SearchHit(second_chunk, 0.8, lexical_score=0.7, reasons=("dense", "sparse"))
            ]),
        })
        chat = FakeChat(
            GeneratedAnswer((AnswerClaim("Combined answer.", ("L1", "L2")),))
        )
        service = RagService(
            Settings(), manager, chat, FakeWeb()  # type: ignore[arg-type]
        )

        result = service.answer_many(
            ("idx-a", "idx-b"), AnswerRequest("Combined?", "session", True, False)
        )

        self.assertEqual(chat.calls, 1)
        self.assertEqual({item.source_name for item in result.citations}, {"alpha.md", "beta.md"})
        self.assertEqual(result.claims[0].citation_ids, ("L1", "L2"))

    def test_optional_citation_policy_allows_an_uncited_generated_claim(self) -> None:
        chat = FakeChat(GeneratedAnswer((AnswerClaim("Uncited answer.", ()),)))
        service, _, _ = self.build_service(make_hit(0.9), chat=chat)

        result = service.answer(
            "idx", AnswerRequest("What is RAG?", "session", True, False, False, False)
        )

        self.assertEqual(chat.calls, 1)
        self.assertEqual(result.claims[0].citation_ids, ())

    def test_focused_retrieval_profile_limits_evidence_to_the_top_hit(self) -> None:
        first = make_hit(0.9)
        second_chunk = Chunk("second", "doc", "guide.md", "Second evidence", 1, 0, 15)
        service, _, _ = self.build_service(first)
        service.index_manager.retriever.hits.append(SearchHit(second_chunk, 0.8, reasons=("dense", "sparse")))

        result = service.answer(
            "idx", AnswerRequest("What is RAG?", "session", False, False, False, True, "focused")
        )

        self.assertEqual(len(result.citations), 1)

    def test_privacy_switches_prevent_all_network_calls(self) -> None:
        service, chat, web = self.build_service(make_hit(0.9))
        result = service.answer("idx", AnswerRequest("什么是 RAG", "s", False, False))
        self.assertEqual(result.decision.route, Route.RETRIEVAL_ONLY)
        self.assertEqual(chat.calls, 0)
        self.assertEqual(web.calls, 0)

    def test_low_confidence_can_refuse_without_web(self) -> None:
        service, chat, web = self.build_service(make_hit(0.05, ("dense",)))
        result = service.answer("idx", AnswerRequest("未知问题", "s", True, False))
        self.assertEqual(result.decision.route, Route.REFUSED)
        self.assertEqual(chat.calls, 0)
        self.assertEqual(web.calls, 0)

    def test_web_route_builds_valid_web_citations(self) -> None:
        web_result = WebSearchResult("r", "可靠来源", "联网找到的资料", "https://example.com")
        service, chat, web = self.build_service(
            make_hit(0.05, ("dense",)),
            chat=FakeChat(GeneratedAnswer((AnswerClaim("联网回答。", ("W1",)),))),
            web=FakeWeb([web_result]),
        )
        result = service.answer("idx", AnswerRequest("联网问题", "s", True, True))
        self.assertEqual(result.decision.route, Route.WEB)
        self.assertEqual(web.calls, 1)
        self.assertEqual(chat.calls, 1)
        self.assertEqual(result.citations[0].citation_id, "W1")

    def test_claims_are_rendered_with_an_explicit_evidence_mapping(self) -> None:
        service, _, _ = self.build_service(
            make_hit(0.9),
            chat=FakeChat(
                GeneratedAnswer(
                    (
                        AnswerClaim("结论一。", ("L1",)),
                        AnswerClaim("结论二。", ("L1",)),
                    )
                )
            ),
        )
        result = service.answer("idx", AnswerRequest("问题", "s", True, False))
        self.assertEqual(result.answer, "结论一。 [L1]\n\n结论二。 [L1]")
        self.assertEqual(result.claims[0].citation_ids, ("L1",))
        self.assertEqual(result.diagnostics["grounded_claim_count"], 2)
        self.assertEqual(result.diagnostics["grounding_citation_count"], 2)

    def test_invalid_model_grounding_fails_closed_instead_of_editing_text(self) -> None:
        invalid = GeneratedAnswer((AnswerClaim("没有可用依据的结论。", ("W99",)),))
        service, _, _ = self.build_service(make_hit(0.9), chat=FakeChat(invalid))
        result = service.answer("idx", AnswerRequest("问题", "s", True, False))
        self.assertEqual(result.decision.route, Route.ERROR)
        self.assertEqual(result.claims, ())
        self.assertNotIn("没有可用依据", result.answer)
        self.assertEqual(result.diagnostics["provider_error"], "GroundingContractError")

    def test_provider_failure_preserves_retrieved_evidence(self) -> None:
        chat = FakeChat(error=ProviderUnavailableError("offline"))
        service, _, _ = self.build_service(make_hit(0.9), chat=chat)
        result = service.answer("idx", AnswerRequest("问题", "s", True, False))
        self.assertEqual(result.decision.route, Route.ERROR)
        self.assertEqual(len(result.citations), 1)
        self.assertNotIn("offline", result.answer)

    def test_embedding_provider_failure_returns_a_sanitized_result(self) -> None:
        service, chat, web = self.build_service(
            make_hit(0.9),
            retrieval_error=ProviderUnavailableError("embedding details stay private"),
        )

        result = service.answer("idx", AnswerRequest("问题", "s", True, True))

        self.assertEqual(result.decision.route, Route.ERROR)
        self.assertEqual(result.diagnostics["embedding_error"], "ProviderUnavailableError")
        self.assertFalse(result.citations)
        self.assertEqual(chat.calls, 0)
        self.assertEqual(web.calls, 0)
        self.assertNotIn("embedding details", result.answer)
        self.assertNotIn("embedding details", str(result.diagnostics))

    def test_recent_questions_contextualize_retrieval_without_cross_session_leakage(self) -> None:
        service, _, _ = self.build_service(make_hit(0.9))
        retriever = service.index_manager.retriever

        first = service.answer("idx", AnswerRequest("介绍混合检索", "session-a", False, False))
        second = service.answer("idx", AnswerRequest("它有什么优点", "session-a", False, False))
        service.answer("idx", AnswerRequest("完全不同的问题", "session-b", False, False))

        self.assertEqual(first.diagnostics["history_turns"], 0)
        self.assertEqual(second.diagnostics["history_turns"], 1)
        self.assertIn("介绍混合检索", retriever.queries[1])
        self.assertIn("它有什么优点", retriever.queries[1])
        self.assertNotIn("云端生成未开启", retriever.queries[1])
        self.assertEqual(retriever.queries[2], "完全不同的问题")

    def test_clear_session_removes_retrieval_history(self) -> None:
        service, _, _ = self.build_service(make_hit(0.9))
        service.answer("idx", AnswerRequest("第一问", "session-a", False, False))
        self.assertTrue(service.clear_session("session-a"))
        service.answer("idx", AnswerRequest("第二问", "session-a", False, False))
        self.assertEqual(service.index_manager.retriever.queries[-1], "第二问")

    def test_research_mode_uses_bounded_query_plan(self) -> None:
        planner = FakePlanner(("子问题一", "子问题二"))
        service, _, _ = self.build_service(make_hit(0.9), planner=planner)
        result = service.answer(
            "idx",
            AnswerRequest("复杂问题", "session", True, False, True),
        )
        self.assertEqual(planner.calls, 1)
        self.assertEqual(len(service.index_manager.retriever.queries), 3)
        self.assertEqual(result.diagnostics["planned_query_count"], 3)
        self.assertEqual(result.diagnostics["web_query_count"], 0)

    def test_research_mode_never_plans_without_cloud_permission(self) -> None:
        planner = FakePlanner(("不应调用",))
        service, _, _ = self.build_service(make_hit(0.9), planner=planner)
        service.answer("idx", AnswerRequest("问题", "session", False, False, True))
        self.assertEqual(planner.calls, 0)

    def test_research_mode_has_bounded_multi_query_web_search(self) -> None:
        planner = FakePlanner(("查询一", "查询二", "查询三", "查询四"))
        web_result = WebSearchResult("r", "来源", "研究资料", "https://example.com")
        service, _, web = self.build_service(
            make_hit(0.9),
            web=FakeWeb([web_result]),
            planner=planner,
        )
        result = service.answer(
            "idx",
            AnswerRequest("复杂问题", "session", True, True, True),
        )
        self.assertEqual(web.calls, service.settings.research_max_web_queries)
        self.assertEqual(result.diagnostics["web_query_count"], 3)
        self.assertEqual(result.decision.route, Route.HYBRID)

    def test_web_failures_are_counted_without_exposing_the_error_message(self) -> None:
        service, _, web = self.build_service(
            make_hit(0.9),
            web=FakeWeb(error=ProviderUnavailableError("network details stay private")),
            planner=FakePlanner(("查询一", "查询二")),
        )
        result = service.answer(
            "idx",
            AnswerRequest("复杂问题", "session", True, True, True),
        )

        self.assertEqual(web.calls, service.settings.research_max_web_queries)
        self.assertEqual(result.diagnostics["web_error"], "ProviderUnavailableError")
        self.assertEqual(
            result.diagnostics["web_error_count"], service.settings.research_max_web_queries
        )
        self.assertEqual(
            result.diagnostics["web_error_counts"],
            f"ProviderUnavailableError:{service.settings.research_max_web_queries}",
        )
        self.assertNotIn("network details", str(result.diagnostics))

    def test_provider_failure_retains_preceding_web_failure_diagnostics(self) -> None:
        service, _, web = self.build_service(
            make_hit(0.9),
            chat=FakeChat(error=ProviderUnavailableError("chat unavailable")),
            web=FakeWeb(error=ProviderUnavailableError("web unavailable")),
            planner=FakePlanner(("查询一",)),
        )
        result = service.answer(
            "idx",
            AnswerRequest("复杂问题", "session", True, True, True),
        )

        self.assertEqual(result.decision.route, Route.ERROR)
        self.assertEqual(result.diagnostics["provider_error"], "ProviderUnavailableError")
        self.assertEqual(result.diagnostics["web_error"], "ProviderUnavailableError")
        self.assertEqual(result.diagnostics["web_error_count"], web.calls)
        self.assertEqual(
            result.diagnostics["web_error_counts"],
            f"ProviderUnavailableError:{web.calls}",
        )


if __name__ == "__main__":
    unittest.main()
