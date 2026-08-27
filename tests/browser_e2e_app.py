"""Deterministic HTTP fixture used by browser end-to-end tests.

It deliberately assembles the production FastAPI application instead of
stubbing ``fetch`` in the browser.  The fixture keeps embedding/model work out
of the UI suite while preserving the public API, authentication, and static
asset boundary that a browser uses in production.
"""

from __future__ import annotations

import io
import logging

from rag_system.api import create_app
from rag_system.application import PlatformUnavailableError
from rag_system.catalog import KnowledgeBaseStatus
from rag_system.observability import JsonEventLogger
from rag_system.rate_limit import TokenBucketRateLimiter
from rag_system.tenancy import ApiKeyAuthenticator, Principal
from tests.test_api import ALL_ROLES_KEY, FakePlatform, _principal, _record


class BrowserE2EPlatform(FakePlatform):
    """Small deterministic implementation of the public application port."""

    def __init__(self) -> None:
        super().__init__()
        self.record = None

    def create_knowledge_base(
        self,
        principal: Principal,
        *,
        display_name: str,
        documents: object,
        idempotency_key: str,
    ):
        if display_name == "故障资料":
            raise PlatformUnavailableError("fixture indexing backend unavailable")
        submission = super().create_knowledge_base(
            principal,
            display_name=display_name,
            documents=documents,
            idempotency_key=idempotency_key,
        )
        self.record = submission.knowledge_base
        return submission

    def get_job(self, principal: Principal, job_id: str):
        if self.record is not None:
            self.record = _record(status=KnowledgeBaseStatus.READY)
        return super().get_job(principal, job_id)

    def list_knowledge_bases(self, principal: Principal, *, limit: int, offset: int):
        del principal, limit, offset
        return () if self.record is None else (self.record,)

    def list_knowledge_bases_after(
        self,
        principal: Principal,
        *,
        updated_at: float,
        resource_id: str,
        limit: int,
    ):
        del principal, updated_at, resource_id, limit
        return ()

    def delete_knowledge_base(self, principal: Principal, resource_id: str) -> bool:
        del principal
        if self.record is None or self.record.resource_id != resource_id:
            return False
        self.record = None
        return True


def create_browser_e2e_app():
    """Return the same composed application served by the browser test runner."""

    platform = BrowserE2EPlatform()
    authenticator = ApiKeyAuthenticator.from_mapping(
        {ALL_ROLES_KEY: _principal("browser-e2e", {"reader", "writer", "operator"})}
    )
    sink = logging.Logger("browser-e2e")
    sink.handlers.clear()
    sink.addHandler(logging.StreamHandler(io.StringIO()))
    return create_app(
        platform=platform,
        authenticator=authenticator,
        rate_limiter=TokenBucketRateLimiter(rate_per_second=100, capacity=100),
        logger=JsonEventLogger(sink),
        readiness=True,
        close_on_shutdown=False,
    )


app = create_browser_e2e_app()
