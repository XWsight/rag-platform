from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from rag_system import server, workbench
from rag_system.config import Settings
from rag_system.knowledge_base_assets import KnowledgeBaseAssets
from rag_system.rag_platform import RagPlatform
from rag_system.rag_service import RagService
from rag_system.runtime_bootstrap import build_service
from rag_system.web_results import rank_web_results


class EntrypointTests(unittest.TestCase):
    def test_api_launcher_keeps_the_single_node_safe_defaults(self) -> None:
        with patch.object(server.uvicorn, "run") as run:
            self.assertEqual(
                server.main(("--host", "127.0.0.2", "--port", "8123", "--access-log")),
                0,
            )

        run.assert_called_once_with(
            "rag_system.asgi:app",
            host="127.0.0.2",
            port=8123,
            log_level="info",
            access_log=True,
            workers=1,
            timeout_graceful_shutdown=30,
        )

    def test_api_launcher_rejects_out_of_range_ports(self) -> None:
        with self.assertRaises(SystemExit) as rejected:
            server.main(("--port", "70000"))
        self.assertEqual(rejected.exception.code, 2)

    def test_workbench_launcher_uses_safe_local_launch_options(self) -> None:
        demo = Mock()
        with patch.dict(os.environ, {}, clear=True), patch(
            "rag_system.workbench.build_service",
            return_value=(object(), Settings()),
        ), patch("rag_system.workbench.create_demo", return_value=demo):
            self.assertEqual(workbench.main(), 0)
            self.assertEqual(os.environ["NO_PROXY"], "127.0.0.1,localhost")
            self.assertEqual(os.environ["no_proxy"], "127.0.0.1,localhost")

        demo.launch.assert_called_once_with(
            inbrowser=True,
            share=False,
            show_error=False,
            max_file_size=Settings().max_file_bytes,
            strict_cors=True,
            enable_monitoring=False,
        )

    def test_legacy_module_names_remain_compatibility_only(self) -> None:
        from rag_system.assets import KnowledgeBaseAssets as legacy_assets
        from rag_system.bootstrap import build_service as legacy_build_service
        from rag_system.platform import RagPlatform as legacy_platform
        from rag_system.service import RagService as legacy_service
        from rag_system.web import rank_web_results as legacy_web_results

        self.assertIs(legacy_assets, KnowledgeBaseAssets)
        self.assertIs(legacy_build_service, build_service)
        self.assertIs(legacy_platform, RagPlatform)
        self.assertIs(legacy_service, RagService)
        self.assertIs(legacy_web_results, rank_web_results)


if __name__ == "__main__":
    unittest.main()
