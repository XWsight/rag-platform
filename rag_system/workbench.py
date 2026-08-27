"""Command-line launcher for the developer-only local Gradio workbench."""

from __future__ import annotations

import os

from rag_system.runtime_bootstrap import build_service
from rag_system.gradio_workbench import create_demo


def main() -> int:
    """Start the non-production workbench with privacy-safe local defaults."""

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    service, settings = build_service()
    demo = create_demo(service, settings)
    demo.launch(
        inbrowser=True,
        share=False,
        show_error=False,
        max_file_size=settings.max_file_bytes,
        strict_cors=True,
        enable_monitoring=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
