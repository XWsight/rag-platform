"""Compatibility launcher; new development commands use ``rag_system.workbench``."""

from rag_system.workbench import main


if __name__ == "__main__":
    raise SystemExit(main())
