"""Validate document paths and assemble parsed content into domain objects."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rag_system.document_parsing import (
    load_docx,
    load_html,
    load_pdf,
    load_plain_text,
    normalize_extracted_text,
)
from rag_system.domain import SourceDocument
from rag_system.loader_contracts import (
    HTML_EXTENSIONS,
    TEXT_EXTENSIONS,
    DocumentLoadError,
    LoaderLimits,
    MissingDocumentDependencyError,
    SUPPORTED_EXTENSIONS,
)
from rag_system.security import safe_source_name
from rag_system.text import stable_digest


class SecureDocumentLoader:
    """Load supported files into stable ``SourceDocument`` domain objects."""

    def __init__(
        self,
        limits: LoaderLimits | None = None,
        *,
        allowed_root: str | os.PathLike[str] | None = None,
        pdf_reader_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.limits = (limits or LoaderLimits()).validate()
        self._pdf_reader_factory = pdf_reader_factory
        self._allowed_root = self._resolve_allowed_root(allowed_root)

    def load(self, paths: Sequence[str | os.PathLike[str]]) -> tuple[SourceDocument, ...]:
        candidates = tuple(paths)
        if not candidates:
            raise DocumentLoadError("请至少选择一个文档。")
        if len(candidates) > self.limits.max_documents:
            raise DocumentLoadError(f"一次最多读取 {self.limits.max_documents} 个文档。")

        validated: list[tuple[Path, int]] = []
        total_bytes = 0
        for value in candidates:
            path, size = self._validate_path(value)
            total_bytes += size
            if total_bytes > self.limits.max_total_file_bytes:
                raise DocumentLoadError("全部文档的文件大小超过安全限制。")
            validated.append((path, size))

        return tuple(self._load_validated(path) for path, _ in validated)

    def load_one(self, path: str | os.PathLike[str]) -> SourceDocument:
        validated, _ = self._validate_path(path)
        return self._load_validated(validated)

    @staticmethod
    def _resolve_allowed_root(value: str | os.PathLike[str] | None) -> Path | None:
        if value is None:
            return None
        try:
            root = Path(value).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise ValueError("allowed_root must be an existing directory") from None
        if not root.is_dir():
            raise ValueError("allowed_root must be an existing directory")
        return root

    def _validate_path(self, value: str | os.PathLike[str]) -> tuple[Path, int]:
        try:
            candidate = Path(value)
            if candidate.is_symlink():
                raise DocumentLoadError("不支持符号链接文档。")
            resolved = candidate.resolve(strict=True)
        except DocumentLoadError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError):
            raise DocumentLoadError("文档路径无效或无法访问。") from None

        if self._allowed_root is not None:
            try:
                resolved.relative_to(self._allowed_root)
            except ValueError:
                raise DocumentLoadError("文档不在允许读取的目录中。") from None
        if not resolved.is_file():
            raise DocumentLoadError("上传内容不是常规文件。")
        if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DocumentLoadError(f"不支持该文档格式；允许的扩展名：{supported}")

        try:
            size = resolved.stat().st_size
        except OSError:
            raise DocumentLoadError("无法读取文档属性。") from None
        if size == 0:
            raise DocumentLoadError("文档为空。")
        if size > self.limits.max_file_bytes:
            raise DocumentLoadError("文档大小超过安全限制。")
        return resolved, size

    def _load_validated(self, path: Path) -> SourceDocument:
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text, encoding = load_plain_text(path)
        elif suffix in HTML_EXTENSIONS:
            text, encoding = load_html(path)
        elif suffix == ".docx":
            text, encoding = load_docx(path, self.limits)
        elif suffix == ".pdf":
            text, encoding = load_pdf(path, self.limits, self._pdf_reader_factory)
        else:  # The extension is checked at the path boundary.
            raise DocumentLoadError("不支持该文档格式。")

        normalized = normalize_extracted_text(text, self.limits)
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        name = safe_source_name(path)
        return SourceDocument(
            document_id=f"doc_{stable_digest([name.lower(), content_hash])}",
            name=name,
            text=normalized,
            content_hash=content_hash,
            encoding=encoding,
        )


__all__ = [
    "DocumentLoadError",
    "LoaderLimits",
    "MissingDocumentDependencyError",
    "SUPPORTED_EXTENSIONS",
    "SecureDocumentLoader",
]
