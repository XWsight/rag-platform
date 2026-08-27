"""Format-specific, bounded document parsing with no network access."""

from __future__ import annotations

import re
import stat
import zipfile
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from rag_system.loader_contracts import (
    DocumentLoadError,
    LoaderLimits,
    MissingDocumentDependencyError,
)


_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FORBIDDEN_XML_DECLARATIONS = re.compile(br"<!\s*(?:DOCTYPE|ENTITY)", re.IGNORECASE)
_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_WORD_PARAGRAPH = f"{_WORD_NAMESPACE}p"
_WORD_TEXT = f"{_WORD_NAMESPACE}t"
_WORD_TAB = f"{_WORD_NAMESPACE}tab"
_WORD_BREAKS = frozenset({f"{_WORD_NAMESPACE}br", f"{_WORD_NAMESPACE}cr"})


class _VisibleHTMLParser(HTMLParser):
    """Extract visible text while discarding executable and styling elements."""

    _IGNORED_ELEMENTS = frozenset({"script", "style", "noscript", "template", "svg", "canvas"})
    _BLOCK_ELEMENTS = frozenset(
        {
            "address", "article", "aside", "blockquote", "div", "dl", "fieldset",
            "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
            "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
            "section", "table", "td", "th", "tr", "ul",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        if self._ignored_depth:
            self._ignored_depth += 1
            return
        if normalized in self._IGNORED_ELEMENTS:
            self._ignored_depth = 1
            return
        if normalized == "br":
            self.parts.append("\n")
        elif normalized in self._BLOCK_ELEMENTS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if self._ignored_depth:
            self._ignored_depth -= 1
            return
        if normalized in self._BLOCK_ELEMENTS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def load_plain_text(path: Path) -> tuple[str, str]:
    return _decode_text(_read_file(path))


def load_html(path: Path) -> tuple[str, str]:
    decoded, encoding = _decode_text(_read_file(path))
    parser = _VisibleHTMLParser()
    try:
        parser.feed(decoded)
        parser.close()
    except Exception:
        raise DocumentLoadError("HTML 文档结构无效，无法安全解析。") from None
    return "".join(parser.parts), encoding


def load_docx(path: Path, limits: LoaderLimits) -> tuple[str, str]:
    if not zipfile.is_zipfile(path):
        raise DocumentLoadError("DOCX 文件结构无效。")
    try:
        with zipfile.ZipFile(path) as archive:
            members = _validate_archive(archive, limits)
            if "[Content_Types].xml" not in members or "word/document.xml" not in members:
                raise DocumentLoadError("DOCX 缺少必要的文档结构。")
            document_xml = _read_archive_member(archive, members["word/document.xml"], max_bytes=limits.max_uncompressed_bytes)
    except DocumentLoadError:
        raise
    except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile):
        raise DocumentLoadError("DOCX 文件损坏或使用了不支持的压缩方式。") from None

    if _FORBIDDEN_XML_DECLARATIONS.search(document_xml):
        raise DocumentLoadError("DOCX 包含不允许的 XML 声明。")
    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError:
        raise DocumentLoadError("DOCX 的文档 XML 无法解析。") from None

    paragraphs: list[str] = []
    character_count = 0
    for paragraph_count, paragraph in enumerate(root.iter(_WORD_PARAGRAPH), start=1):
        if paragraph_count > limits.max_paragraphs:
            raise DocumentLoadError("DOCX 段落数量超过安全限制。")
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == _WORD_TEXT and node.text:
                pieces.append(node.text)
            elif node.tag == _WORD_TAB:
                pieces.append("\t")
            elif node.tag in _WORD_BREAKS:
                pieces.append("\n")
        text = "".join(pieces).strip()
        if text:
            character_count += len(text)
            if character_count > limits.max_characters:
                raise DocumentLoadError("DOCX 文本长度超过安全限制。")
            paragraphs.append(text)
    return "\n\n".join(paragraphs), "docx/xml"


def load_pdf(path: Path, limits: LoaderLimits, pdf_reader_factory: Callable[[Path], Any] | None) -> tuple[str, str]:
    try:
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise DocumentLoadError("PDF 文件头无效。")
    except DocumentLoadError:
        raise
    except OSError:
        raise DocumentLoadError("无法读取 PDF 文档。") from None

    factory = pdf_reader_factory
    if factory is None:
        try:
            from pypdf import PdfReader
        except (ImportError, ModuleNotFoundError):
            raise MissingDocumentDependencyError("读取 PDF 需要安装可选依赖 pypdf。") from None

        def strict_pdf_reader(candidate: Path) -> Any:
            return PdfReader(str(candidate), strict=True)

        factory = strict_pdf_reader

    try:
        reader = factory(path)
        if bool(getattr(reader, "is_encrypted", False)):
            raise DocumentLoadError("不支持加密 PDF。")
        pages = reader.pages
        page_count = len(pages)
        if page_count == 0:
            raise DocumentLoadError("PDF 没有可读取的页面。")
        if page_count > limits.max_pages:
            raise DocumentLoadError("PDF 页数超过安全限制。")

        extracted: list[str] = []
        character_count = 0
        for page in pages:
            page_text = page.extract_text() or ""
            if not isinstance(page_text, str):
                raise DocumentLoadError("PDF 页面返回了无效文本。")
            character_count += len(page_text)
            if character_count > limits.max_characters:
                raise DocumentLoadError("PDF 文本长度超过安全限制。")
            if page_text.strip():
                extracted.append(page_text.strip())
        return "\n\n".join(extracted), "pdf"
    except DocumentLoadError:
        raise
    except Exception:
        raise DocumentLoadError("PDF 损坏或无法安全解析。") from None


def normalize_extracted_text(text: str, limits: LoaderLimits) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHARACTERS.sub("", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        raise DocumentLoadError("文档没有可索引的文本。")
    if len(normalized) > limits.max_characters:
        raise DocumentLoadError("文档文本长度超过安全限制。")
    paragraphs = [part for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if len(paragraphs) > limits.max_paragraphs:
        raise DocumentLoadError("文档段落数量超过安全限制。")
    return normalized


def _validate_archive(archive: zipfile.ZipFile, limits: LoaderLimits) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > limits.max_archive_members:
        raise DocumentLoadError("DOCX 压缩包包含过多文件。")

    members: dict[str, zipfile.ZipInfo] = {}
    total_uncompressed = 0
    total_compressed = 0
    for info in infos:
        name = _validated_member_name(info)
        if name in members:
            raise DocumentLoadError("DOCX 压缩包包含重复路径。")
        members[name] = info
        if info.is_dir():
            continue
        if info.flag_bits & 0x1:
            raise DocumentLoadError("不支持加密的 DOCX 压缩成员。")
        total_uncompressed += info.file_size
        total_compressed += info.compress_size
        if total_uncompressed > limits.max_uncompressed_bytes:
            raise DocumentLoadError("DOCX 解压后大小超过安全限制。")

    if total_uncompressed:
        ratio = total_uncompressed / max(total_compressed, 1)
        if ratio > limits.max_compression_ratio:
            raise DocumentLoadError("DOCX 压缩率异常，已拒绝解析。")
    return members


def _validated_member_name(info: zipfile.ZipInfo) -> str:
    raw_name = info.filename
    if not raw_name or "\\" in raw_name or "\x00" in raw_name:
        raise DocumentLoadError("DOCX 包含无效的内部路径。")
    pure_path = PurePosixPath(raw_name)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise DocumentLoadError("DOCX 包含不安全的内部路径。")
    if pure_path.parts and ":" in pure_path.parts[0]:
        raise DocumentLoadError("DOCX 包含不安全的内部路径。")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise DocumentLoadError("DOCX 不允许包含符号链接。")
    return pure_path.as_posix().rstrip("/")


def _read_archive_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_bytes: int) -> bytes:
    parts: list[bytes] = []
    total = 0
    with archive.open(info, "r") as stream:
        while True:
            chunk = stream.read(min(65_536, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise DocumentLoadError("DOCX 成员解压后大小超过安全限制。")
            parts.append(chunk)
    return b"".join(parts)


def _read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise DocumentLoadError("无法读取文档内容。") from None


def _decode_text(raw: bytes) -> tuple[str, str]:
    sample = raw[:65_536]
    unsafe_controls = sum(byte < 32 and byte not in {9, 10, 13} or byte == 127 for byte in sample)
    if b"\x00" in raw or unsafe_controls > max(4, len(sample) // 100):
        raise DocumentLoadError("文件包含二进制或危险控制字符。")
    for encoding in _TEXT_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("无法识别文本编码，请转换为 UTF-8 后重试。")
