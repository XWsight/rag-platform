"""Create a safe, reviewable derivative layer for a copied RAG Studio base."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_ROOT = _PROJECT_ROOT / "templates" / "derivative"
_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,39}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{7,64}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-name",
        required=True,
        help="New Python package name: lowercase letters, digits, and underscores.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="A new, empty destination directory, usually the package name.",
    )
    parser.add_argument("--product-name", required=True, help="Display name, 1-80 printable chars.")
    parser.add_argument(
        "--product-tagline",
        required=True,
        help="Display tagline, 1-160 printable chars.",
    )
    parser.add_argument(
        "--base-revision",
        help="Optional base Git commit to record; defaults to the current HEAD when available.",
    )
    return parser


def create_derivative(
    *,
    package_name: str,
    output: Path,
    product_name: str,
    product_tagline: str,
    base_revision: str | None = None,
) -> Path:
    """Copy and render the derivative layer without modifying an existing path."""

    normalized_package = package_name.strip()
    if not _IDENTIFIER_PATTERN.fullmatch(normalized_package):
        raise ValueError("package_name must use 2-40 lowercase letters, digits, or underscores")
    normalized_name = _display_text(product_name, maximum=80, label="product_name")
    normalized_tagline = _display_text(product_tagline, maximum=160, label="product_tagline")
    revision = _base_revision(base_revision)
    destination = output.expanduser().resolve()
    if destination == Path(destination.anchor):
        raise ValueError("output cannot be a filesystem root")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("output already exists; refusing to overwrite it")
    if not _TEMPLATE_ROOT.is_dir():
        raise RuntimeError("derivative template directory is unavailable")

    replacements = {
        "{{PACKAGE_NAME}}": normalized_package,
        "{{FACTORY_CLASS}}": _factory_class_name(normalized_package),
        "{{PRODUCT_NAME}}": normalized_name,
        "{{PRODUCT_TAGLINE}}": normalized_tagline,
        "{{BASE_REVISION}}": revision,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        shutil.copytree(_TEMPLATE_ROOT, staged)
        _render_templates(staged, replacements)
        try:
            staged.rename(destination)
        except FileExistsError:
            raise FileExistsError("output already exists; refusing to overwrite it") from None
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        destination = create_derivative(
            package_name=args.package_name,
            output=args.output,
            product_name=args.product_name,
            product_tagline=args.product_tagline,
            base_revision=args.base_revision,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Created derivative layer: {destination}")
    print(f"Start its API with: uvicorn {args.package_name}.api_app:app")
    return 0


def _display_text(value: str, *, maximum: int, label: str) -> str:
    normalized = value.strip()
    if not 1 <= len(normalized) <= maximum or not normalized.isprintable():
        raise ValueError(f"{label} must be 1-{maximum} printable characters")
    return normalized


def _factory_class_name(package_name: str) -> str:
    return "".join(part.capitalize() for part in package_name.split("_")) + "ProviderFactory"


def _base_revision(value: str | None) -> str:
    if value is not None:
        revision = value.strip()
        if _REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError("base_revision must be a 7-64 character lowercase Git commit")
        return revision
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            encoding="ascii",
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "unrecorded"
    revision = result.stdout.strip()
    return revision if _REVISION_PATTERN.fullmatch(revision) is not None else "unrecorded"


def _render_templates(destination: Path, replacements: dict[str, str]) -> None:
    for template_path in sorted(destination.rglob("*.template")):
        rendered = template_path.read_text(encoding="utf-8")
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        unresolved = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", rendered)))
        if unresolved:
            raise RuntimeError("derivative template contains unresolved placeholders")
        target_path = template_path.with_suffix("")
        target_path.write_text(rendered, encoding="utf-8", newline="\n")
        template_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
