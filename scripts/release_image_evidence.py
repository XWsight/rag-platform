"""Record the immutable image digest produced by a release build."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")


def release_image_evidence(*, image: str, digest: str, source_revision: str) -> dict[str, object]:
    """Build a small, schema-stable record for an immutable OCI image."""

    if _IMAGE.fullmatch(image) is None:
        raise ValueError("release image name is invalid")
    if _DIGEST.fullmatch(digest) is None:
        raise ValueError("release image digest must be a sha256 digest")
    if _REVISION.fullmatch(source_revision) is None:
        raise ValueError("release source revision must be a full Git SHA")
    return {
        "schema_version": 1,
        "image": image,
        "digest": digest,
        "immutable_reference": f"{image}@{digest}",
        "source_revision": source_revision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        evidence = release_image_evidence(
            image=arguments.image,
            digest=arguments.digest,
            source_revision=arguments.source_revision,
        )
    except ValueError as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(evidence["immutable_reference"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
