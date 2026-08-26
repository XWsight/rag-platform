"""Bounded multipart upload materialization for the HTTP boundary."""

from collections.abc import Sequence

from fastapi import UploadFile

from rag_system.api_errors import ApiBoundaryError
from rag_system.application import UploadDocument


_UPLOAD_READ_SIZE = 64 * 1024


async def read_uploads(
    files: Sequence[UploadFile],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
) -> tuple[UploadDocument, ...]:
    """Read multipart documents with explicit aggregate limits and cleanup."""

    uploads: list[UploadDocument] = []
    total = 0
    try:
        for upload in files:
            filename = upload.filename
            if not isinstance(filename, str) or not filename.strip():
                raise ApiBoundaryError(422, "invalid_request", "The request could not be validated.")
            content = bytearray()
            while True:
                block = await upload.read(_UPLOAD_READ_SIZE)
                if not block:
                    break
                if len(content) + len(block) > max_file_bytes or total + len(block) > max_total_bytes:
                    raise ApiBoundaryError(
                        413,
                        "upload_limit_exceeded",
                        "The upload exceeds the configured limits.",
                    )
                content.extend(block)
                total += len(block)
            uploads.append(UploadDocument(display_name=filename, source=bytes(content)))
    finally:
        for upload in files:
            await upload.close()
    return tuple(uploads)


__all__ = ["read_uploads"]
