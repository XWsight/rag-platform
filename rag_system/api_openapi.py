"""OpenAPI customizations kept outside the application composition root."""

from typing import Any, cast

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def install_multipart_openapi_schema(app: FastAPI) -> None:
    """Expose upload fields as browser-selectable binary files in Swagger UI."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return cast(dict[str, Any], app.openapi_schema)
        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        request_schema = (
            schema["paths"]["/v1/knowledge-bases"]["post"]["requestBody"]["content"]
            ["multipart/form-data"]["schema"]
        )
        reference = request_schema.get("$ref")
        if isinstance(reference, str):
            component_name = reference.rsplit("/", maxsplit=1)[-1]
            request_schema = schema["components"]["schemas"][component_name]
        request_schema["properties"]["files"] = {
            "type": "array",
            "items": {"type": "string", "format": "binary"},
            "title": "Files",
            "description": "Documents to index",
        }
        app.openapi_schema = schema
        return cast(dict[str, Any], schema)

    app.openapi = custom_openapi


__all__ = ["install_multipart_openapi_schema"]
