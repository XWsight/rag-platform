"""Authentication, role dependencies, and tenant-scoped API rate limiting."""

# Do not enable postponed annotations here: FastAPI resolves the locally-created
# security schemes from these dependency annotations while registering routes.
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import hashlib
import math
from typing import Annotated, Literal, Protocol

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from rag_system.api_errors import ApiBoundaryError
from rag_system.rate_limit import TokenBucketRateLimiter
from rag_system.tenancy import ApiKeyAuthenticator, Principal


Role = Literal["reader", "writer", "operator"]


class RequestConsumer(Protocol):
    """Charge one authenticated request against its tenant's token bucket."""

    def __call__(self, request: Request, principal: Principal, *, tokens: float = 1.0) -> None: ...


@dataclass(frozen=True, slots=True)
class ApiSecurityDependencies:
    """Framework-ready dependencies produced from trusted application components."""

    authenticate_request: Callable[[Request], Principal]
    consume: RequestConsumer
    reader: Callable[..., Awaitable[Principal]]
    writer: Callable[..., Awaitable[Principal]]
    operator: Callable[..., Awaitable[Principal]]


def build_api_security_dependencies(
    *,
    authenticator: ApiKeyAuthenticator,
    rate_limiter: TokenBucketRateLimiter,
) -> ApiSecurityDependencies:
    """Build request dependencies without reflecting credentials or tenant identifiers."""

    api_key_scheme = APIKeyHeader(
        name="X-API-Key",
        scheme_name="ApiKeyAuth",
        description="A tenant-scoped service API key.",
        auto_error=False,
    )
    bearer_scheme = HTTPBearer(
        scheme_name="BearerAuth",
        bearerFormat="API key",
        description="The same tenant API key carried as a Bearer credential.",
        auto_error=False,
    )

    def authenticate_request(request: Request) -> Principal:
        cached = getattr(request.state, "principal", None)
        if isinstance(cached, Principal):
            return cached
        principal = authenticator.authenticate_headers(_raw_headers(request))
        request.state.principal = principal
        request.state.tenant_hash = hashlib.sha256(
            principal.tenant_id.value.encode("utf-8")
        ).hexdigest()[:16]
        return principal

    async def authenticate(
        request: Request,
        _api_key: Annotated[str | None, Security(api_key_scheme)],
        _bearer: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
    ) -> Principal:
        del _api_key, _bearer
        return authenticate_request(request)

    def require_role(role: Role) -> Callable[..., Awaitable[Principal]]:
        async def dependency(
            principal: Annotated[Principal, Depends(authenticate)],
        ) -> Principal:
            if not principal.has_role(role):
                raise ApiBoundaryError(403, "forbidden", "The operation is not permitted.")
            return principal

        return dependency

    def consume(request: Request, principal: Principal, *, tokens: float = 1.0) -> None:
        if getattr(request.state, "upload_rate_preconsumed", False):
            request.state.upload_rate_preconsumed = False
            return
        requested = min(float(tokens), rate_limiter.capacity)
        decision = rate_limiter.acquire(principal.tenant_id.value, tokens=requested)
        request.state.rate_limit_decision = decision
        if not decision.allowed:
            retry_after = max(1, math.ceil(decision.retry_after_seconds))
            raise ApiBoundaryError(
                429,
                "rate_limit_exceeded",
                "The request rate limit was exceeded.",
                headers={"Retry-After": str(retry_after)},
            )

    return ApiSecurityDependencies(
        authenticate_request=authenticate_request,
        consume=consume,
        reader=require_role("reader"),
        writer=require_role("writer"),
        operator=require_role("operator"),
    )


def _raw_headers(request: Request) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in request.scope.get("headers", ())
    )


__all__ = ["ApiSecurityDependencies", "RequestConsumer", "build_api_security_dependencies"]
