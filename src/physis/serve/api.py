"""The HTTP surface. Three routes, one of them doing the work.

    GET  /              service information
    GET  /v1/health     liveness, public and thin
    POST /v1/predict    inference, bearer authenticated

`/v1/score/study` and `/v1/score/image` are gone. They scored through the same
`Scorer` as `/v1/predict` while exposing a second request contract built around
internal concepts: object keys, integer view codes, M/F/O, and a `profile`
switch. Nothing in the product used them.

The regulatory position did not live in that switch. The device claim is
computer-aided triage and notification (21 CFR 892.2080), which permits
prioritizing a worklist but not marking locations on the image for the treating
clinician. What enforces it is that the on-call physician has no account: the
role model is `radiologist | admin`, and the worklist reads scores from the
application's own database rather than calling this service at all. A profile
parameter on an endpoint the worklist never calls was never what was holding
that line.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import predict as predict_route
from .schemas import ErrorResponse, HealthResponse, ServiceInfo

log = logging.getLogger("physis.api")

CONTRACT_VERSION = "1.1"

# Declared on the route so `/openapi.json` carries the security scheme and
# `/docs` grows an Authorize button. auto_error off: the 401 body has to be the
# project envelope, not FastAPI's `{"detail": ...}`.
security = HTTPBearer(auto_error=False, description="Server-to-server API key")

PREDICT_RESPONSES: dict = {
    401: {"model": ErrorResponse, "description": "Missing or invalid bearer token"},
    404: {"model": ErrorResponse, "description": "Image could not be fetched"},
    413: {"model": ErrorResponse, "description": "Image exceeds the size ceiling"},
    415: {"model": ErrorResponse, "description": "Fetched, but not a decodable image"},
    422: {"model": ErrorResponse, "description": "Invalid request"},
    429: {"model": ErrorResponse, "description": "Server at capacity"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
    503: {"model": ErrorResponse, "description": "No model loaded"},
    504: {"model": ErrorResponse, "description": "Inference deadline exceeded"},
}


def envelope(status_code: int, code: str, message: str, errors=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "error_code": code,
            "errors": errors or [],
        },
    )


def api_key() -> str:
    return os.environ.get("PHYSIS_API_KEY", "").strip()


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    """Bearer authentication, when PHYSIS_API_KEY is set.

    Unset leaves the route open, which is what a local process wants and what
    every client had before this existed. On Modal the key arrives from a named
    secret, so a deploy without it fails rather than silently publishing an
    unguarded endpoint.

    `hmac.compare_digest` rather than `==`: a byte-by-byte comparison returns
    faster on a wrong first character than on a wrong last one, and that timing
    difference is enough to recover a key one character at a time.
    """
    expected = api_key()
    if not expected:
        return
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(credentials.credentials.strip(), expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def build_app(scorer_factory) -> FastAPI:
    """`scorer_factory` is a callable returning a Scorer, or None when unloaded."""
    app = FastAPI(
        title="Physis triage API",
        version=CONTRACT_VERSION,
        description=(
            "Pediatric wrist fracture triage. Orders a radiologist's reading "
            "queue by risk; it does not diagnose, and the absence of a box is "
            "not evidence of the absence of a fracture."
        ),
    )

    # No CORS. This is server to server: the Next.js server calls it, not a
    # browser, so an Access-Control-Allow-Origin header would widen the surface
    # without any client needing it. Authentication is the access boundary.

    @app.exception_handler(RequestValidationError)
    async def validation_failed(request: Request, exc: RequestValidationError):
        """Pydantic's rejection, in the project's envelope.

        FastAPI answers `{"detail": [...]}` by default. The client branches on
        `success`, so that body reads as a successful response with every field
        missing rather than as the validation failure it is.
        """
        errors = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            errors.append({"field": location or None, "message": error["msg"]})
        return envelope(422, "VALIDATION_ERROR", "Validation failed", errors)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        codes = {401: "UNAUTHORIZED", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
        response = envelope(
            exc.status_code,
            codes.get(exc.status_code, "ERROR"),
            str(exc.detail),
        )
        for key, value in (exc.headers or {}).items():
            response.headers[key] = value
        return response

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        """Never let an exception reach the client.

        A traceback names filesystem paths, checkpoint locations and sometimes
        argument values. The log keeps all of it; the response says nothing.
        """
        log.exception("unhandled error on %s", request.url.path)
        return envelope(500, "INTERNAL_ERROR", "An unexpected server error occurred.")

    @app.get("/", response_model=ServiceInfo, summary="Service information")
    def root():
        return {
            "service": "Physis triage inference",
            "version": CONTRACT_VERSION,
            "health": "/v1/health",
            "docs": "/docs",
        }

    @app.get(
        "/v1/health",
        response_model=HealthResponse,
        summary="Liveness",
        description=(
            "Public and deliberately thin. Model provenance is on the "
            "authenticated response, which carries model_version."
        ),
    )
    def health():
        loaded = scorer_factory() is not None
        return JSONResponse(
            status_code=200 if loaded else 503,
            content={
                "status": "ok" if loaded else "model_unavailable",
                "contract_version": CONTRACT_VERSION,
            },
        )

    app.include_router(
        predict_route.build_router(scorer_factory, PREDICT_RESPONSES),
        dependencies=[Depends(require_api_key)],
    )
    return app


__all__ = ["build_app", "require_api_key", "CONTRACT_VERSION", "PREDICT_RESPONSES"]
