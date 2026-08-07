"""The HTTP surface, implementing .agents/API_CONTRACT.md.

The two response profiles are enforced here rather than left to client
discipline. The device claim is computer-aided triage and notification
(21 CFR 892.2080), which permits prioritizing images but not marking locations
on the original image, so a client that only ever asks for `triage` cannot leak
location information into the worklist by accident.

The `radiologist` profile carries fracture boxes when a detector is loaded, and
`localization: null` when one is not. OsteoJEPA's surprise map returned a null
result and never filled that field; a map that does not localize is worse than
no map.

Only this profile runs the detector. It costs about a second per image against
the classifier's 79 ms, and the worklist is not permitted to show location
anyway, so putting it on the ranking path would buy nothing and cost twelve
times the latency.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import compat
from .preprocess import UnreadableImage
from .scorer import AgeRequired, Scorer


class ImageIn(BaseModel):
    image_id: str
    r2_key: str | None = None
    content: str | None = None
    view: int | None = None
    laterality: Literal["L", "R"] | None = None


class StudyIn(BaseModel):
    study_id: str | None = None
    profile: Literal["triage", "radiologist"] = "triage"
    images: list[ImageIn] = Field(default_factory=list)
    age_years: float | None = None
    sex: Literal["M", "F", "O"] | None = None


def error(status: int, code: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, **extra})


def fetch_object(key: str) -> bytes:
    """Fetch an uploaded image by object key.

    Configured through PHYSIS_OBJECT_BASE, an https prefix the key is appended
    to. Unset means the deployment has no object store wired up, and the
    contract's 404 is the honest answer: the service cannot read that key.
    """
    base = os.environ.get("PHYSIS_OBJECT_BASE")
    if not base:
        raise FileNotFoundError("no object store configured")
    import urllib.request

    with urllib.request.urlopen(base.rstrip("/") + "/" + key.lstrip("/")) as response:
        return response.read()


def resolve_image(image: ImageIn) -> bytes:
    if bool(image.r2_key) == bool(image.content):
        raise ValueError("image_source_ambiguous")
    if image.content:
        try:
            return base64.b64decode(image.content, validate=True)
        except (binascii.Error, ValueError) as err:
            raise UnreadableImage(str(err)) from err
    return fetch_object(image.r2_key)


def build_app(scorer_factory) -> FastAPI:
    """`scorer_factory` is a callable returning a Scorer, or None when unloaded."""
    app = FastAPI(title="Physis triage", version="1.1")

    # The web client is a browser on a different origin, so without this every
    # request is refused by the browser before it reaches the service - and the
    # failure looks like the API being down rather than a policy decision.
    # PHYSIS_CORS_ORIGINS is a comma-separated allowlist; "*" is the demo default
    # and should be narrowed to the app's origin for anything longer-lived.
    origins = [
        origin.strip()
        for origin in os.environ.get("PHYSIS_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        """A landing page, so an integrator who guesses the base URL is not met
        with a bare 404 and left wondering whether the service is up."""
        scorer = scorer_factory()
        return {
            "service": "Physis triage",
            "contract": "v1",
            "status": "ok" if scorer is not None else "model_unavailable",
            "docs": "/docs",
            "endpoints": [
                "GET  /v1/health",
                "POST /v1/score/study",
                "POST /v1/score/image",
                "POST /v1/predict   (compatibility shape for the web client)",
            ],
            "note": (
                "Triage and notification only. This service does not diagnose, "
                "and the triage profile carries no location information."
            ),
        }

    @app.get("/v1/health")
    def health():
        scorer = scorer_factory()
        if scorer is None:
            return error(503, "model_unavailable")
        return {"status": "ok", "model": scorer.model_info()}

    @app.post("/v1/score/study")
    def score_study(payload: StudyIn):
        scorer = scorer_factory()
        if scorer is None:
            # The client falls back to arrival order for the whole worklist.
            # Nothing in the clinical workflow may depend on this service.
            return error(503, "model_unavailable")

        if not payload.study_id:
            return error(422, "study_id_required")
        if not payload.images:
            return error(422, "images_required")
        try:
            scorer.band_of(payload.age_years)
        except AgeRequired as err:
            # No default age. Scoring against the wrong band silently is worse
            # than refusing.
            return error(422, "age_required", detail=str(err))

        prepared = []
        for image in payload.images:
            try:
                prepared.append({
                    "data": resolve_image(image),
                    "image_id": image.image_id,
                    "view": image.view,
                    "laterality": image.laterality,
                })
            except ValueError as err:
                if str(err) == "image_source_ambiguous":
                    return error(422, "image_source_ambiguous", image_id=image.image_id)
                return error(415, "unreadable_image", image_id=image.image_id)
            except FileNotFoundError:
                return error(404, "object_not_found", image_id=image.image_id)
            except Exception:  # noqa: BLE001 - any fetch failure is the same 404
                return error(404, "object_not_found", image_id=image.image_id)

        # A deadline the service owns, so a slow response becomes a 504 the
        # client can fall back from rather than a hung worklist. Fetching the
        # uploads already happened, so the budget covers what is left.
        budget_ms = float(os.environ.get("PHYSIS_DEADLINE_MS", 30000))
        started = time.perf_counter()
        try:
            result = scorer.score_study(
                prepared,
                study_id=payload.study_id,
                age_years=payload.age_years,
                sex=payload.sex,
                localize=payload.profile == "radiologist",
            )
        except UnreadableImage as err:
            return error(415, "unreadable_image", detail=str(err))
        if (time.perf_counter() - started) * 1000.0 > budget_ms:
            return error(504, "timeout", budget_ms=budget_ms)

        if payload.profile == "radiologist" and result.get("localization") is None:
            # No detector loaded. Present and empty on purpose rather than
            # absent, so a client can tell "no boxes found" from "this
            # deployment cannot localize".
            result["localization"] = None
            result["localization_note"] = (
                "no detector loaded in this deployment"
            )
        return result

    @app.post("/v1/score/image")
    def score_image(payload: StudyIn):
        if len(payload.images) != 1:
            return error(422, "single_image_required")
        payload.study_id = payload.study_id or payload.images[0].image_id
        response = score_study(payload)
        if isinstance(response, JSONResponse):
            return response
        response.pop("triage_score", None)
        return response

    compat.register(app, scorer_factory)
    return app
