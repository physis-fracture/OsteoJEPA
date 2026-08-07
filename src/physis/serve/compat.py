"""`POST /v1/predict` — the shape the web client was built against.

The web application was written against an earlier PRD and expects a different
endpoint, different enum spellings, a `{success, data}` envelope, and a
percentile on 0-100. Adapting here costs one file; adapting there would mean
touching the client, the server action, the viewer and a database migration in
an application that already works.

This is a translation layer and nothing more. It calls the same `Scorer`, so
there is no second copy of preprocessing, calibration, or the padding rule.

Two things it cannot translate, because the quantities no longer exist:
`implicit_age` / `implicit_age_gap` and `surprise_map` / `implicit_age_map` were
OsteoJEPA's, and OsteoJEPA returned a null result. They are returned as `null`
rather than omitted, so a client can tell "we have no value" from "the field was
forgotten". Fracture boxes are returned instead, and they are strictly more
useful than a map that did not localize.
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .preprocess import UnreadableImage
from .scorer import AgeRequired

# The web application's enums, from its Supabase schema.
VIEW_MAP = {"PA": 1, "AP": 1, "LATERAL": 2, "OTHER": 3, "UNKNOWN": None}
LATERALITY_MAP = {"left": "L", "right": "R", "unknown": None}
SEX_MAP = {"male": "M", "female": "F", "unknown": None}

FETCH_TIMEOUT_S = 30


class PredictImage(BaseModel):
    image_id: str
    image_url: str
    view: Literal["PA", "AP", "LATERAL", "OTHER", "UNKNOWN"] = "UNKNOWN"
    laterality: Literal["left", "right", "unknown"] = "unknown"


class PredictRequest(BaseModel):
    study_id: str | None = None
    age_years: float | None = None
    sex: Literal["male", "female", "unknown"] = "unknown"
    images: list[PredictImage] = Field(default_factory=list)


def envelope_error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    """The client reads the body regardless of status, so the envelope carries it.

    `finalizeStudy` branches on `success`, not on the HTTP code, so an error that
    only sets a status would be read as a successful response with missing
    fields.
    """
    return JSONResponse(
        status_code=status,
        content={"success": False, "message": message, "error_code": code, **extra},
    )


def fetch_url(url: str) -> bytes:
    """Fetch a presigned object URL.

    The web application hands out short-lived R2 URLs rather than raw keys, which
    is the better arrangement: this service never needs the bucket credentials.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("image_url must be an http(s) URL")
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as response:
        return response.read()


def register(app: FastAPI, scorer_factory) -> None:
    @app.post("/v1/predict")
    def predict(payload: PredictRequest):
        started = time.perf_counter()
        scorer = scorer_factory()
        if scorer is None:
            return envelope_error(
                503, "SERVICE_UNAVAILABLE", "No model is loaded in this deployment"
            )
        if not payload.study_id:
            return envelope_error(422, "VALIDATION_ERROR", "study_id is required")
        if not payload.images:
            return envelope_error(422, "VALIDATION_ERROR", "at least one image is required")
        try:
            scorer.band_of(payload.age_years)
        except AgeRequired as err:
            return envelope_error(422, "VALIDATION_ERROR", str(err))

        prepared = []
        for image in payload.images:
            try:
                prepared.append({
                    "data": fetch_url(image.image_url),
                    "image_id": image.image_id,
                    "view": VIEW_MAP.get(image.view),
                    "laterality": LATERALITY_MAP.get(image.laterality),
                })
            except ValueError as err:
                return envelope_error(422, "VALIDATION_ERROR", str(err))
            except Exception:  # noqa: BLE001 - any fetch failure reads the same
                return envelope_error(
                    404, "IMAGE_NOT_FOUND", f"could not fetch {image.image_id}"
                )

        budget_ms = float(os.environ.get("PHYSIS_DEADLINE_MS", 30000))
        try:
            result = scorer.score_study(
                prepared,
                study_id=payload.study_id,
                age_years=payload.age_years,
                sex=SEX_MAP.get(payload.sex),
                # The web client has no `profile`; it drives the radiologist's
                # case view, so boxes are always computed for it.
                localize=True,
            )
        except UnreadableImage as err:
            return envelope_error(415, "UNREADABLE_IMAGE", str(err))
        if (time.perf_counter() - started) * 1000.0 > budget_ms:
            return envelope_error(504, "INFERENCE_TIMEOUT", "deadline exceeded")

        boxes_by_image = {
            entry["image_id"]: entry["boxes"]
            for entry in (result.get("localization") or [])
        }
        model = result["model"]
        return {
            "success": True,
            "message": "ok",
            "data": {
                "study_id": result["study_id"],
                "triage_score": result["triage_score"],
                # The web UI formats this with a percent sign and filters at 80
                # and 95, so it is sent on 0-100. The native endpoint keeps 0-1.
                "priority_percentile": round(result["priority_percentile"] * 100.0, 2),
                "age_band": result["age_band"],
                "images": [
                    {
                        "image_id": image["image_id"],
                        "triage_score": image["triage_score"],
                        "valid_patch_fraction": image["valid_patch_fraction"],
                        "boxes": boxes_by_image.get(image["image_id"], []),
                        # OsteoJEPA's quantities. Null, not omitted: absent would
                        # read as an oversight, and there is no value to give.
                        "implicit_age": None,
                        "implicit_age_gap": None,
                        "surprise_map": None,
                        "implicit_age_map": None,
                    }
                    for image in result["images"]
                ],
                "inference_time_ms": result["inference_time_ms"],
                "model_version": f"{model['checkpoint']}@{model['contract']}",
                "model": model,
            },
        }
