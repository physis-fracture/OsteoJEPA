"""`POST /v1/predict` — the one inference route.

It used to be a compatibility shim beside `/v1/score/study` and `/v1/score/image`.
Those are gone. Three public routes onto one scorer meant three request shapes,
three sets of failure modes, and an OpenAPI document that described a surface
larger than the product used.

What that removed, and what it did not:

The `profile` switch is gone from the public API. It chose between returning
boxes and not, which mattered because the detector costs about a second per
image against the classifier's 79 ms. `Scorer.score_study(localize=...)` still
takes the argument, so the fast path exists internally whenever a caller needs
it; there is simply no public contract built around it.

`r2_key` and inline base64 are gone. The service takes a presigned URL and
fetches it, which is the arrangement that keeps bucket credentials out of here.

Integer view codes and M/F/O are gone from the surface. The model still works in
integers; the mapping lives in `schemas.VIEW_MAP` and does not leak into the API.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import APIRouter, Request, Response

from .fetch import (
    ImageFetchError,
    ImageHostRejected,
    ImageTooLarge,
    ImageUnreachable,
    fetch_image,
    redact,
)
from .preprocess import UnreadableImage
from .schemas import (
    LATERALITY_MAP,
    SEX_MAP,
    VIEW_MAP,
    PredictRequest,
    PredictResponse,
)
from .scorer import AgeRequired

log = logging.getLogger("physis.predict")

# A busy container refusing work is better than a container accepting work it
# cannot finish inside the deadline. Counted per container, so with several
# containers the effective ceiling is higher; this bounds one container's queue,
# not the account's spend.
MAX_CONCURRENT = int(os.environ.get("PHYSIS_MAX_CONCURRENT", "8"))
_in_flight = 0


def envelope(status: int, code: str, message: str, errors=None):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status,
        content={
            "success": False,
            "message": message,
            "error_code": code,
            "errors": errors or [],
        },
    )


def build_router(scorer_factory, responses: dict) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/v1/predict",
        response_model=PredictResponse,
        responses=responses,
        summary="Score a study and localize fractures",
        description=(
            "Fetches each presigned image URL, scores it, aggregates the study "
            "score as the maximum over its images, and returns the age-band "
            "percentile. Triage and notification only: this does not diagnose."
        ),
    )
    def predict(payload: PredictRequest, request: Request, response: Response):
        global _in_flight

        # Echoed back so one call can be followed from the client's log into
        # Modal's. Generated when the client does not supply one.
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        response.headers["X-Request-ID"] = request_id

        scorer = scorer_factory()
        if scorer is None:
            return envelope(503, "SERVICE_UNAVAILABLE", "No model is loaded in this deployment")

        if _in_flight >= MAX_CONCURRENT:
            return envelope(429, "RATE_LIMITED", "Server is at capacity, retry shortly")

        started = time.perf_counter()
        _in_flight += 1
        try:
            prepared = []
            for image in payload.images:
                url = str(image.image_url)
                try:
                    data = fetch_image(url)
                except ImageHostRejected as err:
                    # A rejected host is the client sending somewhere it should
                    # not, not a missing object, so it is a validation failure.
                    return envelope(
                        422, "VALIDATION_ERROR", "Validation failed",
                        [{"field": "image_url", "image_id": image.image_id, "message": str(err)}],
                    )
                except ImageTooLarge as err:
                    return envelope(
                        413, "IMAGE_TOO_LARGE", str(err),
                        [{"image_id": image.image_id, "message": str(err)}],
                    )
                except (ImageUnreachable, ImageFetchError):
                    # The URL is not echoed: its query string is a working
                    # credential for as long as it lives.
                    log.warning("fetch failed rid=%s image=%s url=%s",
                                request_id, image.image_id, redact(url))
                    return envelope(
                        404, "IMAGE_NOT_FOUND", f"could not fetch {image.image_id}",
                        [{"image_id": image.image_id, "message": "image could not be fetched"}],
                    )
                prepared.append({
                    "data": data,
                    "image_id": image.image_id,
                    "view": VIEW_MAP.get(image.view),
                    "laterality": LATERALITY_MAP.get(image.laterality),
                })

            budget_ms = float(os.environ.get("PHYSIS_DEADLINE_MS", 30000))
            try:
                result = scorer.score_study(
                    prepared,
                    study_id=payload.study_id,
                    age_years=payload.age_years,
                    sex=SEX_MAP.get(payload.sex),
                    localize=True,
                )
            except AgeRequired as err:
                # Pydantic bounds catch this first; reachable only if the band
                # table and the schema disagree, which is worth a clear answer.
                return envelope(
                    422, "VALIDATION_ERROR", "Validation failed",
                    [{"field": "age_years", "message": str(err)}],
                )
            except UnreadableImage as err:
                return envelope(415, "UNREADABLE_IMAGE", str(err))

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms > budget_ms:
                return envelope(504, "INFERENCE_TIMEOUT", "deadline exceeded")
        finally:
            _in_flight -= 1

        boxes_by_image = {
            entry["image_id"]: entry["boxes"]
            for entry in (result.get("localization") or [])
        }
        localizes = result.get("localization") is not None
        model = result["model"]

        log.info(
            "rid=%s study=%s images=%d score=%.4f band=%s ms=%d model=%s",
            request_id, payload.study_id, len(prepared), result["triage_score"],
            result["age_band"], int(elapsed_ms), model["checkpoint"],
        )

        return {
            "success": True,
            "message": "Prediction completed successfully",
            "data": {
                "study_id": result["study_id"],
                "triage_score": result["triage_score"],
                # 0-100 here. The client formats it with a percent sign and
                # filters at 80 and 95.
                "priority_percentile": round(result["priority_percentile"] * 100.0, 2),
                "age_band": result["age_band"],
                "images": [
                    {
                        "image_id": image["image_id"],
                        "triage_score": image["triage_score"],
                        "valid_patch_fraction": image["valid_patch_fraction"],
                        # [] and null differ: [] is "the detector looked and
                        # found nothing", null is "this deployment cannot
                        # localize". A client must not collapse them.
                        "boxes": boxes_by_image.get(image["image_id"], []) if localizes else None,
                        "implicit_age": None,
                        "implicit_age_gap": None,
                        "surprise_map": None,
                        "implicit_age_map": None,
                    }
                    for image in result["images"]
                ],
                "inference_time_ms": int(round(elapsed_ms)),
                "model_version": f"{model['checkpoint']}@{model['contract']}",
            },
        }

    return router


__all__ = ["build_router", "envelope", "MAX_CONCURRENT"]
