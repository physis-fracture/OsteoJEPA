"""Request and response types, and the one error envelope.

These are the contract. `/openapi.json` is generated from them, so a document
that disagrees with the runtime is not possible: there is one definition and
both read it.

The previous version made `study_id`, `age_years` and `images` optional in the
model and then re-checked them by hand in the route. That produced an OpenAPI
document saying the fields were nullable while the service rejected requests
without them, which is a specification that lies about its own behaviour.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl

# The web application's enums. The model works in integers; that mapping is an
# implementation detail and stays inside the service.
VIEW_MAP = {"PA": 1, "AP": 1, "LATERAL": 2, "OTHER": 3, "UNKNOWN": None}
LATERALITY_MAP = {"left": "L", "right": "R", "unknown": None}
SEX_MAP = {"male": "M", "female": "F", "unknown": None}

# Age is the one field with no fallback. It selects the normalization band, so a
# default would silently score a child against the wrong population.
AGE_MIN, AGE_MAX = 0.2, 19.0

# A wrist examination is a handful of projections. The ceiling is a cost bound,
# not a clinical one: each image is a forward pass plus a detector pass.
MAX_IMAGES = 8


class PredictImage(BaseModel):
    image_id: Annotated[
        str,
        Field(min_length=1, max_length=128, description="Unique within the study; echoed back"),
    ]
    image_url: Annotated[
        HttpUrl,
        Field(description="Presigned HTTPS URL the service fetches. Never persisted, never logged"),
    ]
    view: Annotated[
        Literal["PA", "AP", "LATERAL", "OTHER", "UNKNOWN"],
        Field(description="Projection. UNKNOWN maps to the trained unknown embedding"),
    ] = "UNKNOWN"
    laterality: Annotated[
        Literal["left", "right", "unknown"],
        Field(description="Which wrist"),
    ] = "unknown"


class PredictRequest(BaseModel):
    study_id: Annotated[
        str,
        Field(min_length=1, max_length=128, description="The unit the worklist is ordered by"),
    ]
    age_years: Annotated[
        float,
        Field(
            ge=AGE_MIN,
            le=AGE_MAX,
            description=(
                "Patient age in years. Required, with no default: it selects the "
                "age band the percentile is computed against"
            ),
        ),
    ]
    sex: Literal["male", "female", "unknown"] = "unknown"
    images: Annotated[
        list[PredictImage],
        Field(min_length=1, max_length=MAX_IMAGES),
    ]


class PredictImageResult(BaseModel):
    image_id: str
    triage_score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Calibrated probability for this image alone"),
    ]
    valid_patch_fraction: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Share of the 384x384 canvas that is not padding. Well under 0.5 "
                "means a badly cropped upload and a score resting on few patches"
            ),
        ),
    ]
    boxes: Annotated[
        list[list[float]] | None,
        Field(
            description=(
                "Fracture boxes as [x0, y0, x1, y1, score] in the pixel space of "
                "the uploaded image. An empty list means the detector ran and "
                "returned nothing; null means this deployment cannot localize. "
                "The absence of a box is not evidence of the absence of a fracture"
            )
        ),
    ] = None

    # OsteoJEPA's quantities. It returned a null result, so these have no value
    # to carry. Null rather than omitted, so an existing insert keeps working.
    implicit_age: Annotated[None, Field(deprecated=True)] = None
    implicit_age_gap: Annotated[None, Field(deprecated=True)] = None
    surprise_map: Annotated[None, Field(deprecated=True)] = None
    implicit_age_map: Annotated[None, Field(deprecated=True)] = None


class PredictData(BaseModel):
    study_id: str
    triage_score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Study score: the maximum over the study's images"),
    ]
    priority_percentile: Annotated[
        float,
        Field(ge=0.0, le=100.0, description="Age-band-relative percentile on a 0-100 scale"),
    ]
    age_band: Annotated[str, Field(description="Which population the percentile is against")]
    images: list[PredictImageResult]
    inference_time_ms: Annotated[int, Field(ge=0)]
    model_version: Annotated[str, Field(description="checkpoint@contract, for provenance")]


class PredictResponse(BaseModel):
    success: Literal[True] = True
    message: str
    data: PredictData


class ErrorItem(BaseModel):
    field: str | None = None
    image_id: str | None = None
    message: str


class ErrorResponse(BaseModel):
    """One envelope for every failure.

    The client branches on `success` and never reads the HTTP status, so a body
    that only sets a status would be read as a success with missing fields.
    """

    success: Literal[False] = False
    message: str
    error_code: str
    errors: list[ErrorItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Public and deliberately thin.

    Provenance belongs on the authenticated response, where `model_version` and
    `model` already carry it. An unauthenticated endpoint has no reason to name
    the checkpoint that is loaded.
    """

    status: Literal["ok", "model_unavailable"]
    contract_version: str


class ServiceInfo(BaseModel):
    service: str
    version: str
    health: str
    docs: str


__all__ = [
    "AGE_MAX",
    "AGE_MIN",
    "LATERALITY_MAP",
    "MAX_IMAGES",
    "SEX_MAP",
    "VIEW_MAP",
    "ErrorItem",
    "ErrorResponse",
    "HealthResponse",
    "PredictData",
    "PredictImage",
    "PredictImageResult",
    "PredictRequest",
    "PredictResponse",
    "ServiceInfo",
]
