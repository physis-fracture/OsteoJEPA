"""The API surface and its failure modes.

M6 is accepted on handling the failure path - a missing age, an unreadable file,
a model timeout - so those are tested rather than assumed. The model here is
untrained; what is under test is the contract, not the score.
"""

import base64
import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from PIL import Image

from physis.models.classifier import build_classifier
from physis.serve.api import build_app
from physis.serve.preprocess import UnreadableImage, load_grayscale, preprocess
from physis.serve.scorer import AgeRequired, Scorer
from physis.utils.config import load_config


def png_bytes(height=500, width=380, mode="I;16"):
    array = (np.random.rand(height, width) * 60000).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    cfg = load_config("configs/base.yaml", ["encoder.init=random", "encoder.depth=2"])
    tmp = tmp_path_factory.mktemp("serve")
    model = build_classifier(cfg)
    checkpoint = tmp / "best.pt"
    torch.save({"model": model.state_dict()}, checkpoint)

    bands = OmegaConf.to_container(cfg.age_bands)
    calibration = tmp / "cal.json"
    calibration.write_text(
        json.dumps(
            {
                "checkpoint": "test",
                "temperature": 3.3,
                "bands": [
                    {"name": b["name"], "band_index": i, "n": 100,
                     "quantiles": np.linspace(-10, 10, 101).tolist()}
                    for i, b in enumerate(bands)
                ],
            }
        ),
        encoding="utf-8",
    )
    return Scorer(cfg, str(checkpoint), str(calibration))


@pytest.fixture(scope="module")
def client(scorer):
    return TestClient(build_app(lambda: scorer))


def study_payload(**overrides):
    payload = {
        "study_id": "0001_01",
        "age_years": 11.5,
        "sex": "M",
        "images": [{"image_id": "img0", "content": b64(png_bytes()), "view": 1,
                    "laterality": "L"}],
    }
    payload.update(overrides)
    return payload


# --- the happy path -------------------------------------------------------

def test_scores_a_study(client):
    response = client.post("/v1/score/study", json=study_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["study_id"] == "0001_01"
    assert 0.0 <= body["triage_score"] <= 1.0
    assert 0.0 <= body["priority_percentile"] <= 1.0
    assert body["age_band"] == "11"
    assert body["model"]["contract"] == "v1"
    assert body["inference_time_ms"] > 0


def test_study_score_is_the_max_over_its_images(client, scorer):
    payload = study_payload()
    payload["images"] = [
        {"image_id": "a", "content": b64(png_bytes())},
        {"image_id": "b", "content": b64(png_bytes())},
    ]
    body = client.post("/v1/score/study", json=payload).json()
    assert len(body["images"]) == 2
    assert body["triage_score"] == max(i["triage_score"] for i in body["images"])


def test_health_reports_provenance(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    # Every displayed number must be traceable to a checkpoint and a calibration.
    assert {"checkpoint", "calibration", "contract"} <= set(body["model"])


def test_health_is_503_without_a_model():
    unloaded = TestClient(build_app(lambda: None))
    assert unloaded.get("/v1/health").status_code == 503
    response = unloaded.post("/v1/score/study", json=study_payload())
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


# --- the failure path M6 is accepted on -----------------------------------

@pytest.mark.parametrize("age", [None, 0.0, 25.0])
def test_missing_or_impossible_age_is_422(client, age):
    response = client.post("/v1/score/study", json=study_payload(age_years=age))
    assert response.status_code == 422
    assert response.json()["error"] == "age_required"


def test_missing_study_id_is_422(client):
    response = client.post("/v1/score/study", json=study_payload(study_id=None))
    assert response.status_code == 422
    assert response.json()["error"] == "study_id_required"


def test_both_sources_is_422(client):
    payload = study_payload()
    payload["images"][0]["r2_key"] = "uploads/x.png"
    response = client.post("/v1/score/study", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "image_source_ambiguous"


def test_neither_source_is_422(client):
    payload = study_payload()
    payload["images"][0].pop("content")
    response = client.post("/v1/score/study", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "image_source_ambiguous"


def test_unreadable_upload_is_415(client):
    payload = study_payload()
    payload["images"][0]["content"] = b64(b"this is not an image")
    response = client.post("/v1/score/study", json=payload)
    assert response.status_code == 415
    assert response.json()["error"] == "unreadable_image"


def test_unconfigured_object_store_is_404(client):
    payload = study_payload()
    payload["images"][0] = {"image_id": "img0", "r2_key": "uploads/missing.png"}
    response = client.post("/v1/score/study", json=payload)
    assert response.status_code == 404
    assert response.json()["error"] == "object_not_found"


def test_exceeding_the_deadline_is_504(client, monkeypatch):
    """M6 is accepted on handling a model timeout, so the service owns a deadline."""
    monkeypatch.setenv("PHYSIS_DEADLINE_MS", "0")
    response = client.post("/v1/score/study", json=study_payload())
    assert response.status_code == 504
    assert response.json()["error"] == "timeout"


# --- the regulatory separation --------------------------------------------

def test_triage_profile_carries_no_location_information(client):
    body = client.post("/v1/score/study", json=study_payload(profile="triage")).json()
    forbidden = {"surprise_map", "implicit_age_map", "boxes", "localization"}
    assert not forbidden & set(body)
    assert all(not forbidden & set(image) for image in body["images"])


def test_radiologist_profile_says_localization_is_unavailable(client):
    body = client.post("/v1/score/study", json=study_payload(profile="radiologist")).json()
    assert body["localization"] is None
    assert "null result" in body["localization_note"]


# --- preprocessing --------------------------------------------------------

def test_rgb_upload_is_accepted(client):
    array = (np.random.rand(400, 300, 3) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    payload = study_payload()
    payload["images"][0]["content"] = b64(buffer.getvalue())
    assert client.post("/v1/score/study", json=payload).status_code == 200


def test_preprocess_pads_to_the_canvas_and_reports_geometry():
    prepared = preprocess(np.random.rand(500, 250).astype(np.float32))
    assert prepared["image"].shape == (384, 384)
    assert prepared["image"].min() >= 0.0 and prepared["image"].max() <= 1.0
    assert prepared["geometry"]["new_h"] == 384
    assert prepared["geometry"]["pad_y"] == 0
    assert prepared["valid_mask"].shape == (24, 24)


def test_padding_stays_exactly_zero():
    prepared = preprocess(np.random.rand(500, 250).astype(np.float32) + 1.0)
    pad_x = prepared["geometry"]["pad_x"]
    assert prepared["image"][:, : pad_x - 1].max() == 0.0


def test_blank_image_does_not_divide_by_zero():
    prepared = preprocess(np.full((300, 300), 7.0, dtype=np.float32))
    assert np.isfinite(prepared["image"]).all()


def test_unreadable_bytes_raise():
    with pytest.raises(UnreadableImage):
        load_grayscale(b"nope")


def test_age_outside_the_range_raises(scorer):
    with pytest.raises(AgeRequired):
        scorer.band_of(21.0)
