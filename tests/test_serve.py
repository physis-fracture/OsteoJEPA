"""The API surface and its failure modes.

M6 is accepted on handling the failure path, so those are tested rather than
assumed. The model here is untrained; what is under test is the contract, not
the score.

Three routes exist and no more. `/v1/score/study` and `/v1/score/image` were
removed along with the request shape built around object keys, integer view
codes and a `profile` switch, so the tests that covered them are gone with them.
"""

import functools
import http.server
import io
import json
import threading

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from PIL import Image

from physis.models.classifier import build_classifier
from physis.serve import fetch
from physis.serve.api import build_app
from physis.serve.preprocess import (
    UnreadableImage,
    detect_preprocessed,
    load_grayscale,
    preprocess,
)
from physis.serve.scorer import AgeRequired, Scorer
from physis.utils.config import load_config


def png_bytes(height=500, width=380):
    array = (np.random.rand(height, width) * 60000).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def allow_loopback(monkeypatch):
    """The image server below is plain http on 127.0.0.1.

    Production refuses both. The escape hatch is loopback-only and is what
    serve_local.py sets, so exercising it here also covers the flag.
    """
    monkeypatch.setenv("PHYSIS_ALLOW_LOOPBACK_FETCH", "1")
    monkeypatch.delenv("PHYSIS_API_KEY", raising=False)
    monkeypatch.delenv("PHYSIS_IMAGE_HOSTS", raising=False)


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
    return TestClient(build_app(lambda: scorer), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """A throwaway http server standing in for presigned R2 URLs."""
    directory = tmp_path_factory.mktemp("images")
    for name in ("img0.png", "img1.png"):
        (directory / name).write_bytes(png_bytes())
    (directory / "notanimage.png").write_bytes(b"nope")

    rgb = (np.random.rand(400, 300, 3) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    (directory / "rgb.png").write_bytes(buffer.getvalue())

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def payload(images_url, **overrides):
    body = {
        "study_id": "0001_01",
        "age_years": 11.5,
        "sex": "male",
        "images": [
            {"image_id": "img0", "image_url": f"{images_url}/img0.png",
             "view": "PA", "laterality": "left"}
        ],
    }
    body.update(overrides)
    return body


# --- the API surface ------------------------------------------------------

def test_root_reports_the_service(client):
    body = client.get("/").json()
    assert body["service"] == "Physis triage inference"
    assert body["health"] == "/v1/health"


def test_health_is_public_and_thin(client):
    body = client.get("/v1/health").json()
    assert body == {"status": "ok", "contract_version": "1.1"}
    # Provenance belongs on the authenticated response. An unauthenticated
    # endpoint has no reason to name the checkpoint that is loaded.
    assert "model" not in body


def test_health_is_503_without_a_model():
    unloaded = TestClient(build_app(lambda: None))
    response = unloaded.get("/v1/health")
    assert response.status_code == 503
    assert response.json()["status"] == "model_unavailable"


def test_the_removed_endpoints_are_gone(client, images):
    for path in ("/v1/score/study", "/v1/score/image"):
        assert client.post(path, json=payload(images)).status_code == 404


# --- the happy path -------------------------------------------------------

def test_scores_a_study(client, images):
    response = client.post("/v1/predict", json=payload(images))
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["study_id"] == "0001_01"
    assert 0.0 <= data["triage_score"] <= 1.0
    # 0-100 here, not 0-1: the client formats it with a percent sign.
    assert 0.0 <= data["priority_percentile"] <= 100.0
    assert data["age_band"] == "11"
    assert isinstance(data["inference_time_ms"], int)
    assert data["model_version"].endswith("@v1")


def test_study_score_is_the_max_over_its_images(client, images):
    body = client.post("/v1/predict", json=payload(
        images,
        images=[
            {"image_id": "a", "image_url": f"{images}/img0.png"},
            {"image_id": "b", "image_url": f"{images}/img1.png"},
        ],
    )).json()
    data = body["data"]
    assert len(data["images"]) == 2
    assert data["triage_score"] == max(i["triage_score"] for i in data["images"])


def test_request_id_is_echoed(client, images):
    response = client.post(
        "/v1/predict", json=payload(images), headers={"X-Request-ID": "abc123"}
    )
    assert response.headers["X-Request-ID"] == "abc123"
    # Generated when the client does not supply one, so a call is always
    # traceable from the client's log into Modal's.
    assert client.post("/v1/predict", json=payload(images)).headers["X-Request-ID"]


def test_deprecated_osteojepa_fields_are_null_not_absent(client, images):
    image = client.post("/v1/predict", json=payload(images)).json()["data"]["images"][0]
    for field in ("implicit_age", "implicit_age_gap", "surprise_map", "implicit_age_map"):
        assert field in image and image[field] is None


def test_boxes_are_null_when_no_detector_is_loaded(client, images):
    """Null and empty differ: [] is "looked and found nothing", null is
    "this deployment cannot localize"."""
    image = client.post("/v1/predict", json=payload(images)).json()["data"]["images"][0]
    assert image["boxes"] is None


# --- localization ---------------------------------------------------------

@pytest.fixture(scope="module")
def client_with_detector(tmp_path_factory, scorer):
    from physis.models.detector import build_detector

    cfg = load_config("configs/base.yaml", ["detector.init=random"])
    tmp = tmp_path_factory.mktemp("detector")
    path = tmp / "det.pt"
    torch.save({"model": build_detector(cfg).state_dict()}, path)
    with_detector = Scorer(
        scorer.cfg,
        str(scorer._checkpoint_path),
        str(scorer._calibration_path),
        detector_checkpoint=str(path),
    )
    return TestClient(build_app(lambda: with_detector))


def test_boxes_are_a_list_when_a_detector_is_loaded(client_with_detector, images):
    data = client_with_detector.post("/v1/predict", json=payload(images)).json()["data"]
    assert data["model_version"].endswith("@v1.1")
    for image in data["images"]:
        assert isinstance(image["boxes"], list)
        for box in image["boxes"]:
            assert len(box) == 5
            assert box[0] <= box[2] and box[1] <= box[3]
            assert 0.0 <= box[4] <= 1.0


# --- authentication -------------------------------------------------------

@pytest.fixture
def guarded(scorer, monkeypatch):
    monkeypatch.setenv("PHYSIS_API_KEY", "s3cret")
    return TestClient(build_app(lambda: scorer))


def test_open_when_no_key_is_configured(client, images):
    assert client.post("/v1/predict", json=payload(images)).status_code == 200


def test_predict_needs_the_key(guarded, images):
    response = guarded.post("/v1/predict", json=payload(images))
    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["error_code"] == "UNAUTHORIZED"
    assert response.headers["WWW-Authenticate"] == "Bearer"

    ok = guarded.post(
        "/v1/predict", json=payload(images), headers={"Authorization": "Bearer s3cret"}
    )
    assert ok.status_code == 200


def test_a_wrong_or_malformed_key_is_refused(guarded, images):
    for header in ("Bearer wrong", "s3cret", "Basic s3cret", "Bearer "):
        response = guarded.post(
            "/v1/predict", json=payload(images), headers={"Authorization": header}
        )
        assert response.status_code == 401, header


def test_health_stays_open_when_a_key_is_set(guarded):
    assert guarded.get("/v1/health").status_code == 200
    assert guarded.get("/").status_code == 200


# --- validation -----------------------------------------------------------

@pytest.mark.parametrize("override,field", [
    ({"study_id": ""}, "study_id"),
    ({"age_years": 0.1}, "age_years"),
    ({"age_years": 25.0}, "age_years"),
    ({"images": []}, "images"),
    ({"sex": "other"}, "sex"),
])
def test_invalid_requests_are_422_in_the_envelope(client, images, override, field):
    response = client.post("/v1/predict", json=payload(images, **override))
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error_code"] == "VALIDATION_ERROR"
    # FastAPI's own {"detail": [...]} would read as a success with every field
    # missing, because the client branches on `success`.
    assert any(item["field"] == field for item in body["errors"]), body["errors"]


@pytest.mark.parametrize("missing", ["study_id", "age_years", "images"])
def test_required_fields_have_no_default(client, images, missing):
    body = payload(images)
    del body[missing]
    assert client.post("/v1/predict", json=body).status_code == 422


def test_an_invalid_enum_is_refused(client, images):
    body = payload(images)
    body["images"][0]["view"] = "OBLIQUE"
    assert client.post("/v1/predict", json=body).status_code == 422


def test_more_images_than_the_ceiling_are_refused(client, images):
    body = payload(images, images=[
        {"image_id": f"i{n}", "image_url": f"{images}/img0.png"} for n in range(9)
    ])
    assert client.post("/v1/predict", json=body).status_code == 422


def test_no_model_loaded_is_503(images):
    unloaded = TestClient(build_app(lambda: None))
    response = unloaded.post("/v1/predict", json=payload(images))
    assert response.status_code == 503
    assert response.json()["error_code"] == "SERVICE_UNAVAILABLE"


def test_unfetchable_url_is_404(client, images):
    body = payload(images)
    body["images"][0]["image_url"] = f"{images}/absent.png"
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 404
    assert response.json()["error_code"] == "IMAGE_NOT_FOUND"


def test_fetched_but_undecodable_is_415(client, images):
    body = payload(images)
    body["images"][0]["image_url"] = f"{images}/notanimage.png"
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 415
    assert response.json()["error_code"] == "UNREADABLE_IMAGE"


def test_the_deadline_produces_504(client, images, monkeypatch):
    monkeypatch.setenv("PHYSIS_DEADLINE_MS", "0")
    response = client.post("/v1/predict", json=payload(images))
    assert response.status_code == 504
    assert response.json()["error_code"] == "INFERENCE_TIMEOUT"


def test_an_unexpected_error_never_leaks_internals(images):
    def explode():
        raise RuntimeError("/secret/path/to/checkpoint.pt")

    broken = TestClient(build_app(explode), raise_server_exceptions=False)
    response = broken.post("/v1/predict", json=payload(images))
    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "INTERNAL_ERROR"
    assert "secret" not in json.dumps(body)


# --- SSRF -----------------------------------------------------------------
#
# The service fetches a URL a caller supplies, from inside a network the caller
# cannot reach. Authentication narrows who can ask; it does not make the request
# safe.

@pytest.mark.parametrize("url", [
    "http://example.com/x.png",            # plain http
    "https://127.0.0.1/x.png",             # loopback
    "https://localhost/x.png",
    "https://169.254.169.254/latest/meta",  # cloud metadata
    "https://10.0.0.5/x.png",              # private
    "https://192.168.1.1/x.png",
    "https://[::1]/x.png",
])
def test_dangerous_urls_are_refused(url, monkeypatch):
    monkeypatch.delenv("PHYSIS_ALLOW_LOOPBACK_FETCH", raising=False)
    with pytest.raises(fetch.ImageFetchError):
        fetch.check_url(url)


def test_the_production_allowlist_admits_r2_and_nothing_beside_it(monkeypatch):
    """The host the web application actually signs its uploads through.

    An allowlist that quietly stopped matching would fail open only in the sense
    that every real upload would start returning 422, so this pins the shape of
    a presigned R2 URL against the entry that admits it.
    """
    monkeypatch.delenv("PHYSIS_ALLOW_LOOPBACK_FETCH", raising=False)
    host = "2b3bf3b4058f753954a9b0d4cc31de54.r2.cloudflarestorage.com"
    fetch.check_url(
        f"https://{host}/physis-uploads/study.png?X-Amz-Signature=abc", allowlist=(host,)
    )
    for refused in (
        f"http://{host}/x.png",                      # plain http
        "https://other.r2.cloudflarestorage.com/x.png",  # a different account
        f"https://{host}.evil.net/x.png",            # suffix confusion
    ):
        with pytest.raises(fetch.ImageFetchError):
            fetch.check_url(refused, allowlist=(host,))


def test_the_allowlist_matches_hosts_and_subdomains_only(monkeypatch):
    monkeypatch.delenv("PHYSIS_ALLOW_LOOPBACK_FETCH", raising=False)
    allow = ("r2.dev",)
    fetch.check_url("https://bucket.r2.dev/x.png", allowlist=allow)
    # A bare endswith would wave this through.
    with pytest.raises(fetch.ImageHostRejected):
        fetch.check_url("https://evil-r2.dev/x.png", allowlist=allow)
    with pytest.raises(fetch.ImageHostRejected):
        fetch.check_url("https://example.com/x.png", allowlist=allow)


def test_a_rejected_host_reaches_the_client_as_422(client, images, monkeypatch):
    monkeypatch.setenv("PHYSIS_IMAGE_HOSTS", "r2.example")
    monkeypatch.delenv("PHYSIS_ALLOW_LOOPBACK_FETCH", raising=False)
    body = payload(images)
    body["images"][0]["image_url"] = "https://example.com/x.png"
    response = client.post("/v1/predict", json=body)
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "image_url"


def test_the_byte_ceiling_is_enforced(images):
    with pytest.raises(fetch.ImageTooLarge):
        fetch.fetch_image(f"{images}/img0.png", max_bytes=64)


def test_a_presigned_url_is_never_logged_whole():
    url = "https://bucket.r2.dev/uploads/x.png?X-Amz-Signature=deadbeef&X-Amz-Expires=900"
    redacted = fetch.redact(url)
    assert "Signature" not in redacted and "deadbeef" not in redacted
    # The path identifies the object and is safe to keep.
    assert redacted == "https://bucket.r2.dev/uploads/x.png"
    # Port survives, credentials in a userinfo prefix do not.
    assert fetch.redact("https://u:pw@host:8443/a.png") == "https://host:8443/a.png"


def test_loopback_escape_hatch_is_loopback_only(monkeypatch):
    monkeypatch.setenv("PHYSIS_ALLOW_LOOPBACK_FETCH", "1")
    fetch.check_url("http://127.0.0.1:8000/x.png")
    # The addresses an SSRF attempt actually wants stay blocked.
    with pytest.raises(fetch.ImageHostRejected):
        fetch.check_url("http://169.254.169.254/latest/meta")
    with pytest.raises(fetch.ImageHostRejected):
        fetch.check_url("http://10.0.0.5/x.png")


# --- OpenAPI --------------------------------------------------------------

def test_openapi_describes_exactly_the_runtime(client):
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/", "/v1/health", "/v1/predict"}

    predict = spec["paths"]["/v1/predict"]["post"]
    assert predict["security"], "bearer auth missing from the security scheme"
    assert set(predict["responses"]) >= {
        "200", "401", "404", "413", "415", "422", "429", "500", "503", "504"
    }
    # `"schema": {}` for a 200 is the defect this replaces.
    assert predict["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]

    request = spec["components"]["schemas"]["PredictRequest"]
    assert set(request["required"]) == {"study_id", "age_years", "images"}
    assert request["properties"]["age_years"]["minimum"] == 0.2
    assert request["properties"]["age_years"]["maximum"] == 19.0


def test_no_cors_header_is_advertised(client, images):
    """Server to server. A browser is not a client, so the header is surface
    with no user."""
    response = client.post(
        "/v1/predict",
        json=payload(images),
        headers={"Origin": "https://physis.example"},
    )
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


# --- preprocessing --------------------------------------------------------

def test_rgb_upload_is_accepted(client, images):
    """A PACS export re-saved as RGB PNG is a normal thing to receive."""
    body = payload(images)
    body["images"][0]["image_url"] = f"{images}/rgb.png"
    assert client.post("/v1/predict", json=body).status_code == 200


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


# --- already-preprocessed uploads ------------------------------------------
#
# The dataset ships 384x384 canvases that have been through this pipeline
# already. Running it again would recompute the percentile clip over the padding
# and, far worse, read pad_x and pad_y as zero, marking all 576 patches valid.


def padded_canvas(new_w, new_h, value=0.7):
    canvas = np.zeros((384, 384), dtype=np.float32)
    pad_x, pad_y = (384 - new_w) // 2, (384 - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = value
    return canvas, pad_x, pad_y


@pytest.mark.parametrize("new_w,new_h", [(255, 384), (384, 261), (150, 384)])
def test_preprocessed_canvas_is_detected_with_its_geometry(new_w, new_h):
    canvas, pad_x, pad_y = padded_canvas(new_w, new_h)
    found = detect_preprocessed(canvas)
    assert found is not None
    assert (found["pad_x"], found["pad_y"]) == (pad_x, pad_y)
    assert (found["new_w"], found["new_h"]) == (new_w, new_h)


def test_preprocessed_upload_keeps_padding_out_of_the_mask():
    canvas, pad_x, _ = padded_canvas(255, 384)
    prepared = preprocess(canvas)
    assert prepared["geometry"]["already_preprocessed"] is True
    mask = prepared["valid_mask"]
    first_valid_column = int(np.flatnonzero(mask.any(axis=0))[0])
    assert first_valid_column * 16 >= pad_x


def test_a_real_upload_is_not_mistaken_for_a_preprocessed_one():
    assert detect_preprocessed(np.random.rand(500, 380).astype(np.float32)) is None
    assert detect_preprocessed(np.full((384, 384), 0.5, dtype=np.float32)) is None
    prepared = preprocess(np.full((384, 384), 0.5, dtype=np.float32))
    assert prepared["geometry"]["pad_x"] == 0
    assert prepared["valid_mask"].all()


def test_detection_is_never_wider_than_the_truth():
    """Losing an edge patch is acceptable; admitting a padding patch is not."""
    for new_w in (150, 255, 300, 384):
        canvas, pad_x, pad_y = padded_canvas(new_w, 384)
        canvas[:, pad_x] = 0.0
        found = detect_preprocessed(canvas)
        if found is None:
            continue
        assert found["pad_x"] >= pad_x
        assert found["pad_x"] + found["new_w"] <= pad_x + new_w


def test_percentile_never_leaves_the_unit_interval():
    """A score above every reference value is the 100th percentile, not the 101st."""
    from physis.serve.calibration import percentile_for

    entry = {"quantiles": np.linspace(-10, 10, 101).tolist()}
    assert percentile_for(1e6, entry) == 1.0
    assert percentile_for(-1e6, entry) == 0.0
    assert 0.0 <= percentile_for(0.0, entry) <= 1.0
