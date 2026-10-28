"""Tests for the super-resolution web application."""
import asyncio
import io
from pathlib import Path
import sys

from PIL import Image
import pytest
from fastapi import HTTPException, UploadFile

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.main import discover_eligible_models, run_inference


def test_interpolation_methods_are_available_without_checkpoints(tmp_path):
    models = discover_eligible_models(tmp_path)

    assert len(models) == 8
    assert {model["model_name"] for model in models} == {
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos",
    }
    assert {model["scale"] for model in models} == {2, 4}
    assert all(model["type"] == "interpolation" for model in models)


def test_lr_inference_runs_interpolation_methods_at_their_scales(tmp_path, monkeypatch):
    models = discover_eligible_models(tmp_path)
    monkeypatch.setattr("app.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("app.main.discover_eligible_models", lambda _: models)

    image_buffer = io.BytesIO()
    Image.new("RGB", (6, 5), color="red").save(image_buffer, format="PNG")
    uploaded_file = UploadFile(
        filename="input.png", file=io.BytesIO(image_buffer.getvalue())
    )

    call_res = run_inference(
        file=uploaded_file,
        mode="lr_input",
        selected_models="interpolation_nearest_x2,interpolation_bilinear_x4",
        device_request="cpu",
        degradation_type="bicubic",
        noise_sigma=15.0,
        ref_file=None,
    )
    result = asyncio.run(call_res) if asyncio.iscoroutine(call_res) else call_res

    assert [(item["width"], item["height"]) for item in result["results"]] == [
        (12, 10),
        (24, 20),
    ]
    assert [item["model_name"] for item in result["results"]] == [
        "nearest",
        "bilinear",
    ]


def test_interpolation_methods_are_rejected_for_hr_reference_mode(tmp_path, monkeypatch):
    models = discover_eligible_models(tmp_path)
    monkeypatch.setattr("app.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("app.main.discover_eligible_models", lambda _: models)

    with pytest.raises(HTTPException) as exc_info:
        run_inference(
            file=UploadFile(filename="input.png", file=io.BytesIO()),
            mode="hr_reference",
            selected_models="interpolation_nearest_x2",
            device_request="cpu",
            degradation_type="bicubic",
            noise_sigma=15.0,
            ref_file=None,
        )
    assert exc_info.value.detail == (
        "Standard interpolation methods are available only in LR input mode."
    )


def test_get_benchmark_summary_endpoint():
    from app.main import get_benchmark_summary
    data = get_benchmark_summary()
    assert "records" in data
    assert len(data["records"]) > 0
    first = data["records"][0]
    assert "model" in first
    assert "mean_psnr" in first
    assert "mean_ssim" in first


def test_get_analytics_endpoint():
    from app.main import get_analytics
    html_content = get_analytics()
    assert isinstance(html_content, str)
    assert len(html_content) > 0
