"""FastAPI web server for Image Super-Resolution Benchmark GUI."""
import base64
import io
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
import torch

from app.services import (
    apply_degradation_to_pil,
    compute_metrics,
    create_export_zip,
    discover_eligible_models,
    generate_diff_map,
    load_model_from_checkpoint,
    pil_to_tensor,
    tensor_to_pil,
)
from src.models import MODELS
from src.utils.device import get_device

logger = logging.getLogger("sr_project.app")
logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).parent.parent.resolve()

app = FastAPI(title="Image Super-Resolution Benchmark UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and sessions cache
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

SESSIONS_CACHE: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Super-Resolution Benchmark UI</h1><p>Frontend static/index.html not found.</p>"


@app.get("/analytics", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_analytics():
    analytics_file = static_dir / "analytics.html"
    if analytics_file.exists():
        with open(analytics_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Super-Resolution Benchmark Analytics</h1><p>Frontend static/analytics.html not found.</p>"


@app.get("/api/benchmark-summary")
def get_benchmark_summary():
    """Returns aggregated benchmark metrics across all models and datasets from summary.csv."""
    import csv

    summary_path = REPO_ROOT / "results" / "summary.csv"
    if not summary_path.exists():
        return {"records": []}

    records = []
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "experiment": row["experiment"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "scale": int(row["scale"]) if row.get("scale") else 2,
                    "degradation": row.get("degradation", ""),
                    "noise_sigma": float(row["noise_sigma"]) if row.get("noise_sigma") else 0.0,
                    "n_images": int(row["n_images"]) if row.get("n_images") else 0,
                    "param_count": int(row["param_count"]) if row.get("param_count") else 0,
                    "mean_inference_time_s": float(row["mean_inference_time_s"]) if row.get("mean_inference_time_s") else 0.0,
                    "mean_psnr": float(row["mean_psnr"]) if row.get("mean_psnr") else 0.0,
                    "mean_ssim": float(row["mean_ssim"]) if row.get("mean_ssim") else 0.0,
                })
            except Exception:
                continue

    return {"records": records}


@app.get("/api/models")
def get_models():
    """Returns discovered eligible neural models and environment capabilities."""
    models = discover_eligible_models(REPO_ROOT)

    lpips_available = False
    try:
        import lpips
        lpips_available = True
    except ImportError:
        lpips_available = False

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    return {
        "models": models,
        "lpips_available": lpips_available,
        "cuda_available": cuda_available,
        "device_name": device_name,
    }


def _process_image_upload(file_bytes: bytes, filename: str) -> Image.Image:
    """Validates and loads uploaded image bytes into RGB PIL Image."""
    if len(file_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 20MB limit.")

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        # Handle EXIF orientation if present
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_rgb = pil_img.convert("RGB")
        return pil_rgb
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to decode image '{filename}': {str(e)}"
        )


def _pil_to_base64_data_uri(pil_img: Image.Image, format_str: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=format_str)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format_str.lower()};base64,{b64_str}"


@app.post("/api/run")
def run_inference(
    file: UploadFile = File(...),
    mode: str = Form("lr_input"),  # 'lr_input' | 'hr_reference'
    selected_models: str = Form(...),  # Comma-separated model IDs
    device_request: str = Form("auto"),  # 'auto' | 'cpu' | 'cuda'
    degradation_type: str = Form("bicubic"),  # 'bicubic' | 'bicubic_noise'
    noise_sigma: float = Form(15.0),
    ref_file: Optional[UploadFile] = File(None),
):
    """Executes super-resolution inference on uploaded image across selected models."""
    t0_total = time.time()

    model_ids = [m.strip() for m in selected_models.split(",") if m.strip()]
    if not model_ids:
        raise HTTPException(status_code=400, detail="No models selected for inference.")

    all_eligible = {m["id"]: m for m in discover_eligible_models(REPO_ROOT)}
    missing_ids = [mid for mid in model_ids if mid not in all_eligible]
    if missing_ids:
        raise HTTPException(
            status_code=400, detail=f"Requested models not eligible or found: {missing_ids}"
        )
    if mode == "hr_reference" and any(
        all_eligible[mid]["type"] == "interpolation" for mid in model_ids
    ):
        raise HTTPException(
            status_code=400,
            detail="Standard interpolation methods are available only in LR input mode.",
        )

    # 1. Read input image (synchronously in threadpool)
    input_bytes = file.file.read()
    input_pil = _process_image_upload(input_bytes, file.filename)
    orig_w, orig_h = input_pil.size

    # Check separate HR reference upload if supplied in LR mode
    sep_hr_pil: Optional[Image.Image] = None
    if mode == "lr_input" and ref_file is not None and ref_file.filename:
        ref_bytes = ref_file.file.read()
        if len(ref_bytes) > 0:
            sep_hr_pil = _process_image_upload(ref_bytes, ref_file.filename)

    # Resolve target device
    device = get_device(device_request)
    actual_device_str = str(device)
    if device.type == "cuda":
        actual_device_str += f" ({torch.cuda.get_device_name(device)})"

    # Determine LR input and ground truth HR reference depending on mode
    model_results = []
    session_id = str(uuid.uuid4())

    # We determine global scale from selected models
    max_scale = max(all_eligible[mid]["scale"] for mid in model_ids)

    if mode == "hr_reference":
        add_noise = (degradation_type == "bicubic_noise")
        try:
            lr_pil, hr_cropped_pil = apply_degradation_to_pil(
                input_pil, scale=max_scale, add_noise=add_noise, noise_sigma=noise_sigma
            )
            ground_truth_hr_pil = hr_cropped_pil
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Degradation failed: {e}")
    else:  # lr_input mode
        lr_pil = input_pil
        ground_truth_hr_pil = sep_hr_pil

    lr_tensor = pil_to_tensor(lr_pil).unsqueeze(0).to(device)
    gt_hr_tensor = (
        pil_to_tensor(ground_truth_hr_pil).unsqueeze(0)
        if ground_truth_hr_pil is not None
        else None
    )

    # Process each selected model
    for mid in model_ids:
        minfo = all_eligible[mid]
        m_label = minfo["label"]
        scale = minfo["scale"]

        try:
            if minfo["type"] == "interpolation":
                model = MODELS.build(minfo["model_name"], scale=scale).to(device)
                model.eval()
            else:
                model = load_model_from_checkpoint(
                    minfo, device=device, repo_root=REPO_ROOT
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0_inf = time.time()

            with torch.no_grad():
                sr_tensor = model(lr_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_time = time.time() - t0_inf

            sr_pil = tensor_to_pil(sr_tensor)
            sr_w, sr_h = sr_pil.size

            # Compute metrics if GT reference is available
            metrics = {}
            diff_pil = None
            if gt_hr_tensor is not None:
                # Check scale alignment
                expected_hr_w = lr_pil.width * scale
                expected_hr_h = lr_pil.height * scale

                if (
                    gt_hr_tensor.shape[2] == expected_hr_h
                    and gt_hr_tensor.shape[3] == expected_hr_w
                ):
                    metrics = compute_metrics(
                        sr_tensor.cpu(), gt_hr_tensor, scale=scale
                    )
                    diff_pil = generate_diff_map(sr_pil, ground_truth_hr_pil)
                else:
                    logger.warning(
                        f"Mismatched reference size for model '{m_label}': expected ({expected_hr_w}x{expected_hr_h}), got ({gt_hr_tensor.shape[3]}x{gt_hr_tensor.shape[2]}). Metrics skipped."
                    )

            model_results.append(
                {
                    "model_id": mid,
                    "model_label": m_label,
                    "model_name": minfo["model_name"],
                    "scale": scale,
                    "degradation": minfo["degradation"],
                    "param_count": minfo["param_count"],
                    "width": sr_w,
                    "height": sr_h,
                    "inference_time_s": float(inf_time),
                    "metrics": metrics,
                    "sr_pil": sr_pil,
                    "diff_pil": diff_pil,
                    "sr_b64": _pil_to_base64_data_uri(sr_pil),
                    "diff_b64": _pil_to_base64_data_uri(diff_pil) if diff_pil else None,
                }
            )

        except Exception as e:
            logger.error(f"Inference error for model '{m_label}': {e}")
            model_results.append(
                {
                    "model_id": mid,
                    "model_label": m_label,
                    "error": str(e),
                }
            )

    elapsed_total = time.time() - t0_total

    # Prepare response payload
    response_data = {
        "session_id": session_id,
        "mode": mode,
        "degradation_type": degradation_type if mode == "hr_reference" else "none",
        "device_used": actual_device_str,
        "input_width": lr_pil.width,
        "input_height": lr_pil.height,
        "has_reference": (ground_truth_hr_pil is not None),
        "lr_b64": _pil_to_base64_data_uri(lr_pil),
        "hr_b64": _pil_to_base64_data_uri(ground_truth_hr_pil) if ground_truth_hr_pil else None,
        "total_elapsed_s": elapsed_total,
        "results": [
            {k: v for k, v in res.items() if k not in ("sr_pil", "diff_pil")}
            for res in model_results
        ],
    }

    # Store PIL images in session cache for ZIP export
    SESSIONS_CACHE[session_id] = {
        "input_pil": lr_pil,
        "hr_pil": ground_truth_hr_pil,
        "results": model_results,
        "summary": response_data,
    }

    return response_data


@app.get("/api/download/{session_id}")
def download_export_zip(session_id: str):
    """Generates and downloads a ZIP bundle of session outputs."""
    if session_id not in SESSIONS_CACHE:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    session_data = SESSIONS_CACHE[session_id]
    input_pil = session_data["input_pil"]
    results = [r for r in session_data["results"] if "error" not in r]

    zip_bytes = create_export_zip(input_pil, results, session_data["summary"])

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=sr_benchmark_{session_id[:8]}.zip"},
    )
