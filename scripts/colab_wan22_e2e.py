#!/usr/bin/env python3
"""Wan2.2 eval + short GPU generate, meant for Google Colab.

Reads HF tokens from the process environment or Colab Secrets
(HF_TOKEN / HUGGINGFACE_TOKEN). Never prints the token, never writes it
into the repo, never puts it on a command line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("/tmp/wan22_colab_e2e_report.json")


def _token_from_colab() -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return None
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def _install_token() -> bool:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _token_from_colab()
    )
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"huggingface login skipped: {type(exc).__name__}")
    return True


def _gpu_info() -> dict:
    info: dict = {"cuda_available": False}
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception as exc:
        info["torch_error"] = type(exc).__name__
    return info


def _run_pytest() -> dict:
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    tests = [
        str(ROOT / "tests" / "test_wan22_upgrade.py"),
        str(ROOT / "tests" / "test_wan22_eval_cli_e2e.py"),
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q", "--tb=line"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _short_generate() -> dict:
    import asyncio

    from visionflow.services.generation.wan_video_service import WanVideoGenerationService
    from visionflow.shared.models import VideoGenerationRequest, WanTask

    os.environ.setdefault("WAN_MAX_VIDEO_DURATION", "2")
    os.environ.setdefault("WAN_GPU_MEMORY_FRACTION", "0.85")

    request = VideoGenerationRequest(
        prompt="A red cat walks across a wooden table, natural light",
        model_key="ti2v-5B",
        task=WanTask.TI2V_5B,
        duration=1,
        resolution="1280x704",
        fps=24,
        quality="medium",
        guidance_scale=5.0,
        num_inference_steps=4,
        seed=42,
        use_prompt_extend=False,
        sample_solver="unipc",
    )

    async def _go() -> dict:
        service = WanVideoGenerationService()
        return await service.generate_video(request)

    result = asyncio.run(_go())
    safe = {
        "status": result.get("status"),
        "error": result.get("error"),
        "video_path": result.get("video_path"),
        "model_used": result.get("model_used"),
        "task": result.get("task"),
        "model_key": result.get("model_key"),
        "hf_revision": result.get("hf_revision"),
        "upstream_git_sha": result.get("upstream_git_sha"),
        "seed": result.get("seed"),
        "sample_steps": result.get("sample_steps"),
        "generation_time": result.get("generation_time"),
        "metadata_path": result.get("metadata_path"),
    }
    video = result.get("video_path")
    if video:
        path = Path(video)
        safe["video_exists"] = path.is_file()
        safe["video_bytes"] = path.stat().st_size if path.is_file() else 0
    return safe


def main() -> int:
    report: dict = {"hf_token_present": False, "gpu": {}, "pytest": {}, "generate": {}}
    try:
        sys.path.insert(0, str(ROOT / "src"))
        report["hf_token_present"] = _install_token()
        report["gpu"] = _gpu_info()
        print("hf_token_present", report["hf_token_present"])
        print("gpu", json.dumps(report["gpu"]))
        report["pytest"] = _run_pytest()
        print("pytest_returncode", report["pytest"]["returncode"])
        print(report["pytest"]["stdout_tail"])
        if report["gpu"].get("cuda_available"):
            report["generate"] = _short_generate()
        else:
            report["generate"] = {"status": "skipped", "error": "no CUDA device"}
        print("generate", json.dumps(report["generate"], default=str))
    except Exception:
        report["crash"] = traceback.format_exc()
        print(report["crash"])
        REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        return 1
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("report", REPORT)
    generate_status = report.get("generate", {}).get("status")
    if report["pytest"]["returncode"] != 0:
        return 2
    if generate_status not in {"completed", "skipped"}:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
