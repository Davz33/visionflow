"""Thin mapping of Wan2.2 generate.py onto this eval harness.

Flag names match upstream generate.py (Wan-Video/Wan2.2, SHA
42bf4cfaa384bc21833865abc2f9e6c0e67233dc, 2026-03-17). This module does
not import the Wan tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .wan_model_zoo import (
    DEFAULT_TASK,
    HF_REVISION_DEFAULT,
    SAMPLE_SOLVERS,
    SUPPORTED_SIZES,
    WAN22_GIT_SHA,
    get_model,
    normalize_size,
)


@dataclass
class WanGenerateRequest:
    """Subset of generate.py arguments that an eval run actually needs."""

    task: str = DEFAULT_TASK
    size: str = "1280*704"
    prompt: str = ""
    ckpt_dir: Optional[str] = None
    image: Optional[str] = None
    audio: Optional[str] = None
    pose_video: Optional[str] = None
    src_root_path: Optional[str] = None
    use_prompt_extend: bool = False
    prompt_extend_method: str = "local_qwen"
    prompt_extend_model: Optional[str] = None
    prompt_extend_target_lang: str = "en"
    sample_solver: str = "unipc"
    sample_steps: Optional[int] = None
    sample_shift: Optional[float] = None
    sample_guide_scale: Optional[float] = None
    frame_num: Optional[int] = None
    base_seed: int = -1
    offload_model: bool = True
    convert_model_dtype: bool = True
    t5_cpu: bool = False
    enable_tts: bool = False
    replace_flag: bool = False
    distilled: bool = False
    model_key: Optional[str] = None

    def resolved_model_key(self) -> str:
        if self.model_key:
            return self.model_key
        if self.distilled and self.task == "animate-2-14B":
            return "animate-2-14B-distilled"
        return self.task


def _require_file(label: str, path_str: Optional[str]) -> None:
    if not path_str:
        raise ValueError(f"{label} is required for this task")
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_generate_request(
    req: WanGenerateRequest, *, check_files: bool = False
) -> WanGenerateRequest:
    """Apply the same checks generate.py _validate_args performs.

    Set check_files=True before a real generate so missing inputs fail early.
    """
    spec = get_model(req.resolved_model_key())
    size = normalize_size(req.size)
    allowed = SUPPORTED_SIZES.get(spec.task)
    if allowed and size not in allowed:
        raise ValueError(
            f"size {size} is not supported for task {spec.task}; "
            f"allowed: {', '.join(allowed)}"
        )
    if req.sample_solver not in SAMPLE_SOLVERS:
        raise ValueError(
            f"sample_solver {req.sample_solver!r} not in {SAMPLE_SOLVERS}"
        )
    if spec.task == "i2v-A14B" and not req.image:
        raise ValueError("i2v-A14B requires --image")
    if spec.task == "s2v-14B" and not req.enable_tts and not req.audio:
        raise ValueError("s2v-14B requires --audio unless --enable_tts is set")
    for needed in spec.requires:
        if needed == "image" and spec.task == "ti2v-5B":
            continue
        if needed == "image" and not req.image:
            raise ValueError(f"{spec.task} requires an image path")
        if needed == "audio" and not req.audio and not req.enable_tts:
            raise ValueError(f"{spec.task} requires an audio path")
        if needed == "pose_video" and not req.pose_video and not req.src_root_path:
            raise ValueError(f"{spec.task} requires pose_video or src_root_path")
    if check_files:
        if req.image:
            _require_file("image", req.image)
        if req.audio:
            _require_file("audio", req.audio)
        if req.pose_video:
            _require_file("pose_video", req.pose_video)
    if req.sample_steps is None:
        req.sample_steps = spec.sample_steps
    if req.sample_shift is None:
        req.sample_shift = spec.sample_shift
    if req.sample_guide_scale is None:
        req.sample_guide_scale = spec.sample_guide_scale
    if req.frame_num is None:
        req.frame_num = spec.frame_num
    if req.ckpt_dir is None:
        req.ckpt_dir = spec.hf_id
    req.size = size
    return req


def build_pipeline_kwargs(req: WanGenerateRequest) -> Dict[str, Any]:
    """Keyword args for Diffusers WanPipeline / WanImageToVideoPipeline."""
    req = validate_generate_request(req)
    spec = get_model(req.resolved_model_key())
    width, height = (int(p) for p in req.size.split("*"))
    kwargs: Dict[str, Any] = {
        "prompt": req.prompt,
        "height": height,
        "width": width,
        "num_frames": req.frame_num,
        "guidance_scale": req.sample_guide_scale,
        "num_inference_steps": req.sample_steps,
    }
    if spec.task.startswith("i2v") or (spec.task == "ti2v-5B" and req.image):
        kwargs["image"] = req.image
    return kwargs


def result_identity(req: WanGenerateRequest, seed: int) -> Dict[str, Any]:
    """Fields written into generation result JSON for later scoring."""
    spec = get_model(req.resolved_model_key())
    return {
        "task": spec.task,
        "model_key": spec.key,
        "model_used": spec.hf_id,
        "model_revision": spec.hf_revision,
        "hf_revision": spec.hf_revision or HF_REVISION_DEFAULT,
        "upstream_git_sha": spec.upstream_git_sha or WAN22_GIT_SHA,
        "upstream_repo": spec.github_repo,
        "family": spec.family,
        "size_class": spec.size_class,
        "distilled": spec.distilled,
        "legacy": spec.legacy,
        "seed": seed,
        "use_prompt_extend": req.use_prompt_extend,
        "prompt_extend_method": req.prompt_extend_method if req.use_prompt_extend else None,
        "sample_solver": req.sample_solver,
        "sample_steps": req.sample_steps,
        "sample_shift": req.sample_shift,
        "sample_guide_scale": req.sample_guide_scale,
        "size": req.size,
        "frame_num": req.frame_num,
        "fps": spec.sample_fps,
    }


def prompt_extend_ablation_flags() -> List[Dict[str, Any]]:
    """On/off pairs for the official --use_prompt_extend eval axis."""
    return [
        {"use_prompt_extend": False, "prompt_extend_method": None},
        {
            "use_prompt_extend": True,
            "prompt_extend_method": "local_qwen",
            "prompt_extend_target_lang": "en",
        },
    ]


def few_step_ablation(task: str) -> List[Dict[str, Any]]:
    """Official sample_steps plus a short-run setting.

    Animate-2 ships a distilled Diffusers checkpoint (4 steps). Other
    tasks keep generate.py defaults and a halved-step probe.
    """
    spec = get_model(task if task != "animate-2-14B" else "animate-2-14B")
    rows = [
        {
            "label": "default",
            "sample_solver": "unipc",
            "sample_steps": spec.sample_steps,
            "distilled": False,
        },
        {
            "label": "dpm++",
            "sample_solver": "dpm++",
            "sample_steps": spec.sample_steps,
            "distilled": False,
        },
        {
            "label": "few-step",
            "sample_solver": "unipc",
            "sample_steps": max(4, spec.sample_steps // 5),
            "distilled": False,
        },
    ]
    if task == "animate-2-14B":
        distilled = get_model("animate-2-14B-distilled")
        rows.append(
            {
                "label": "official-distilled",
                "sample_solver": "unipc",
                "sample_steps": distilled.sample_steps,
                "distilled": True,
                "model_key": distilled.key,
            }
        )
    return rows


def as_generate_argv(req: WanGenerateRequest) -> List[str]:
    """Reproduce the upstream CLI for a dry-run or log line."""
    req = validate_generate_request(req)
    argv = [
        "python",
        "generate.py",
        "--task",
        req.task,
        "--size",
        req.size,
        "--ckpt_dir",
        str(req.ckpt_dir),
        "--prompt",
        req.prompt,
        "--sample_solver",
        req.sample_solver,
        "--sample_steps",
        str(req.sample_steps),
        "--base_seed",
        str(req.base_seed),
    ]
    if req.offload_model:
        argv.extend(["--offload_model", "True"])
    if req.convert_model_dtype:
        argv.append("--convert_model_dtype")
    if req.t5_cpu:
        argv.append("--t5_cpu")
    if req.use_prompt_extend:
        argv.extend(
            [
                "--use_prompt_extend",
                "--prompt_extend_method",
                req.prompt_extend_method,
                "--prompt_extend_target_lang",
                req.prompt_extend_target_lang,
            ]
        )
    if req.image:
        argv.extend(["--image", req.image])
    if req.audio:
        argv.extend(["--audio", req.audio])
    if req.pose_video:
        argv.extend(["--pose_video", req.pose_video])
    if req.enable_tts:
        argv.append("--enable_tts")
    if req.replace_flag:
        argv.append("--replace_flag")
    return argv


def request_to_dict(req: WanGenerateRequest) -> Dict[str, Any]:
    return asdict(req)
