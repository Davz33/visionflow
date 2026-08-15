"""Official Wan open-weight pins as of 2026-08-15.

Sources (original dates, not download dates):
- Wan2.2 README / generate.py: https://github.com/Wan-Video/Wan2.2
  last main commit 42bf4cfaa384bc21833865abc2f9e6c0e67233dc (2026-03-17)
- HuggingFace Wan-AI collection Wan2.2, updated ~2026-08-06
- Wan-Animate-2 main 3ad2fef7d61d6200c9c653e0fe47be7616b323f3 (2026-08-08)

Wan 2.5 / 2.6 / 3.0 exist on Alibaba Cloud Model Studio as API products.
They have no official downloadable weights on Wan-Video GitHub or Wan-AI HF.
This harness pins open weights only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# Git SHAs verified 2026-08-15 via GitHub API.
WAN22_GITHUB = "https://github.com/Wan-Video/Wan2.2"
WAN22_GIT_SHA = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN22_GIT_DATE = "2026-03-17T10:48:41Z"

ANIMATE2_GITHUB = "https://github.com/Wan-Video/Wan-Animate-2"
ANIMATE2_GIT_SHA = "3ad2fef7d61d6200c9c653e0fe47be7616b323f3"
ANIMATE2_GIT_DATE = "2026-08-08T06:27:43Z"

WAN21_GITHUB = "https://github.com/Wan-Video/Wan2.1"

HF_REVISION_DEFAULT = "main"

# Official generate.py --task values (Wan2.2 WAN_CONFIGS).
OFFICIAL_TASKS = (
    "t2v-A14B",
    "i2v-A14B",
    "ti2v-5B",
    "s2v-14B",
    "animate-14B",
)

# Extra open-weight task from Wan-Animate-2 (HF collection, Aug 2026).
EXTENDED_TASKS = OFFICIAL_TASKS + ("animate-2-14B",)

# generate.py --sample_solver choices.
SAMPLE_SOLVERS = ("unipc", "dpm++")

# Official SIZE_CONFIGS keys use width*height.
SIZE_CONFIGS: Dict[str, Tuple[int, int]] = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
    "1024*704": (1024, 704),
    "704*1024": (704, 1024),
}

SUPPORTED_SIZES: Dict[str, Tuple[str, ...]] = {
    "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "ti2v-5B": ("704*1280", "1280*704"),
    "s2v-14B": (
        "720*1280",
        "1280*720",
        "480*832",
        "832*480",
        "1024*704",
        "704*1024",
        "704*1280",
        "1280*704",
    ),
    "animate-14B": ("720*1280", "1280*720"),
    "animate-2-14B": ("720*1280", "1280*720"),
}

# Consumer default: TI2V-5B 720P@24fps on ~24GB (official README, 2025-07-28).
DEFAULT_TASK = "ti2v-5B"
DEFAULT_SIZE = "1280*704"
DEFAULT_FPS = 24
DEFAULT_HF_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


@dataclass(frozen=True)
class WanModelSpec:
    """One official or legacy checkpoint this harness can name."""

    key: str
    task: str
    family: str
    size_class: str
    hf_id: str
    hf_id_native: str
    diffusers_pipeline: str
    description: str
    recommended_vram_gb: int
    default_size: str
    sample_fps: int
    sample_steps: int
    sample_shift: float
    sample_guide_scale: float
    frame_num: int
    flow_shift: float
    max_resolution: Tuple[int, int]
    github_repo: str
    upstream_git_sha: str
    hf_revision: str = HF_REVISION_DEFAULT
    distilled: bool = False
    legacy: bool = False
    requires: Sequence[str] = field(default_factory=tuple)


WAN_MODELS: Dict[str, WanModelSpec] = {
    "ti2v-5B": WanModelSpec(
        key="ti2v-5B",
        task="ti2v-5B",
        family="wan2.2",
        size_class="5B",
        hf_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        hf_id_native="Wan-AI/Wan2.2-TI2V-5B",
        diffusers_pipeline="WanPipeline",
        description="Hybrid text/image-to-video, Wan2.2-VAE 16x16x4, 720P@24fps",
        recommended_vram_gb=24,
        default_size="1280*704",
        sample_fps=24,
        sample_steps=50,
        sample_shift=5.0,
        sample_guide_scale=5.0,
        frame_num=121,
        flow_shift=5.0,
        max_resolution=(1280, 704),
        github_repo=WAN22_GITHUB,
        upstream_git_sha=WAN22_GIT_SHA,
        requires=(),
    ),
    "t2v-A14B": WanModelSpec(
        key="t2v-A14B",
        task="t2v-A14B",
        family="wan2.2",
        size_class="A14B",
        hf_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        hf_id_native="Wan-AI/Wan2.2-T2V-A14B",
        diffusers_pipeline="WanPipeline",
        description="Text-to-video MoE, 14B active / 27B total, 480P and 720P",
        recommended_vram_gb=80,
        default_size="1280*720",
        sample_fps=16,
        sample_steps=40,
        sample_shift=12.0,
        sample_guide_scale=4.0,
        frame_num=81,
        flow_shift=12.0,
        max_resolution=(1280, 720),
        github_repo=WAN22_GITHUB,
        upstream_git_sha=WAN22_GIT_SHA,
        requires=(),
    ),
    "i2v-A14B": WanModelSpec(
        key="i2v-A14B",
        task="i2v-A14B",
        family="wan2.2",
        size_class="A14B",
        hf_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        hf_id_native="Wan-AI/Wan2.2-I2V-A14B",
        diffusers_pipeline="WanImageToVideoPipeline",
        description="Image-to-video MoE, 480P and 720P",
        recommended_vram_gb=80,
        default_size="1280*720",
        sample_fps=16,
        sample_steps=40,
        sample_shift=12.0,
        sample_guide_scale=4.0,
        frame_num=81,
        flow_shift=12.0,
        max_resolution=(1280, 720),
        github_repo=WAN22_GITHUB,
        upstream_git_sha=WAN22_GIT_SHA,
        requires=("image",),
    ),
    "s2v-14B": WanModelSpec(
        key="s2v-14B",
        task="s2v-14B",
        family="wan2.2",
        size_class="14B",
        hf_id="Wan-AI/Wan2.2-S2V-14B",
        hf_id_native="Wan-AI/Wan2.2-S2V-14B",
        diffusers_pipeline="WanS2VPipeline",
        description="Speech-to-video, image + audio, 480P and 720P",
        recommended_vram_gb=80,
        default_size="1024*704",
        sample_fps=16,
        sample_steps=40,
        sample_shift=3.0,
        sample_guide_scale=4.5,
        frame_num=80,
        flow_shift=3.0,
        max_resolution=(1280, 720),
        github_repo=WAN22_GITHUB,
        upstream_git_sha=WAN22_GIT_SHA,
        requires=("image", "audio"),
    ),
    "animate-14B": WanModelSpec(
        key="animate-14B",
        task="animate-14B",
        family="wan2.2",
        size_class="14B",
        hf_id="Wan-AI/Wan2.2-Animate-14B-Diffusers",
        hf_id_native="Wan-AI/Wan2.2-Animate-14B",
        diffusers_pipeline="WanAnimatePipeline",
        description="Character animation and replacement (Sep 2025 weights)",
        recommended_vram_gb=80,
        default_size="1280*720",
        sample_fps=30,
        sample_steps=20,
        sample_shift=5.0,
        sample_guide_scale=1.0,
        frame_num=77,
        flow_shift=5.0,
        max_resolution=(1280, 720),
        github_repo=WAN22_GITHUB,
        upstream_git_sha=WAN22_GIT_SHA,
        requires=("image", "pose_video"),
    ),
    "animate-2-14B": WanModelSpec(
        key="animate-2-14B",
        task="animate-2-14B",
        family="wan2.2",
        size_class="14B",
        hf_id="Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
        hf_id_native="Wan-AI/Wan2.2-Animate-2-14B",
        diffusers_pipeline="WanAnimate2Pipeline",
        description="Wan-Animate-2 character animation (HF 2026-07-14, Diffusers 2026-08-06)",
        recommended_vram_gb=80,
        default_size="1280*720",
        sample_fps=30,
        sample_steps=20,
        sample_shift=5.0,
        sample_guide_scale=1.0,
        frame_num=77,
        flow_shift=5.0,
        max_resolution=(1280, 720),
        github_repo=ANIMATE2_GITHUB,
        upstream_git_sha=ANIMATE2_GIT_SHA,
        requires=("image", "pose_video"),
    ),
    "animate-2-14B-distilled": WanModelSpec(
        key="animate-2-14B-distilled",
        task="animate-2-14B",
        family="wan2.2",
        size_class="14B",
        hf_id="Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
        hf_id_native="Wan-AI/Wan2.2-Animate-2-14B",
        diffusers_pipeline="WanAnimate2Pipeline",
        description="Official few-step distilled Animate-2 (HF created 2026-08-06)",
        recommended_vram_gb=80,
        default_size="1280*720",
        sample_fps=30,
        sample_steps=4,
        sample_shift=5.0,
        sample_guide_scale=1.0,
        frame_num=77,
        flow_shift=5.0,
        max_resolution=(1280, 720),
        github_repo=ANIMATE2_GITHUB,
        upstream_git_sha=ANIMATE2_GIT_SHA,
        distilled=True,
        requires=("image", "pose_video"),
    ),
    # Legacy Wan2.1 kept so older eval runs still resolve.
    "wan2.1-t2v-1.3b": WanModelSpec(
        key="wan2.1-t2v-1.3b",
        task="t2v-1.3B",
        family="wan2.1",
        size_class="1.3B",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        hf_id_native="Wan-AI/Wan2.1-T2V-1.3B",
        diffusers_pipeline="WanPipeline",
        description="Legacy Wan2.1 480P text-to-video",
        recommended_vram_gb=8,
        default_size="832*480",
        sample_fps=16,
        sample_steps=50,
        sample_shift=3.0,
        sample_guide_scale=5.0,
        frame_num=81,
        flow_shift=3.0,
        max_resolution=(832, 480),
        github_repo=WAN21_GITHUB,
        upstream_git_sha="unknown",
        legacy=True,
        requires=(),
    ),
    "wan2.1-t2v-14b": WanModelSpec(
        key="wan2.1-t2v-14b",
        task="t2v-14B",
        family="wan2.1",
        size_class="14B",
        hf_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        hf_id_native="Wan-AI/Wan2.1-T2V-14B",
        diffusers_pipeline="WanPipeline",
        description="Legacy Wan2.1 720P text-to-video",
        recommended_vram_gb=15,
        default_size="1280*720",
        sample_fps=16,
        sample_steps=50,
        sample_shift=5.0,
        sample_guide_scale=5.0,
        frame_num=81,
        flow_shift=5.0,
        max_resolution=(1280, 720),
        github_repo=WAN21_GITHUB,
        upstream_git_sha="unknown",
        legacy=True,
        requires=(),
    ),
}

# Aliases used by older configs and quality routers.
MODEL_ALIASES: Dict[str, str] = {
    "wan2-1-fast": "wan2.1-t2v-1.3b",
    "wan2.1-t2v-1.3B": "wan2.1-t2v-1.3b",
    "wan2.1-t2v-14B": "wan2.1-t2v-14b",
    "multimodalart/wan2-1-fast": "wan2.1-t2v-1.3b",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "ti2v-5B",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "t2v-A14B",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "i2v-A14B",
    "Wan-AI/Wan2.2-S2V-14B": "s2v-14B",
    "Wan-AI/Wan2.2-Animate-14B-Diffusers": "animate-14B",
    "Wan-AI/Wan2.2-Animate-2-14B-Diffusers": "animate-2-14B",
    "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers": "animate-2-14B-distilled",
}


def normalize_size(size: str) -> str:
    """Map 1280x720 or 1280*720 onto the official SIZE_CONFIGS key."""
    token = size.strip().lower().replace("x", "*")
    if token in SIZE_CONFIGS:
        return token
    raise ValueError(f"Unknown size {size!r}; expected one of {sorted(SIZE_CONFIGS)}")


def size_to_wxh(size: str) -> str:
    """API-facing WIDTHxHEIGHT form."""
    width, height = SIZE_CONFIGS[normalize_size(size)]
    return f"{width}x{height}"


def resolve_model_key(name: str) -> str:
    if name in WAN_MODELS:
        return name
    aliased = MODEL_ALIASES.get(name)
    if aliased:
        return aliased
    raise KeyError(f"Unknown Wan model {name!r}")


def get_model(name: str) -> WanModelSpec:
    return WAN_MODELS[resolve_model_key(name)]


def models_for_task(task: str, include_legacy: bool = False) -> List[WanModelSpec]:
    out = [spec for spec in WAN_MODELS.values() if spec.task == task]
    if not include_legacy:
        out = [spec for spec in out if not spec.legacy]
    return out


def current_zoo_metadata() -> Dict[str, object]:
    """Pins recorded on every generation result JSON."""
    return {
        "wan_family": "wan2.2",
        "upstream_git_sha": WAN22_GIT_SHA,
        "upstream_git_date": WAN22_GIT_DATE,
        "upstream_repo": WAN22_GITHUB,
        "animate2_git_sha": ANIMATE2_GIT_SHA,
        "animate2_git_date": ANIMATE2_GIT_DATE,
        "hf_revision": HF_REVISION_DEFAULT,
        "pinned_on": "2026-08-15",
        "default_task": DEFAULT_TASK,
        "default_hf_id": DEFAULT_HF_ID,
        "open_weight_only": True,
        "api_only_skipped": ["wan2.5", "wan2.6", "wan3.0"],
    }


def select_model_for_quality(quality: str, resolution: str) -> str:
    """Map old quality/resolution knobs onto the Wan2.2 zoo."""
    quality = quality.lower()
    try:
        size_key = normalize_size(resolution)
        width, height = SIZE_CONFIGS[size_key]
    except ValueError:
        parts = resolution.lower().replace("*", "x").split("x")
        width, height = int(parts[0]), int(parts[1])
    pixels = width * height
    if quality in {"low", "medium"} or pixels <= 832 * 480:
        return "ti2v-5B" if pixels >= 704 * 1280 else "wan2.1-t2v-1.3b"
    if quality == "ultra":
        return "t2v-A14B"
    return "ti2v-5B"
