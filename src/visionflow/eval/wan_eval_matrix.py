"""Expand the Wan2.2 eval matrix into concrete generation cells.

Axes match what Wan2.2 actually ships (generate.py + HF zoo, 2026-08-15):
task, size class (5B vs A14B/14B), prompt-extend on/off, resolution/fps
presets, and sample_steps / distilled solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

from ..services.generation.wan_generate_spec import (
    WanGenerateRequest,
    few_step_ablation,
    prompt_extend_ablation_flags,
)
from ..services.generation.wan_model_zoo import get_model, normalize_size


DEFAULT_SUITE = Path("config/wan_eval_suite.yaml")


@dataclass(frozen=True)
class EvalCell:
    """One generate+score job."""

    cell_id: str
    task: str
    model_key: str
    size: str
    duration_s: float
    fps: int
    use_prompt_extend: bool
    sample_solver: str
    sample_steps: int
    distilled: bool
    prompt: str
    image: Optional[str] = None
    audio: Optional[str] = None
    pose_video: Optional[str] = None
    seed: int = 42
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_generate_request(self) -> WanGenerateRequest:
        return WanGenerateRequest(
            task=self.task,
            size=self.size,
            prompt=self.prompt,
            image=self.image,
            audio=self.audio,
            pose_video=self.pose_video,
            use_prompt_extend=self.use_prompt_extend,
            sample_solver=self.sample_solver,
            sample_steps=self.sample_steps,
            base_seed=self.seed,
            distilled=self.distilled,
            model_key=self.model_key,
        )


def load_eval_suite(path: Optional[Path] = None) -> Dict[str, Any]:
    suite_path = path or DEFAULT_SUITE
    text = Path(suite_path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Eval suite {suite_path} is not a mapping")
    return data


def _size_for_task(task: str, preset: Dict[str, Any]) -> str:
    raw = preset.get("size")
    if raw:
        return normalize_size(str(raw))
    return get_model(task).default_size


def expand_eval_matrix(suite: Optional[Dict[str, Any]] = None) -> List[EvalCell]:
    data = suite if suite is not None else load_eval_suite()
    prompts = data.get("prompts") or {}
    cells: List[EvalCell] = []
    index = 0
    for entry in data.get("tasks") or []:
        task = entry["task"]
        model_key = entry.get("model_key") or task
        prompt = entry.get("prompt") or prompts.get(task) or prompts.get("default", "")
        image = entry.get("image")
        audio = entry.get("audio")
        pose_video = entry.get("pose_video")
        seed = int(entry.get("seed", data.get("default_seed", 42)))
        duration_s = float(entry.get("duration_s", data.get("default_duration_s", 5)))
        extend_flags = (
            prompt_extend_ablation_flags()
            if entry.get("prompt_extend_ablation", True)
            else [{"use_prompt_extend": False, "prompt_extend_method": None}]
        )
        step_rows = (
            few_step_ablation(task)
            if entry.get("scheduler_ablation", False)
            else [
                {
                    "label": "default",
                    "sample_solver": entry.get("sample_solver", "unipc"),
                    "sample_steps": entry.get("sample_steps") or get_model(model_key).sample_steps,
                    "distilled": bool(entry.get("distilled", False)),
                    "model_key": model_key,
                }
            ]
        )
        presets = entry.get("presets") or [{"name": "default"}]
        for preset in presets:
            size = _size_for_task(task, preset)
            fps = int(preset.get("fps") or get_model(model_key).sample_fps)
            preset_duration = float(preset.get("duration_s", duration_s))
            for extend in extend_flags:
                for step in step_rows:
                    cell_model = step.get("model_key") or model_key
                    index += 1
                    cells.append(
                        EvalCell(
                            cell_id=f"{task}-{preset.get('name', 'p')}-{step['label']}-"
                            f"pe{int(extend['use_prompt_extend'])}-{index:03d}",
                            task=task,
                            model_key=cell_model,
                            size=size,
                            duration_s=preset_duration,
                            fps=fps,
                            use_prompt_extend=bool(extend["use_prompt_extend"]),
                            sample_solver=step["sample_solver"],
                            sample_steps=int(step["sample_steps"]),
                            distilled=bool(step.get("distilled", False)),
                            prompt=prompt,
                            image=image,
                            audio=audio,
                            pose_video=pose_video,
                            seed=seed,
                            tags={
                                "preset": preset.get("name"),
                                "size_class": get_model(cell_model).size_class,
                                "family": get_model(cell_model).family,
                                "scheduler_label": step["label"],
                            },
                        )
                    )
    return cells


def iter_eval_cells(path: Optional[Path] = None) -> Iterator[EvalCell]:
    yield from expand_eval_matrix(load_eval_suite(path) if path else None)
