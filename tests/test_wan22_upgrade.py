"""Wan2.2 zoo, generate.py mapping, and eval matrix (no GPU required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from visionflow.eval.wan_eval_matrix import expand_eval_matrix, load_eval_suite
from visionflow.services.generation.wan_generate_spec import (
    WanGenerateRequest,
    as_generate_argv,
    few_step_ablation,
    result_identity,
    validate_generate_request,
)
from visionflow.services.generation.wan_model_zoo import (
    DEFAULT_HF_ID,
    DEFAULT_TASK,
    WAN22_GIT_SHA,
    current_zoo_metadata,
    get_model,
    normalize_size,
    resolve_model_key,
    select_model_for_quality,
)
from visionflow.shared.models import VideoGenerationRequest, WanTask


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "config" / "wan_eval_suite.yaml"


def test_default_task_is_ti2v_5b():
    assert DEFAULT_TASK == "ti2v-5B"
    spec = get_model(DEFAULT_TASK)
    assert spec.hf_id == DEFAULT_HF_ID
    assert spec.hf_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert spec.size_class == "5B"
    assert spec.sample_fps == 24
    assert spec.sample_steps == 50
    assert spec.upstream_git_sha == WAN22_GIT_SHA


def test_legacy_wan21_alias_still_resolves():
    assert resolve_model_key("wan2-1-fast") == "wan2.1-t2v-1.3b"
    assert resolve_model_key("multimodalart/wan2-1-fast") == "wan2.1-t2v-1.3b"
    assert get_model("wan2.1-t2v-14b").legacy is True


def test_official_sizes_and_star_form():
    assert normalize_size("1280x704") == "1280*704"
    assert normalize_size("1280*720") == "1280*720"
    with pytest.raises(ValueError):
        normalize_size("512x512")


def test_ti2v_rejects_480p():
    req = WanGenerateRequest(task="ti2v-5B", size="832*480", prompt="a cat")
    with pytest.raises(ValueError, match="not supported"):
        validate_generate_request(req)


def test_i2v_requires_image():
    req = WanGenerateRequest(task="i2v-A14B", size="1280*720", prompt="a cat")
    with pytest.raises(ValueError, match="image"):
        validate_generate_request(req)


def test_s2v_requires_audio_unless_tts():
    req = WanGenerateRequest(task="s2v-14B", size="1024*704", prompt="talk", image="ref.png")
    with pytest.raises(ValueError, match="audio"):
        validate_generate_request(req)
    ok = WanGenerateRequest(
        task="s2v-14B",
        size="1024*704",
        prompt="talk",
        image="ref.png",
        enable_tts=True,
    )
    validate_generate_request(ok)


def test_prompt_extend_and_solver_in_argv():
    req = validate_generate_request(
        WanGenerateRequest(
            task="t2v-A14B",
            size="832*480",
            prompt="boxing cats",
            use_prompt_extend=True,
            sample_solver="dpm++",
            sample_steps=8,
            base_seed=7,
        )
    )
    argv = as_generate_argv(req)
    assert "--use_prompt_extend" in argv
    assert "dpm++" in argv
    assert "8" in argv
    identity = result_identity(req, 7)
    assert identity["seed"] == 7
    assert identity["upstream_git_sha"] == WAN22_GIT_SHA
    assert identity["hf_revision"] == "main"
    assert identity["use_prompt_extend"] is True
    assert identity["size_class"] == "A14B"


def test_animate2_distilled_few_step():
    rows = few_step_ablation("animate-2-14B")
    distilled = [row for row in rows if row.get("distilled")]
    assert distilled
    assert distilled[0]["sample_steps"] == 4
    spec = get_model("animate-2-14B-distilled")
    assert spec.hf_id.endswith("Distilled-Diffusers")
    assert spec.distilled is True


def test_eval_suite_covers_new_tasks():
    suite = load_eval_suite(SUITE)
    cells = expand_eval_matrix(suite)
    tasks = {cell.task for cell in cells}
    assert {"ti2v-5B", "t2v-A14B", "i2v-A14B", "s2v-14B", "animate-14B", "animate-2-14B"} <= tasks
    size_classes = {cell.tags["size_class"] for cell in cells}
    assert "5B" in size_classes
    assert "A14B" in size_classes or "14B" in size_classes
    extend_values = {cell.use_prompt_extend for cell in cells if cell.task == "ti2v-5B"}
    assert extend_values == {True, False}
    distilled_cells = [cell for cell in cells if cell.distilled]
    assert distilled_cells
    assert any(cell.fps == 24 and cell.size == "1280*704" for cell in cells)
    assert any(cell.sample_solver == "dpm++" for cell in cells)


def test_api_request_accepts_star_resolution_and_new_fields():
    req = VideoGenerationRequest(
        prompt="a cat on a surfboard",
        resolution="1280*704",
        task=WanTask.TI2V_5B,
        use_prompt_extend=True,
        sample_solver="unipc",
        num_inference_steps=4,
        seed=42,
    )
    assert req.resolution == "1280x704"
    assert req.task == WanTask.TI2V_5B
    assert req.use_prompt_extend is True
    assert req.num_inference_steps == 4


def test_quality_router_keeps_legacy_small_frames():
    assert select_model_for_quality("medium", "512x512") == "wan2.1-t2v-1.3b"
    assert select_model_for_quality("medium", "1280x704") == "ti2v-5B"
    assert select_model_for_quality("ultra", "1280x720") == "t2v-A14B"


def test_zoo_metadata_skips_api_only_releases():
    meta = current_zoo_metadata()
    assert meta["open_weight_only"] is True
    assert "wan2.5" in meta["api_only_skipped"]
    assert meta["upstream_git_sha"] == WAN22_GIT_SHA
    assert meta["pinned_on"] == "2026-08-15"
