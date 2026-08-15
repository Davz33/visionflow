"""Process-level checks for the Wan2.2 eval CLI added in 7550509.

These tests start a real interpreter. They do not load Diffusers or a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "visionflow.eval.cli"]


def _env() -> dict:
    env = dict(os.environ)
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


def _run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI, *args],
        cwd=str(ROOT),
        env=_env(),
        check=check,
        capture_output=True,
        text=True,
    )


def _payload(args: List[str]) -> dict:
    completed = _run(args)
    return json.loads(completed.stdout)


def test_cli_zoo_lists_nine_open_weight_models():
    data = _payload(["zoo"])
    assert data["pins"]["default_task"] == "ti2v-5B"
    assert data["pins"]["default_hf_id"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert data["pins"]["open_weight_only"] is True
    assert set(data["models"]) == {
        "ti2v-5B",
        "t2v-A14B",
        "i2v-A14B",
        "s2v-14B",
        "animate-14B",
        "animate-2-14B",
        "animate-2-14B-distilled",
        "wan2.1-t2v-1.3b",
        "wan2.1-t2v-14b",
    }


def test_cli_defaults_match_ti2v_5b_pin():
    data = _payload(["defaults"])
    assert data["task"] == "ti2v-5B"
    assert data["size"] == "1280*704"
    assert data["fps"] == 24
    assert data["sample_steps"] == 50


def test_cli_matrix_expands_official_axes():
    data = _payload(["matrix"])
    assert data["count"] == 35
    tasks = {cell["task"] for cell in data["cells"]}
    assert {
        "ti2v-5B",
        "t2v-A14B",
        "i2v-A14B",
        "s2v-14B",
        "animate-14B",
        "animate-2-14B",
    } <= tasks
    extend = {cell["use_prompt_extend"] for cell in data["cells"] if cell["task"] == "ti2v-5B"}
    assert extend == {True, False}
    assert any(cell["distilled"] for cell in data["cells"])
    assert any(cell["sample_solver"] == "dpm++" for cell in data["cells"])


def test_cli_plan_writes_identity_and_json_out(tmp_path: Path):
    out = tmp_path / "plan.json"
    data = _payload(
        [
            "plan",
            "--task",
            "ti2v-5B",
            "--size",
            "1280*704",
            "--prompt",
            "a cat on a surfboard",
            "--use_prompt_extend",
            "--sample_solver",
            "dpm++",
            "--sample_steps",
            "8",
            "--base_seed",
            "7",
            "--json_out",
            str(out),
        ]
    )
    assert "--use_prompt_extend" in data["argv"]
    assert data["identity"]["seed"] == 7
    assert data["identity"]["hf_revision"] == "main"
    assert data["identity"]["upstream_git_sha"]
    assert data["model"]["hf_id"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert json.loads(out.read_text(encoding="utf-8"))["identity"]["seed"] == 7


def test_cli_plan_rejects_i2v_without_image():
    completed = _run(
        ["plan", "--task", "i2v-A14B", "--size", "1280*720", "--prompt", "a cat"],
        check=False,
    )
    assert completed.returncode == 2
    assert "image" in completed.stderr.lower()


def test_cli_plan_accepts_i2v_with_image_path():
    data = _payload(
        [
            "plan",
            "--task",
            "i2v-A14B",
            "--size",
            "1280*720",
            "--prompt",
            "a cat",
            "--image",
            "examples/i2v_input.JPG",
        ]
    )
    assert "--image" in data["argv"]
    assert data["identity"]["task"] == "i2v-A14B"


def test_cli_plan_distilled_animate2():
    data = _payload(
        [
            "plan",
            "--task",
            "animate-2-14B",
            "--size",
            "1280*720",
            "--prompt",
            "people",
            "--image",
            "img.jpg",
            "--pose_video",
            "pose.mp4",
            "--distilled",
            "--sample_steps",
            "4",
        ]
    )
    assert data["model"]["key"] == "animate-2-14B-distilled"
    assert data["identity"]["distilled"] is True
    assert data["identity"]["sample_steps"] == 4


def test_scripts_entry_point_defaults():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "wan_eval_cli.py"), "defaults"],
        cwd=str(ROOT),
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    assert data["hf_id"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def test_check_files_fails_when_suite_assets_are_absent():
    from visionflow.eval.wan_eval_matrix import expand_eval_matrix, load_eval_suite
    from visionflow.services.generation.wan_generate_spec import validate_generate_request

    suite = load_eval_suite(ROOT / "config" / "wan_eval_suite.yaml")
    i2v = next(cell for cell in expand_eval_matrix(suite) if cell.task == "i2v-A14B")
    with pytest.raises(FileNotFoundError):
        validate_generate_request(i2v.to_generate_request(), check_files=True)
