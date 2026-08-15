"""Wan2.2 eval CLI. Flag names follow upstream generate.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from visionflow.eval.wan_eval_matrix import expand_eval_matrix, load_eval_suite
from visionflow.services.generation.wan_generate_spec import (
    WanGenerateRequest,
    as_generate_argv,
    result_identity,
    validate_generate_request,
)
from visionflow.services.generation.wan_model_zoo import (
    DEFAULT_HF_ID,
    DEFAULT_TASK,
    WAN22_GIT_SHA,
    WAN_MODELS,
    current_zoo_metadata,
    get_model,
)


def _add_generate_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default=DEFAULT_TASK, help="Official generate.py task")
    parser.add_argument("--size", default="1280*704", help="width*height, official SIZE_CONFIGS")
    parser.add_argument("--prompt", default="", help="Text prompt")
    parser.add_argument("--ckpt_dir", default=None, help="HF id or local checkpoint dir")
    parser.add_argument("--image", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--pose_video", default=None)
    parser.add_argument("--use_prompt_extend", action="store_true", default=False)
    parser.add_argument(
        "--prompt_extend_method",
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
    )
    parser.add_argument("--prompt_extend_target_lang", default="en", choices=["zh", "en"])
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--model_key", default=None, help="Zoo key, e.g. ti2v-5B")
    parser.add_argument("--distilled", action="store_true", default=False)
    parser.add_argument(
        "--json_out",
        default=None,
        help="Write result metadata JSON here (dry-run still writes the plan)",
    )


def _request_from_args(args: argparse.Namespace) -> WanGenerateRequest:
    return WanGenerateRequest(
        task=args.task,
        size=args.size,
        prompt=args.prompt,
        ckpt_dir=args.ckpt_dir,
        image=args.image,
        audio=args.audio,
        pose_video=getattr(args, "pose_video", None),
        use_prompt_extend=args.use_prompt_extend,
        prompt_extend_method=args.prompt_extend_method,
        prompt_extend_target_lang=args.prompt_extend_target_lang,
        sample_solver=args.sample_solver,
        sample_steps=args.sample_steps,
        base_seed=args.base_seed,
        distilled=args.distilled,
        model_key=args.model_key,
    )


def cmd_zoo(_args: argparse.Namespace) -> int:
    payload: Dict[str, Any] = {
        "pins": current_zoo_metadata(),
        "models": {},
    }
    for key, spec in WAN_MODELS.items():
        payload["models"][key] = {
            "task": spec.task,
            "family": spec.family,
            "size_class": spec.size_class,
            "hf_id": spec.hf_id,
            "hf_id_native": spec.hf_id_native,
            "diffusers_pipeline": spec.diffusers_pipeline,
            "sample_steps": spec.sample_steps,
            "sample_fps": spec.sample_fps,
            "default_size": spec.default_size,
            "recommended_vram_gb": spec.recommended_vram_gb,
            "distilled": spec.distilled,
            "legacy": spec.legacy,
            "upstream_git_sha": spec.upstream_git_sha,
        }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    suite = load_eval_suite(Path(args.suite) if args.suite else None)
    cells = expand_eval_matrix(suite)
    rows = []
    for cell in cells:
        rows.append(
            {
                "cell_id": cell.cell_id,
                "task": cell.task,
                "model_key": cell.model_key,
                "size": cell.size,
                "fps": cell.fps,
                "duration_s": cell.duration_s,
                "use_prompt_extend": cell.use_prompt_extend,
                "sample_solver": cell.sample_solver,
                "sample_steps": cell.sample_steps,
                "distilled": cell.distilled,
                "seed": cell.seed,
                "tags": cell.tags,
            }
        )
    json.dump({"count": len(rows), "cells": rows, "pins": current_zoo_metadata()}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    req = validate_generate_request(_request_from_args(args))
    spec = get_model(req.resolved_model_key())
    seed = req.base_seed if req.base_seed >= 0 else 42
    plan = {
        "argv": as_generate_argv(req),
        "identity": result_identity(req, seed),
        "model": {
            "key": spec.key,
            "hf_id": spec.hf_id,
            "pipeline": spec.diffusers_pipeline,
        },
        "pins": current_zoo_metadata(),
        "note": "Dry-run only. This harness does not vendor Wan source.",
    }
    text = json.dumps(plan, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    return 0


def cmd_defaults(_args: argparse.Namespace) -> int:
    spec = get_model(DEFAULT_TASK)
    json.dump(
        {
            "task": DEFAULT_TASK,
            "hf_id": DEFAULT_HF_ID,
            "size": spec.default_size,
            "fps": spec.sample_fps,
            "sample_steps": spec.sample_steps,
            "sample_solver": "unipc",
            "upstream_git_sha": WAN22_GIT_SHA,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wan-eval",
        description="Wan2.2 eval harness: zoo, matrix, and generate.py-compatible plans",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    zoo = sub.add_parser("zoo", help="Print pinned Wan2.2 model zoo")
    zoo.set_defaults(func=cmd_zoo)

    defaults = sub.add_parser("defaults", help="Print default task / HF id")
    defaults.set_defaults(func=cmd_defaults)

    matrix = sub.add_parser("matrix", help="Expand config/wan_eval_suite.yaml")
    matrix.add_argument("--suite", default="config/wan_eval_suite.yaml")
    matrix.set_defaults(func=cmd_matrix)

    plan = sub.add_parser("plan", help="Validate a generate.py-shaped request")
    _add_generate_flags(plan)
    plan.set_defaults(func=cmd_plan)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
