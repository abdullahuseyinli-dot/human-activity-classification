"""Execute the additional runs in a locked POLAR confirmation plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    return parser.parse_args()


def append_argument(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def command_for_run(
    *, runner: Path, manifest: Path, output_dir: Path, configuration: dict
) -> list[str]:
    command = [
        sys.executable,
        str(runner),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
    ]
    argument_names = {
        "run_role": "--run-role",
        "model_kind": "--model-kind",
        "task": "--task",
        "view": "--view",
        "unfreeze_strategy": "--unfreeze-strategy",
        "top_n_blocks": "--top-n-blocks",
        "augmentation": "--augmentation",
        "batch_size": "--batch-size",
        "grad_accum_steps": "--grad-accum-steps",
        "workers": "--workers",
        "head_lr": "--head-lr",
        "backbone_lr": "--backbone-lr",
        "weight_decay": "--weight-decay",
        "layer_decay": "--layer-decay",
        "dropout": "--dropout",
        "mixup_alpha": "--mixup-alpha",
        "label_smoothing": "--label-smoothing",
        "class_balance": "--class-balance",
        "max_epochs": "--max-epochs",
        "min_epochs": "--min-epochs",
        "patience": "--patience",
        "warmup_fraction": "--warmup-fraction",
        "gradient_clip": "--gradient-clip",
        "seed": "--seed",
        "train_size": "--train-size",
    }
    missing = sorted(set(argument_names) - set(configuration))
    if missing:
        raise RuntimeError(f"Confirmation configuration is incomplete: {missing}")
    expected_effective_batch_size = int(configuration["batch_size"]) * int(
        configuration["grad_accum_steps"]
    )
    if int(configuration.get("effective_batch_size", -1)) != expected_effective_batch_size:
        raise RuntimeError("Confirmation effective batch size is inconsistent")
    unexpected = sorted(set(configuration) - set(argument_names) - {"effective_batch_size"})
    if unexpected:
        raise RuntimeError(f"Confirmation configuration has unsupported fields: {unexpected}")
    for field, argument in argument_names.items():
        append_argument(command, argument, configuration[field])
    return command


def main() -> None:
    args = parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "DEVELOPMENT_CONFIRMATION_PLAN":
        raise RuntimeError("Expected a DEVELOPMENT_CONFIRMATION_PLAN")
    if plan.get("test_rows_read") != 0 or plan.get("test_used_for_selection"):
        raise RuntimeError("Confirmation plan violates the test gate")

    manifest = args.manifest.resolve()
    training_root = args.training_root.resolve()
    runner = Path(__file__).resolve().with_name("train_polar_candidate.py")
    runs = plan["additional_runs"]
    print(
        json.dumps(
            {
                "status": "CONFIRMATION_QUEUE_START",
                "plan_sha256": sha256_file(plan_path),
                "manifest_sha256": sha256_file(manifest),
                "additional_runs": len(runs),
                "test_rows_read": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for index, run in enumerate(runs, start=1):
        output_dir = training_root / run["run_id"]
        print(f"[confirmation {index:02d}/{len(runs):02d}] {run['run_id']}", flush=True)
        command = command_for_run(
            runner=runner,
            manifest=manifest,
            output_dir=output_dir,
            configuration=run["configuration"],
        )
        subprocess.run(command, cwd=runner.parents[1], check=True)
    print("[done] confirmation queue complete", flush=True)


if __name__ == "__main__":
    main()
