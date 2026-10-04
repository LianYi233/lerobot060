#!/usr/bin/env python3
"""Package two Piper direct-flow runs for analysis, using only Python's standard library."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

TASKS = {
    1: "1-put_the_apple_on_the_yellow_plate",
    2: "2-remove_the_cuboid_from_blue_plate",
    3: "3-move_the_tennis_from_yellow_plate_to_blue_plate",
    4: "4-pick_the_red_cube_into_the_yellow_plate",
}
REQUIRED = (
    "training_diagnostics.jsonl",
    "action_eval.jsonl",
    "action_eval_samples.json",
    "piper_normalization.json",
    "best_action_checkpoints.json",
)


def resolve_run(args, profile):
    explicit = getattr(args, f"{profile}_dir")
    run_name = f"pi05-may-{TASKS[args.task]}-dual_prompt_only-seed{args.seed}"
    if explicit is not None:
        root = explicit.expanduser().resolve()
        # Accept either the task directory itself or its timestamped RUN_GROUP parent.
        if (root / run_name).is_dir():
            root /= run_name
        if not root.is_dir() or not any((root / name).is_file() for name in REQUIRED):
            raise ValueError(f"{profile}: 未找到任务训练日志，请检查目录：{root}")
        return root
    pattern = f"piper-task{args.task}-direct-{profile}-12k-*/{run_name}"
    candidates = sorted(p.resolve() for p in args.search_root.expanduser().glob(pattern) if p.is_dir())
    if len(candidates) != 1:
        listed = "\n".join(f"  {p}" for p in candidates) or "  无匹配目录"
        raise ValueError(
            f"{profile}: 找到 {len(candidates)} 个候选，不自动选择最新实验。\n{listed}\n"
            f"请使用 --{profile}_dir 指定本次训练的任务目录或 RUN_GROUP 目录。"
        )
    return candidates[0]


def collect(run, profile, args):
    checkpoint = Path("checkpoints") / f"{args.step:06d}" / "pretrained_model"
    requested = [*REQUIRED, str(checkpoint / "train_config.json")]
    files = [(run / name, name) for name in requested if (run / name).is_file()]
    missing = [name for name in requested if not (run / name).is_file()]
    # Include the small saved model config too, but never recurse into checkpoint directories.
    model_config = checkpoint / "config.json"
    if (run / model_config).is_file():
        files.append((run / model_config, str(model_config)))

    train_config = run / checkpoint / "train_config.json"
    if train_config.is_file():
        config = json.loads(train_config.read_text(encoding="utf-8"))
        relative = config.get("policy", {}).get("use_relative_actions")
        if relative is not (profile == "relative"):
            raise ValueError(
                f"{profile}: train_config 中 use_relative_actions={relative!r} 与组别不符：{run}"
            )

    log = args.log_root.expanduser() / run.parent.name / f"{run.name}.log"
    if log.is_file():
        files.append((log, "training.log"))
    manifest = {
        "profile": profile,
        "run_directory": str(run),
        "requested_checkpoint_step": args.step,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [name for _, name in files],
        "missing_required_files": missing,
        "optional_training_log": str(log) if log.is_file() else None,
        "note": "Only analysis logs/configs are packaged; no model weights, optimizer state or videos.",
    }
    return files, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute_dir", type=Path, help="absolute 任务目录或其 RUN_GROUP 父目录")
    parser.add_argument("--relative_dir", type=Path, help="relative 任务目录或其 RUN_GROUP 父目录")
    parser.add_argument("--search_root", type=Path, default=Path("/data1/wyn/chkpt/2601-lerobot"))
    parser.add_argument("--log_root", type=Path, default=Path("/data1/wyn/logs"))
    parser.add_argument("--task", type=int, choices=TASKS, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=12000)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / f"piper-analysis-{datetime.now():%Y%m%d-%H%M%S}",
    )
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step 必须为正整数")
    output = args.output_dir.expanduser().resolve()
    try:
        # Resolve and check BOTH runs before creating either archive.
        runs = {profile: resolve_run(args, profile) for profile in ("absolute", "relative")}
        if runs["absolute"] == runs["relative"]:
            raise ValueError("absolute 和 relative 不能指向同一个目录")
        collected = {profile: collect(run, profile, args) for profile, run in runs.items()}
        for profile in runs:
            archive = output / f"{profile}.zip"
            if archive.exists():
                raise FileExistsError(f"文件已存在，不覆盖：{archive}；请更换 --output_dir")
        output.mkdir(parents=True, exist_ok=True)
        for profile, (files, manifest) in collected.items():
            archive = output / f"{profile}.zip"
            with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
                for source, name in files:
                    bundle.write(source, f"{profile}/{name}")
                bundle.writestr(
                    f"{profile}/pack_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                )
            status = "PACK_PARTIAL" if manifest["missing_required_files"] else "PACK_OK"
            print(f"{status}: {archive} ({archive.stat().st_size / 1024**2:.2f} MiB)")
            print(f"  来源：{runs[profile]}")
            for name in manifest["missing_required_files"]:
                print(f"  缺少：{name}")
            if manifest["optional_training_log"] is None:
                print("  未找到可选的完整 .log；已打包现有分析文件。")
        print("请上传上面生成的 absolute.zip 和 relative.zip。")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"PACK_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
