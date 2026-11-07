#!/usr/bin/env python3
"""Plan, manually record and summarize paired Piper trials; never control a robot.

Only Python's standard library is used. JSONL records are append-only, locked and
fsync'ed. A hardware-invalid attempt may be retried explicitly; a policy failure
or protection stop remains a failed planned trial.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "piper-revision-trials-v1"
DEFAULT_METHODS = ("priming", "direct_cpc", "flow_warmup", "priming_no_cpc")
TASKS = {
    "T1": "苹果放到黄色盘",
    "T2": "长方体从蓝盘移到预先指定的合法区域",
    "T3": "网球由黄色盘转移到蓝色盘",
    "T4": "红方块放到黄色盘",
}
OUTCOMES = ("success", "failure", "policy_stop", "invalid_hardware")
FAILURES = ("missed_grasp", "drop_in_transport", "wrong_target", "release_failure", "timeout", "other")
MODEL_FIELDS = ("checkpoint", "checkpoint_sha256", "model_manifest", "model_manifest_sha256", "cumulative_updates")
TRIAL_FIELDS = ("trial_id", "task_id", "method", "train_seed", "layout_id", "policy_seed", "order")


def _digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的整数")
    return value


def _task(value):
    task = str(value).upper()
    if task.isdigit():
        task = f"T{task}"
    if task not in TASKS:
        raise ValueError(f"未知任务 {value!r}；可选 T1,T2,T3,T4")
    return task


def _unique(values, name):
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} 不得为空或重复")
    return values


def _model_identity(model):
    """Bind a payload hash and a training-manifest hash without loading ML code.

    Supplied hashes are identifiers taken from an audited export. Missing hashes
    are computed from local files once. No model inference/weight loading occurs.
    """
    if not isinstance(model, dict):
        raise ValueError("请先提供模型身份：checkpoint、model_manifest、SHA256 和 cumulative_updates")
    result = {field: model.get(field) for field in MODEL_FIELDS}
    for field in ("checkpoint", "model_manifest"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"模型身份缺少 {field}")
    for field, hash_field in (("checkpoint", "checkpoint_sha256"), ("model_manifest", "model_manifest_sha256")):
        if not result[hash_field]:
            path = Path(result[field]).expanduser()
            if field == "checkpoint" and path.is_dir():
                path /= "model.safetensors"
            if not path.is_file():
                raise ValueError(f"无法读取 {path}；请从已核验的训练记录提供 {hash_field}")
            result[hash_field] = _file_sha256(path)
        value = result[hash_field]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(f"{hash_field} 必须是 64 位 SHA256 十六进制值")
        result[hash_field] = value.lower()
    _integer(result["cumulative_updates"], "模型 cumulative_updates", 1)
    return result


def make_plan(methods, tasks, train_seeds, trials_per_task=20, policy_seed=20261107,
              order_seed=20261108, cumulative_updates=4000, models=None):
    methods = _unique(list(methods), "methods")
    for method in methods:
        if not isinstance(method, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", method):
            raise ValueError("method 只能包含小写字母、数字、下划线和连字符，并以字母开头")
    tasks = _unique([_task(task) for task in tasks], "tasks")
    train_seeds = _unique([_integer(seed, "train_seed") for seed in train_seeds], "train_seeds")
    for value, name, minimum in ((trials_per_task, "trials_per_task", 1), (policy_seed, "policy_seed", 0),
                                 (order_seed, "order_seed", 0), (cumulative_updates, "cumulative_updates", 1)):
        _integer(value, name, minimum)
    bindings = {}
    for model in models or []:
        if not isinstance(model, dict):
            raise ValueError("模型索引的每一项必须是 JSON 对象")
        key = (model["method"], _task(model.get("task_id", model.get("task"))),
               _integer(model["train_seed"], "模型 train_seed"))
        if key in bindings or key[0] not in methods or key[1] not in tasks or key[2] not in train_seeds:
            raise ValueError(f"模型索引的组别重复或不在计划中：{key}")
        identity = _model_identity(model)
        if identity["cumulative_updates"] != cumulative_updates:
            raise ValueError(f"{key} 的累计更新次数与计划不一致")
        bindings[key] = identity
    order_rng = random.Random(order_seed)
    trials = []
    # All methods and training seeds share a layout's inference-noise seed.
    # Hash-derived seeds do not depend on method order or the selected task subset.
    for task in tasks:
        for number in range(1, trials_per_task + 1):
            layout_id = f"{task}-L{number:03d}"
            noise_seed = int(_digest([policy_seed, layout_id])[:8], 16)
            block = [(method, seed) for method in methods for seed in train_seeds]
            order_rng.shuffle(block)
            for method, seed in block:
                trials.append({
                    "trial_id": f"{layout_id}-s{seed}-{method}",
                    "task_id": task, "method": method, "train_seed": seed,
                    "layout_id": layout_id, "policy_seed": noise_seed,
                    "order": len(trials) + 1,
                    "model": bindings.get((method, task, seed)),
                })
    plan = {
        "schema": SCHEMA,
        "methods": methods, "tasks": tasks, "train_seeds": train_seeds,
        "trials_per_task": trials_per_task, "policy_seed": policy_seed,
        "order_seed": order_seed, "cumulative_updates": cumulative_updates,
        "task_descriptions": {task: TASKS[task] for task in tasks},
        "note": "layout_id 是人工布置的配对场景编号，不会自动复位机器人；生成计划不代表完成试验。",
        "trials": trials,
    }
    plan["plan_id"] = _digest(plan)
    return plan


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("schema") != SCHEMA or plan.get("plan_id") != _digest(body):
        raise ValueError("计划版本或校验值不匹配；不要在试验开始后手动改写计划")
    # Rebuild the deterministic schedule to catch inconsistent allocations or identities.
    models = {}
    for row in plan["trials"]:
        if row.get("model") is not None:
            key = (row["method"], row["task_id"], row["train_seed"])
            if key in models and models[key]["checkpoint_sha256"] != row["model"]["checkpoint_sha256"]:
                raise ValueError("同一 method/task/seed 绑定了不同模型")
            models[key] = dict(row["model"], method=key[0], task_id=key[1], train_seed=key[2])
    rebuilt = make_plan(*(plan[name] for name in ("methods", "tasks", "train_seeds", "trials_per_task",
                                               "policy_seed", "order_seed", "cumulative_updates")),
                        models=list(models.values()))
    if rebuilt != plan:
        raise ValueError("计划中的试验分配、顺序或模型身份不一致")
    return plan


def _validate_outcome(outcome, duration, failure_category, reason):
    if outcome not in OUTCOMES:
        raise ValueError(f"未知 outcome：{outcome}")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise ValueError("duration_seconds 必须为有限且非负的实际时长")
    if outcome != "invalid_hardware" and duration == 0:
        raise ValueError("有效试验的 duration_seconds 必须大于 0")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("reason 必须是文本")
    if outcome == "failure" and failure_category not in FAILURES:
        raise ValueError(f"failure 必须填写 failure_category：{', '.join(FAILURES)}")
    if outcome == "success" and failure_category is not None:
        raise ValueError("success 不得填写失败类别")
    if outcome == "policy_stop" and failure_category != "policy_safety_stop":
        raise ValueError("policy_stop 的失败类别必须为 policy_safety_stop")
    if outcome == "invalid_hardware" and failure_category != "hardware_invalid":
        raise ValueError("invalid_hardware 的失败类别必须为 hardware_invalid")
    if (outcome in ("invalid_hardware", "policy_stop") or failure_category == "other") and not (reason or "").strip():
        raise ValueError("硬件无效、保护停止及 other 失败必须填写具体 reason")


def _parse_jsonl(text):
    if text and not text.endswith("\n"):
        raise ValueError("JSONL 最后一行未完整换行；请核对并恢复记录，工具不会跳过损坏的行")
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSONL 存在无法解析的记录：{exc}") from exc


def validate_records(plan, records):
    planned = {row["trial_id"]: row for row in plan["trials"]}
    histories = defaultdict(list)
    bindings = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("JSONL 的每一行必须是试验记录对象")
        trial = planned.get(record.get("trial_id"))
        if trial is None or record.get("plan_id") != plan["plan_id"]:
            raise ValueError("记录属于其他计划，或 trial_id 不在计划中")
        if any(record.get(field) != trial[field] for field in TRIAL_FIELDS):
            raise ValueError(f"记录的场景/方法/seed/顺序与计划不一致：{trial['trial_id']}")
        outcome = record.get("outcome")
        _validate_outcome(outcome, record.get("duration_seconds"), record.get("failure_category"), record.get("reason"))
        expected_success = None if outcome == "invalid_hardware" else outcome == "success"
        if record.get("success") is not expected_success:
            raise ValueError("success 字段与 outcome 冲突")
        previous = histories[trial["trial_id"]]
        expected_attempt = f"{trial['trial_id']}.a{len(previous) + 1}"
        if record.get("attempt_id") != expected_attempt:
            raise ValueError(f"重复或不连续的 attempt_id；预期 {expected_attempt}")
        if previous:
            if previous[-1]["outcome"] != "invalid_hardware" or record.get("retry_of") != previous[-1]["attempt_id"]:
                raise ValueError("仅允许显式重试最近一次 invalid_hardware；策略失败/保护停止不得重试覆盖")
        elif record.get("retry_of") is not None:
            raise ValueError("首次尝试不得有 retry_of")
        identity = _model_identity(record.get("model"))
        key = (trial["method"], trial["task_id"], trial["train_seed"])
        if identity["cumulative_updates"] != plan["cumulative_updates"]:
            raise ValueError("记录的累计更新次数与计划不一致")
        if (trial.get("model") is not None and trial["model"] != identity) or (key in bindings and bindings[key] != identity):
            raise ValueError(f"{key} 的 checkpoint 或训练 manifest 与已绑定身份冲突")
        bindings[key] = identity
        previous.append(record)
    return histories, bindings


def read_records(plan, path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        records = _parse_jsonl(stream.read())
        validate_records(plan, records)
        return records


def record_trial(plan, log_path, trial_id, outcome, duration_seconds, *, model=None,
                 retry_of=None, failure_category=None, reason=None, video_path=None,
                 initial_photo_path=None, initial_pose=None):
    trial = next((row for row in plan["trials"] if row["trial_id"] == trial_id), None)
    if trial is None:
        raise ValueError(f"trial_id 不在计划中：{trial_id}")
    if outcome in ("policy_stop", "invalid_hardware"):
        automatic = "policy_safety_stop" if outcome == "policy_stop" else "hardware_invalid"
        if failure_category is not None and failure_category != automatic:
            raise ValueError("outcome 与 failure_category 冲突")
        failure_category = automatic
    _validate_outcome(outcome, duration_seconds, failure_category, reason)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        records = _parse_jsonl(stream.read())
        histories, bindings = validate_records(plan, records)
        previous = histories[trial_id]
        key = (trial["method"], trial["task_id"], trial["train_seed"])
        identity = dict(trial.get("model") or bindings.get(key) or {})
        identity.update({field: value for field, value in (model or {}).items() if value is not None})
        record = {
            "plan_id": plan["plan_id"],
            **{field: trial[field] for field in TRIAL_FIELDS},
            "attempt_id": f"{trial_id}.a{len(previous) + 1}", "retry_of": retry_of,
            "model": _model_identity(identity),
            "outcome": outcome,
            "success": None if outcome == "invalid_hardware" else outcome == "success",
            "duration_seconds": duration_seconds, "failure_category": failure_category,
            "reason": reason, "video_path": video_path,
            "initial_photo_path": initial_photo_path, "initial_pose": initial_pose,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        }
        validate_records(plan, [*records, record])
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        _sync_directory(log_path.parent)
    return record


def wilson95(successes, completed):
    if not completed:
        return None
    z = 1.959963984540054
    proportion = successes / completed
    denominator = 1 + z * z / completed
    center = (proportion + z * z / (2 * completed)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / completed + z * z / (4 * completed**2)) / denominator
    return [100 * max(0, center - margin), 100 * min(1, center + margin)]


def summarize(plan, records):
    histories, _ = validate_records(plan, records)

    def counts(rows):
        events = [event for row in rows for event in histories[row["trial_id"]]]
        completed = [histories[row["trial_id"]][-1] for row in rows
                     if histories[row["trial_id"]] and histories[row["trial_id"]][-1]["outcome"] != "invalid_hardware"]
        successes = sum(event["outcome"] == "success" for event in completed)
        result = {
            "planned": len(rows), "completed": len(completed), "pending": len(rows) - len(completed),
            "not_started": sum(not histories[row["trial_id"]] for row in rows),
            "retry_pending": sum(bool(histories[row["trial_id"]]) and histories[row["trial_id"]][-1]["outcome"] == "invalid_hardware" for row in rows),
            "attempts": len(events), "invalid_hardware_attempts": sum(event["outcome"] == "invalid_hardware" for event in events),
            "successes": successes, "failures": len(completed) - successes,
            "policy_stops": sum(event["outcome"] == "policy_stop" for event in completed),
            "complete": len(completed) == len(rows),
            "pooled_success_rate_pct": 100 * successes / len(completed) if completed else None,
        }
        return result

    per_task_seed = []
    per_method_seed = []
    per_method = []
    for method in plan["methods"]:
        method_rows = [row for row in plan["trials"] if row["method"] == method]
        seed_stats = []
        for seed in plan["train_seeds"]:
            seed_rows = [row for row in method_rows if row["train_seed"] == seed]
            task_stats = []
            for task in plan["tasks"]:
                result = counts([row for row in seed_rows if row["task_id"] == task])
                result.update(method=method, task_id=task, train_seed=seed,
                              wilson95_pct=wilson95(result["successes"], result["completed"]))
                per_task_seed.append(result)
                task_stats.append(result)
            result = counts(seed_rows)
            result.update(method=method, train_seed=seed,
                          macro_success_rate_pct=statistics.mean(task["pooled_success_rate_pct"] for task in task_stats) if result["complete"] else None)
            seed_stats.append(result)
            per_method_seed.append(result)
        result = counts(method_rows)
        complete_seeds = [seed["train_seed"] for seed in seed_stats if seed["complete"]]
        # Do not hide unfinished seeds or tasks when presenting training-seed variability.
        all_complete = result["complete"] and len(complete_seeds) >= 2
        rates = [seed["macro_success_rate_pct"] for seed in seed_stats]
        result.update(method=method, completed_train_seeds=complete_seeds,
                      macro_success_rate_pct=statistics.mean(rates) if result["complete"] else None,
                      training_seed_mean_pct=statistics.mean(rates) if all_complete else None,
                      training_seed_std_pct=statistics.stdev(rates) if all_complete else None)
        per_method.append(result)
    return {
        "schema": SCHEMA, "plan_id": plan["plan_id"],
        "status": counts(plan["trials"]), "per_task_seed": per_task_seed,
        "per_method_seed": per_method_seed, "per_method": per_method,
        "note": "SR 分母仅为已完成的有效试验；pending 不计成败。Wilson 区间不代表训练 seed 变异，也未校正场景相关性。宏平均及 seed 均值/标准差仅在对应计划完整后报告。",
    }


def _write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if exclusive:
            os.link(temporary, path)  # Atomic and refuses to replace an existing plan.
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _sync_directory(path.parent)


def print_summary(report):
    status = report["status"]
    print(f"计划 {status['planned']}；已完成 {status['completed']}；待完成 {status['pending']} "
          f"（未开始 {status['not_started']}，硬件无效待重试 {status['retry_pending']}）")
    print(f"实际尝试 {status['attempts']}；硬件无效 {status['invalid_hardware_attempts']}；策略保护停止 {status['policy_stops']}")
    print("method | task | train_seed | successes/completed | completed/planned | SR% | Wilson95%")
    for row in report["per_task_seed"]:
        rate = "pending" if row["pooled_success_rate_pct"] is None else f"{row['pooled_success_rate_pct']:.2f}"
        interval = "pending" if row["wilson95_pct"] is None else "–".join(f"{value:.2f}" for value in row["wilson95_pct"])
        print(f"{row['method']} | {row['task_id']} | {row['train_seed']} | {row['successes']}/{row['completed']} | "
              f"{row['completed']}/{row['planned']} | {rate} | {interval}")
    for row in report["per_method"]:
        macro = "pending" if row["macro_success_rate_pct"] is None else f"{row['macro_success_rate_pct']:.2f}%"
        pooled = "pending" if row["pooled_success_rate_pct"] is None else f"{row['pooled_success_rate_pct']:.2f}%"
        spread = "pending（需要所有计划完成且 >=2 个独立训练 seed）"
        if row["training_seed_std_pct"] is not None:
            spread = f"{row['training_seed_mean_pct']:.2f} ± {row['training_seed_std_pct']:.2f}%"
        print(f"{row['method']}: equal-task macro={macro}; pooled completed={pooled}; training-seed mean±sample std={spread}")
    print(report["note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="生成固定配对评测计划，不执行机器人")
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    plan_parser.add_argument("--tasks", default="T1,T2,T3,T4")
    plan_parser.add_argument("--train-seeds", default="0")
    plan_parser.add_argument("--trials-per-task", type=int, default=20)
    plan_parser.add_argument("--policy-seed", type=int, default=20261107)
    plan_parser.add_argument("--order-seed", type=int, default=20261108)
    plan_parser.add_argument("--cumulative-updates", type=int, default=4000)
    plan_parser.add_argument("--models", type=Path, help="可选模型索引 JSON 数组；缺省时首次记录必须绑定模型")
    record_parser = commands.add_parser("record", help="录入已执行试验的人工判定，不执行机器人")
    summary_parser = commands.add_parser("summary", help="核验日志并汇总已完成/待完成及成功率")
    for child in (record_parser, summary_parser):
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--log", type=Path, required=True)
    record_parser.add_argument("--trial-id", required=True)
    record_parser.add_argument("--outcome", choices=OUTCOMES, required=True)
    record_parser.add_argument("--duration-seconds", type=float, required=True)
    record_parser.add_argument("--retry-of")
    record_parser.add_argument("--failure-category", choices=(*FAILURES, "policy_safety_stop", "hardware_invalid"))
    record_parser.add_argument("--reason")
    record_parser.add_argument("--video-path")
    record_parser.add_argument("--initial-photo-path")
    record_parser.add_argument("--initial-pose", help="实际摆放记录，例如物体/盘的坐标和朝向；不是随机种子")
    for field in MODEL_FIELDS:
        record_parser.add_argument("--" + field.replace("_", "-"), type=int if field == "cumulative_updates" else str)
    summary_parser.add_argument("--output", type=Path, help="另存 JSON 汇总；可原子更新此报告")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            models = json.loads(args.models.read_text(encoding="utf-8")) if args.models else None
            if models is not None and not isinstance(models, list):
                raise ValueError("--models 必须是 JSON 数组")
            plan = make_plan([value.strip() for value in args.methods.split(",")],
                             [value.strip() for value in args.tasks.split(",")],
                             [int(value) for value in args.train_seeds.split(",")],
                             args.trials_per_task, args.policy_seed, args.order_seed,
                             args.cumulative_updates, models)
            _write_json(args.output, plan, exclusive=True)
            print(f"已生成 {len(plan['trials'])} 次试验计划，已执行 0 次：{args.output}")
            print(f"plan_id={plan['plan_id']}；请依 trials[].order 布置/执行；每次只录入已观察到的真实结果。")
        else:
            plan = load_plan(args.plan)
            if args.command == "record":
                result = record_trial(plan, args.log, args.trial_id, args.outcome, args.duration_seconds,
                                      model={field: getattr(args, field) for field in MODEL_FIELDS},
                                      retry_of=args.retry_of, failure_category=args.failure_category, reason=args.reason,
                                      video_path=args.video_path, initial_photo_path=args.initial_photo_path,
                                      initial_pose=args.initial_pose)
                print(f"已记录 {result['attempt_id']}: {result['outcome']}；保护停止计失败，硬件无效不计成败。")
            else:
                result = summarize(plan, read_records(plan, args.log))
                if args.output:
                    if args.output.resolve() in (args.plan.resolve(), args.log.resolve()):
                        raise ValueError("汇总输出不得覆盖计划或原始日志")
                    _write_json(args.output, result)
                print_summary(result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
