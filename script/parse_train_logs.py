#!/usr/bin/env python3
"""Parse train-diy-v0_92 logs and metrics into clean CSV/JSON files for analysis.

Produces:
  - episodes.csv       : per-episode reward, frames, eval flag
  - training_metrics.csv: time-series of algorithm/env/basic metrics
  - metrics.csv         : Prometheus time-series (all metrics, pivoted)
  - summary.json        : overview stats
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path("logs/train-diy-v0_92")


def parse_aisrv_logs() -> tuple[list[dict], list[dict]]:
    """Extract episode terminations and training_metrics from aisrv logs."""
    episodes = []
    metrics = []

    log_path = LOG_DIR / "aisrv-logs.jsonl"
    if not log_path.exists():
        print(f"SKIP: {log_path} not found")
        return episodes, metrics

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", "")
            time_str = entry.get("time", "")

            # Episode termination: "episode_105 terminated in fno_7910, truncated:False, eval:False, reward_sum:-51.777..."
            m = re.search(
                r"episode_(\d+)\s+terminated.*?fno_(\d+).*?truncated:(True|False).*?eval:(True|False).*?reward_sum:([-\d.]+)",
                msg,
            )
            if m:
                episodes.append(
                    {
                        "time": time_str,
                        "episode": int(m.group(1)),
                        "frame": int(m.group(2)),
                        "truncated": m.group(3) == "True",
                        "eval": m.group(4) == "True",
                        "reward_sum": float(m.group(5)),
                    }
                )
                continue

            # Training metrics: "training_metrics <category> is {...}"
            # category can be multi-word like "env selfplay", "env common_ai"
            m = re.search(r"training_metrics\s+(.+?)\s+is\s+(\{.+\})", msg)
            if m:
                category = m.group(1).replace(" ", "_")
                try:
                    data = ast.literal_eval(m.group(2))
                except (SyntaxError, ValueError):
                    continue
                metrics.append({"time": time_str, "category": category, **data})

    # Deduplicate training_metrics: keep only one per (time, category)
    seen = set()
    deduped = []
    for row in metrics:
        key = (row["time"], row["category"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    # Sort by time
    episodes.sort(key=lambda x: x["time"])
    deduped.sort(key=lambda x: x["time"])

    return episodes, deduped


def parse_metrics_yaml() -> list[dict]:
    """Parse the raw metrics YAML output into list of {name, time_index, value}."""
    path = LOG_DIR / "metrics.yaml"
    if not path.exists():
        print(f"SKIP: {path} not found")
        return []

    with open(path) as f:
        text = f.read()

    # Parse the YAML-like output (it's kaiwu CLI format, not standard YAML)
    # Each metric block:
    #   ---
    #   name: <metric_name>
    #   has_data: true
    #   points: N
    #   raw: [v1, v2, ...]
    records = []
    current_name = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("---"):
            continue
        # Only "name: <value>" sets the current metric name
        if stripped.startswith("name:"):
            current_name = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("raw:"):
            if current_name is None:
                continue
            raw_data = stripped[len("raw:"):].strip()
            try:
                values = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            if not isinstance(values, list):
                continue
            for i, v in enumerate(values):
                records.append(
                    {"metric": current_name, "time_step": i, "value": float(v)}
                )

    return records


def save_episodes(episodes: list[dict]) -> None:
    path = LOG_DIR / "episodes.csv"
    if not episodes:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "episode", "frame", "truncated", "eval", "reward_sum"])
        writer.writeheader()
        writer.writerows(episodes)
    print(f"  episodes.csv: {len(episodes)} rows")


def save_training_metrics(metrics: list[dict]) -> None:
    path = LOG_DIR / "training_metrics.csv"
    if not metrics:
        return
    # Collect all field names
    all_fields = set()
    for row in metrics:
        all_fields.update(row.keys())
    fields = ["time", "category"] + sorted(all_fields - {"time", "category"})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    print(f"  training_metrics.csv: {len(metrics)} rows")


def save_metrics_pivoted(records: list[dict]) -> None:
    """Pivot time-series so each metric is a column."""
    path = LOG_DIR / "metrics.csv"
    if not records:
        return

    # Group by time_step
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    for r in records:
        by_step[r["time_step"]][r["metric"]] = r["value"]

    all_metrics = sorted(set(r["metric"] for r in records))
    steps = sorted(by_step.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step"] + all_metrics)
        for step in steps:
            row = [step] + [by_step[step].get(m, "") for m in all_metrics]
            writer.writerow(row)
    print(f"  metrics.csv: {len(steps)} time steps x {len(all_metrics)} metrics")


def save_summary(episodes: list[dict], metrics: list[dict]) -> None:
    path = LOG_DIR / "summary.json"

    # Episode stats
    train_eps = [e for e in episodes if not e["eval"]]
    eval_eps = [e for e in episodes if e["eval"]]

    train_rewards = [e["reward_sum"] for e in train_eps]
    eval_rewards = [e["reward_sum"] for e in eval_eps]

    summary = {
        "task": "train-diy-v0_92",
        "task_id": 218841,
        "episodes": {
            "total": len(episodes),
            "train": len(train_eps),
            "eval": len(eval_eps),
        },
        "reward": {
            "train": {
                "count": len(train_rewards),
                "mean": sum(train_rewards) / len(train_rewards) if train_rewards else 0,
                "min": min(train_rewards) if train_rewards else 0,
                "max": max(train_rewards) if train_rewards else 0,
            },
            "eval": {
                "count": len(eval_rewards),
                "mean": sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0,
                "min": min(eval_rewards) if eval_rewards else 0,
                "max": max(eval_rewards) if eval_rewards else 0,
            },
        },
        "latest_metrics": {},
    }

    # Latest algo metrics
    algo_metrics = [m for m in metrics if m["category"] == "algorithm"]
    if algo_metrics:
        latest = algo_metrics[-1]
        summary["latest_metrics"]["algorithm"] = {
            k: v for k, v in latest.items() if k not in ("time", "category")
        }

    # Latest env metrics
    for cat in ("env", "basic"):
        cat_metrics = [m for m in metrics if m["category"] == cat]
        if cat_metrics:
            latest = cat_metrics[-1]
            summary["latest_metrics"][cat] = {
                k: v for k, v in latest.items() if k not in ("time", "category")
            }

    # Latest selfplay/common_ai metrics
    for sub in ("selfplay", "common_ai"):
        sub_metrics = [m for m in metrics if m["category"] == f"env_{sub}"]
        if sub_metrics:
            latest = sub_metrics[-1]
            summary["latest_metrics"][f"env_{sub}"] = {
                k: v for k, v in latest.items() if k not in ("time", "category")
            }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  summary.json written")


def main() -> None:
    print("Parsing aisrv logs...")
    episodes, metrics = parse_aisrv_logs()
    print(f"  Episodes: {len(episodes)}, Training metrics: {len(metrics)}")

    print("Parsing Prometheus metrics...")
    metric_records = parse_metrics_yaml()
    print(f"  Metric records: {len(metric_records)}")

    print("\nSaving files to", LOG_DIR.resolve())
    save_episodes(episodes)
    save_training_metrics(metrics)
    save_metrics_pivoted(metric_records)
    save_summary(episodes, metrics)

    print("\nDone! Files in", LOG_DIR.resolve())
    print("  episodes.csv       — per-episode reward/frame/eval")
    print("  training_metrics.csv — algorithm/env metrics over time")
    print("  metrics.csv         — Prometheus time-series (all metrics)")
    print("  summary.json        — overview statistics")
    print("  aisrv-logs.jsonl    — raw aisrv logs (keep for reference)")


if __name__ == "__main__":
    main()
