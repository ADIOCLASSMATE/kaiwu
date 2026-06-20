#!/usr/bin/env python3
"""Pull all meaningful training data from Kaiwu platform API.

Fetches:
  1. ALL metrics: 37 standard + every custom metric from monitor_builder.py
  2. ERROR/WARNING logs only (skips INFO noise)
  3. Task metadata (start/end time, status)

Output per task under data/<task_name>/:
  summary.json   — task info + latest metric snapshot grouped by category
  metrics.csv    — wide time series (all metrics aligned by time step)
  metrics.json   — full raw arrays + metadata
  errors.jsonl   — non-INFO logs (empty if training is clean)

Usage:
  uv run python script/pull_training_data.py --task-id 219006
  uv run python script/pull_training_data.py --task-id 219006 --name train-diy-v0_94
  uv run python script/pull_training_data.py --task-id 219006 --no-logs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── session / auth (same as kaiwu_remote.py) ──────────────────────────

def kaiwu_sign(timestamp: int, token: str, url_path: str) -> int:
    last = url_path.rstrip("/").split("/")[-1]
    payload = str(timestamp) + token[-32:] + last
    value = 5381
    for char in payload:
        value = (value + ((value << 5) + ord(char))) & 0xFFFFFFFF
    return value & 0x7FFFFFFF


def load_session() -> dict:
    path = Path.home() / ".kaiwu/session.json"
    if not path.exists():
        raise SystemExit("Kaiwu session not found. Run: kaiwu login")
    return json.loads(path.read_text(encoding="utf-8"))


def kaiwu_api(url_path: str, body: dict, timeout: int = 120) -> dict:
    session = load_session()
    token = session.get("token") or ""
    ts = int(time.time())
    headers = {
        "Accept": "application/json",
        "Accept-Language": "zh",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "x-kaiwu-ts": str(ts),
        "x-kaiwu-auth": str(kaiwu_sign(ts, token, url_path)),
    }
    req = urllib.request.Request(
        "https://tencentarena.com" + url_path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") not in (None, 0):
        raise SystemExit(
            f"API error {data.get('code')}: {data.get('msg') or data.get('reason')}"
        )
    return data.get("data", data)


# ── context helpers ───────────────────────────────────────────────────

def build_context(session: dict, args: argparse.Namespace) -> dict:
    domain_type = args.domain_type or session.get("domain_type", "course")
    domain_id = args.domain_id or session.get("stage_id", 0)
    experiment_id = args.experiment_id or session.get("experiment_id", 0)
    ctx: dict = {
        "domain": {"type": domain_type, "id": domain_id},
        "experiment_id": experiment_id,
    }
    # competition 需要 team_id；course 不需要
    if domain_type != "course":
        team_id = args.team_id or session.get("team_id", 0)
        if team_id:
            ctx["competition_team_id"] = team_id
    return ctx


def api_prefix(ctx: dict) -> str:
    return "/api/v5/Course" if ctx["domain"]["type"] == "course" else "/api/v5/Competition"


# ── metric name discovery ─────────────────────────────────────────────

# 37 standard platform metrics with battle-tested PromQL expressions
# (from kaiwu-cli/commands/metric.js METRIC_PROFILES)
STANDARD_METRICS: dict[str, str] = {
    "predict_succ_cnt": 'sum(max_over_time(kaiwu_actor_predict_succ_cnt{}[1h]))',
    "train_global_step": 'sum(kaiwu_train_global_step{})',
    "load_model_succ_cnt": 'sum(kaiwu_actor_load_last_model_succ_cnt{})',
    "sample_receive_cnt": 'sum(kaiwu_sample_receive_cnt{})',
    "train_success_cnt": 'sum(kaiwu_train_success_cnt{})',
    "episode_cnt": 'sum(max_over_time(kaiwu_episode_cnt{}[1h]))',
    "sample_production_and_consumption_ratio": 'avg(kaiwu_sample_production_and_consumption_ratio{})',
    "reward": 'avg(kaiwu_reward{})',
    "total_loss": 'avg(kaiwu_total_loss{})',
    "value_loss": 'avg(kaiwu_value_loss{})',
    "policy_loss": 'avg(kaiwu_policy_loss{})',
    "entropy_loss": 'avg(kaiwu_entropy_loss{})',
    "win_rate": 'avg(kaiwu_win_rate{})',
    "self_tower_hp": 'avg(kaiwu_self_tower_hp{})',
    "enemy_tower_hp": 'avg(kaiwu_enemy_tower_hp{})',
    "frame": 'avg(kaiwu_frame{})',
    "money_per_frame": 'avg(kaiwu_money_per_frame{})',
    "kill": 'avg(kaiwu_kill{})',
    "death": 'avg(kaiwu_death{})',
    "hurt_by_hero": 'avg(kaiwu_hurt_by_hero{})',
    "hurt_to_hero": 'avg(kaiwu_hurt_to_hero{})',
    "batch_train_cost_time_ms": 'avg(kaiwu_batch_train_cost_time_ms{})',
    "real_train_cost_time_ms": 'avg(kaiwu_real_train_cost_time_ms{})',
    "data_fetch_cost_time_ms": 'avg(kaiwu_data_fetch_cost_time_ms{})',
    "aisrv_learner_proxy_queue_len": 'avg(kaiwu_aisrv_learner_proxy_queue_len{})',
    "reverb_ready_size": 'avg(kaiwu_reverb_ready_size{})',
    "max_sample_size": 'avg(kaiwu_max_sample_size{})',
    "sample_consume_rate": 'avg(kaiwu_sample_consume_rate{})',
    "sample_product_rate": 'avg(kaiwu_sample_product_rate{})',
    "push_to_cos_err_cnt": 'sum(kaiwu_push_to_cos_err_cnt{})',
    "push_to_cos_succ_cnt": 'sum(kaiwu_push_to_cos_succ_cnt{})',
    "push_to_model_pool_err_cnt": 'sum(kaiwu_push_to_model_pool_err_cnt{})',
    "push_to_model_pool_succ_cnt": 'sum(kaiwu_push_to_model_pool_succ_cnt{})',
    "pull_from_model_pool_err_cnt": 'sum(kaiwu_pull_from_model_pool_err_cnt{})',
    "pull_from_model_pool_succ_cnt": 'sum(kaiwu_pull_from_model_pool_succ_cnt{})',
    "send_to_reverb_err_cnt": 'sum(kaiwu_send_to_reverb_err_cnt{})',
    "send_to_reverb_succ_cnt": 'sum(kaiwu_send_to_reverb_succ_cnt{})',
}


def _parse_monitor_via_mock(monitor_path: Path) -> list[str]:
    """Execute monitor_builder.py with mocked kaiwudrl to capture metric names.

    This is the most reliable method — it runs the actual build_monitor() code,
    so all loops, conditionals, and computed names are captured correctly.
    Falls back gracefully if the mock environment can't be set up.
    """
    import importlib.util
    import sys
    from unittest.mock import MagicMock

    class CaptureBuilder:
        """Mock MonitorConfigBuilder that captures metric names from add_metric calls."""
        def __init__(self):
            self.names: list[str] = []

        def title(self, name):
            return self

        def add_group(self, **kw):
            return self

        def add_panel(self, **kw):
            return self

        def add_metric(self, metrics_name=None, expr=None, **kw):
            if metrics_name:
                self.names.append(metrics_name)
            return self

        def end_panel(self):
            return self

        def end_group(self):
            return self

        def build(self):
            return {}

    saved = {}
    try:
        # Mock kaiwudrl package tree
        kaiwudrl = MagicMock()
        kaiwudrl.common = MagicMock()
        kaiwudrl.common.monitor = MagicMock()
        kaiwudrl.common.monitor.monitor_config_builder = MagicMock()
        kaiwudrl.common.monitor.monitor_config_builder.MonitorConfigBuilder = CaptureBuilder
        for mod_name in [
            "kaiwudrl", "kaiwudrl.common", "kaiwudrl.common.monitor",
            "kaiwudrl.common.monitor.monitor_config_builder",
        ]:
            saved[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = locals()[mod_name.replace(".", "_")] \
                if mod_name == "kaiwudrl" else (
                    kaiwudrl.common if mod_name == "kaiwudrl.common" else
                    kaiwudrl.common.monitor if mod_name == "kaiwudrl.common.monitor" else
                    kaiwudrl.common.monitor.monitor_config_builder
                )

        # Mock agent_diy.conf.conf (for GameConfig.ATTACK_BUTTONS)
        agent_diy = MagicMock()
        agent_diy.conf = MagicMock()
        agent_diy.conf.conf = MagicMock()
        agent_diy.conf.conf.GameConfig = type("GC", (), {"ATTACK_BUTTONS": (3, 4, 5, 6, 8, 10, 11)})
        for mod_name in ["agent_diy", "agent_diy.conf", "agent_diy.conf.conf"]:
            saved[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = (
                agent_diy if mod_name == "agent_diy" else
                agent_diy.conf if mod_name == "agent_diy.conf" else
                agent_diy.conf.conf
            )

        # Mock agent_ppo.conf.conf (for FeatureConfig.HERO_CONFIG_IDS + GameConfig.ATTACK_BUTTONS)
        agent_ppo = MagicMock()
        agent_ppo.conf = MagicMock()
        agent_ppo.conf.conf = MagicMock()
        agent_ppo.conf.conf.FeatureConfig = type("FC", (), {"HERO_CONFIG_IDS": [112, 133, 199]})
        agent_ppo.conf.conf.GameConfig = type("GC", (), {"ATTACK_BUTTONS": (3, 4, 5, 6, 8, 10, 11)})
        for mod_name in ["agent_ppo", "agent_ppo.conf", "agent_ppo.conf.conf"]:
            saved[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = (
                agent_ppo if mod_name == "agent_ppo" else
                agent_ppo.conf if mod_name == "agent_ppo.conf" else
                agent_ppo.conf.conf
            )

        spec = importlib.util.spec_from_file_location(
            "_pull_monitor_builder", monitor_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # build_monitor creates its own MonitorConfigBuilder() inside —
        # that will be our CaptureBuilder, so the instance gets .add_metric calls.
        # We need to capture from the instance, not the class.
        # Monkey-patch the class to capture the instance.
        builder_instance = None

        class CaptureBuilderWithRef(CaptureBuilder):
            def __init__(self):
                super().__init__()
                nonlocal builder_instance
                builder_instance = self

        kaiwudrl.common.monitor.monitor_config_builder.MonitorConfigBuilder = CaptureBuilderWithRef

        spec2 = importlib.util.spec_from_file_location(
            "_pull_monitor_builder2", monitor_path
        )
        mod2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(mod2)
        mod2.build_monitor()

        return builder_instance.names if builder_instance else []

    except Exception as e:
        print(f"  [WARN] Mock-based metric extraction failed: {e}", file=sys.stderr)
        return []
    finally:
        # Restore sys.modules
        for name, orig in saved.items():
            if orig is not None:
                sys.modules[name] = orig
            else:
                sys.modules.pop(name, None)
        # Clean up our temporary modules
        for tmp in ["_pull_monitor_builder", "_pull_monitor_builder2"]:
            sys.modules.pop(tmp, None)


def parse_monitor_metrics(monitor_path: Path) -> dict[str, str]:
    """Extract all metric names from monitor_builder.py.

    Uses mock-based execution (runs build_monitor() with intercepted add_metric calls),
    which handles all loops, conditionals, and computed names correctly.

    Returns {metric_name: promql_expr}. PromQL uses kaiwu_ prefix for API compatibility.
    """
    if not monitor_path.exists():
        return {}

    names = _parse_monitor_via_mock(monitor_path)

    if not names:
        # Fallback: regex scan for string-literal add_metric calls
        text = monitor_path.read_text(encoding="utf-8")
        names = list(set(
            m.group(1) for m in
            re.finditer(r'add_metric\(\s*metrics_name\s*=\s*["\'](\w+)["\']', text)
        ))

    return {name: f"avg(kaiwu_{name}{{}})" for name in sorted(set(names))}


def default_monitor_paths() -> list[Path]:
    """Return monitor builders for the PPO agent only."""
    return [
        ROOT / "agent_ppo/conf/monitor_builder.py",
    ]


def discover_monitor_metrics(monitor_paths: list[Path]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for monitor_path in monitor_paths:
        for name, expr in parse_monitor_metrics(monitor_path).items():
            metrics.setdefault(name, expr)
    return metrics


def discover_all_metrics(monitor_paths: list[Path] | None = None) -> dict[str, str]:
    """Merge standard 37 metrics with all custom metrics from monitor_builder.py.

    Standard metrics keep their battle-tested expressions (some use sum/max_over_time).
    Custom metrics use auto-generated avg(<name>{}).
    """
    if monitor_paths is None:
        monitor_paths = default_monitor_paths()

    all_metrics = dict(STANDARD_METRICS)  # copy
    custom = discover_monitor_metrics(monitor_paths)

    # Add custom metrics that don't overlap with standard
    for name, expr in sorted(custom.items()):
        if name not in all_metrics:
            all_metrics[name] = expr

    return all_metrics


def display_monitor_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ── task helpers ──────────────────────────────────────────────────────

def find_task(ctx: dict, prefix: str, task_id: int | None, name: str | None) -> dict:
    """Get task metadata from ListTrainTask."""
    body: dict = {**ctx, "page": {"current": 1, "size": 50}}
    if name:
        body["name"] = name
    data = kaiwu_api(f"{prefix}/ListTrainTask", body)
    tasks = data.get("train_task") or []
    if not tasks:
        raise SystemExit("No tasks found")
    if task_id:
        for t in tasks:
            if t.get("id") == task_id:
                return t
        raise SystemExit(f"task-id={task_id} not found in recent 50 tasks")
    # by name
    exact = next((t for t in tasks if t.get("name") == name), None)
    if exact:
        return exact
    return tasks[0]


def get_task_timespan(task: dict, args: argparse.Namespace) -> tuple[str, str]:
    """Return (start_iso, end_iso) from args or task metadata."""
    start = args.start or task.get("start_time") or task.get("created_at") or ""
    end = args.end or task.get("end_time") or ""
    if not end:
        end = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    if not start:
        raise SystemExit("Cannot determine start_time from task. Use --start.")
    return start, end


# ── metric pull ───────────────────────────────────────────────────────

def pull_metrics(
    ctx: dict, prefix: str, task_id: int,
    start_iso: str, end_iso: str,
    metrics: dict[str, str],
    step: int = 15,
    batch_size: int = 50,
) -> list[dict]:
    """Pull all metrics in batches. Returns combined raw API results."""
    items = list(metrics.items())
    all_results: list[dict] = []
    n_batches = (len(items) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch = items[b * batch_size : (b + 1) * batch_size]
        queries = [
            {"name": name, "expr": expr, "id": f"{name}_{i}", "step": str(step)}
            for i, (name, expr) in enumerate(batch)
        ]
        body: dict = {
            **ctx,
            "train_task_id": task_id,
            "start_time": {"timestamp": start_iso},
            "end_time": {"timestamp": end_iso},
            "queries": queries,
        }
        label = f"batch {b+1}/{n_batches}"
        data = kaiwu_api(f"{prefix}/GetTrainMetricRange", body, timeout=120)
        results = data.get("results") or []
        all_results.extend(results)
        with_data = sum(1 for r in results if r.get("items"))
        print(f"  {label}: {len(results)} queries, {with_data} with data")

    return all_results


def process_metric_results(
    results: list[dict], queries_map: dict[str, str], expr_map: dict[str, str]
) -> dict:
    """Convert raw API results into structured data.

    Returns:
        {
            "metrics": {name: {has_data, points, min, max, last, values}},
            "time_steps": int,  # max data points across all metrics
            "aligned": {step: {name: value}},  # for CSV
        }
    """
    # Build id → name map
    id_to_name: dict[str, str] = {}
    for r in results:
        rid = r.get("id", "")
        # id format: "name_N" — strip trailing _N to get name
        m = re.match(r"^(.+)_\d+$", rid)
        id_to_name[rid] = m.group(1) if m else rid

    metrics_data: dict[str, dict] = {}
    max_steps = 0

    for r in results:
        rid = r.get("id", "")
        name = id_to_name.get(rid, rid)
        items = r.get("items") or []

        # Flatten all values across items (items = different label sets)
        all_values: list[float] = []
        for item in items:
            for v in item.get("values") or []:
                try:
                    all_values.append(float(v.get("value", 0)))
                except (ValueError, TypeError):
                    all_values.append(0.0)

        non_zero = [v for v in all_values if v != 0]
        has_data = (
            "yes" if non_zero else ("all_zero" if all_values else "no")
        )

        metrics_data[name] = {
            "has_data": has_data,
            "points": len(all_values),
            "min": min(all_values) if all_values else None,
            "max": max(all_values) if all_values else None,
            "last": all_values[-1] if all_values else None,
            "values": all_values,
            "expr": expr_map.get(name, ""),
        }

        if len(all_values) > max_steps:
            max_steps = len(all_values)

    # Build aligned time step grid
    aligned: dict[int, dict[str, float | None]] = {}
    for name, md in metrics_data.items():
        for i, v in enumerate(md.get("values") or []):
            if i not in aligned:
                aligned[i] = {}
            aligned[i][name] = v

    return {
        "metrics": metrics_data,
        "time_steps": max_steps,
        "aligned": aligned,
    }


# ── log pull ──────────────────────────────────────────────────────────

def pull_error_logs(
    ctx: dict, prefix: str, task_id: int,
    start_iso: str, end_iso: str,
) -> list[dict]:
    """Pull non-INFO logs (ERROR, WARNING) via GetTrainLog.

    Queries each level separately to avoid pulling thousands of INFO entries.
    """
    all_logs: list[dict] = []
    for level in ("ERROR", "WARNING"):
        page = 1
        while True:
            body: dict = {
                **ctx,
                "train_task_id": task_id,
                "start_time": {"timestamp": start_iso},
                "end_time": {"timestamp": end_iso},
                "query": "query_log",
                "page": {"size": 100, "current": page},
                "var": {"message": "*", "level": level, "module": "*"},
            }
            data = kaiwu_api(f"{prefix}/GetTrainLog", body)
            logs = data.get("logs") or []
            if not logs:
                break
            all_logs.extend(logs)
            if len(logs) < 100:
                break
            page += 1
            if page > 100:
                break
    return all_logs


# ── output ────────────────────────────────────────────────────────────

METRIC_GROUPS: dict[str, list[str]] = {
    "algorithm": ["reward", "total_loss", "value_loss", "policy_loss", "entropy_loss",
                   "grad_norm", "learning_rate", "is_train_rate"],
    "training_basic": ["train_global_step", "episode_cnt", "sample_production_and_consumption_ratio",
                        "sample_receive_cnt", "predict_succ_cnt", "load_model_succ_cnt",
                        "train_success_cnt"],
    "env": ["win_rate", "win", "self_tower_hp", "enemy_tower_hp", "frame",
            "kill", "kill_cnt", "death", "dead_cnt", "money_per_frame",
            "hurt_to_hero", "hurt_by_hero", "final_level", "final_money",
            "final_hp_ratio", "episode_len"],
    "feature_health": ["feat_nan", "feat_inf", "feat_neg", "feat_frames"],
    "token_exists": [],  # populated dynamically
    "token_act": [],     # populated dynamically
    "token_dead": [],    # populated dynamically
    "global_act": ["feat_global_mean", "feat_global_std", "feat_global_dead"],
    "reward_items": [],  # populated dynamically
    "action": [],        # populated dynamically
    "entropy_per_head": ["entropy_head_0", "entropy_head_1", "entropy_head_2",
                         "entropy_head_3", "entropy_head_4", "entropy_head_5"],
    "advantage": ["adv_mean", "adv_std"],
    "distance_shaping": [],  # populated dynamically
    "out_of_range_breakdown": [],  # populated dynamically
    "noop_context": [],  # populated dynamically
    "training_distribution": [],  # populated dynamically
    "direction_head_distribution": [],  # populated dynamically
    "action_mask_health": [],  # populated dynamically
    "idle_health": ["idle_triggered", "idle_triggered_rate"],
    "system": ["batch_train_cost_time_ms", "real_train_cost_time_ms",
               "data_fetch_cost_time_ms", "aisrv_learner_proxy_queue_len",
               "reverb_ready_size", "max_sample_size", "sample_consume_rate",
               "sample_product_rate"],
    "errors": ["push_to_cos_err_cnt", "push_to_cos_succ_cnt",
               "push_to_model_pool_err_cnt", "push_to_model_pool_succ_cnt",
               "pull_from_model_pool_err_cnt", "pull_from_model_pool_succ_cnt",
               "send_to_reverb_err_cnt", "send_to_reverb_succ_cnt"],
    "misc_metrics": [],  # populated dynamically; catches future metrics
}


def _populate_dynamic_groups(available: set[str]) -> None:
    """Fill in dynamic group lists based on what's actually available."""
    tokens = ["main_hero", "enemy_hero", "own_tower", "enemy_tower",
              "own_minions", "enemy_minions", "monsters", "bullets", "cakes"]
    rwd_prefixes = ["rwd_tower_hp_point", "rwd_lane_progress", "rwd_lane_presence",
                     "rwd_retreat_recover", "rwd_recall_recover", "rwd_hp_point", "rwd_danger_penalty",
                     "rwd_kill", "rwd_death", "rwd_money", "rwd_exp",
                     "rwd_last_hit", "rwd_last_hit_focus", "rwd_minion_hp_point",
                     "rwd_kill_monster", "rwd_idle_penalty", "rwd_tower_attack",
                     "rwd_distance_penalty", "rwd_terminal"]

    # token exists
    METRIC_GROUPS["token_exists"] = [
        f"feat_{t}_exists" for t in tokens if f"feat_{t}_exists" in available
    ]
    # token mean/std/dead
    METRIC_GROUPS["token_act"] = []
    for t in tokens:
        for suffix in ("_mean", "_std"):
            name = f"feat_{t}{suffix}"
            if name in available:
                METRIC_GROUPS["token_act"].append(name)
    METRIC_GROUPS["token_dead"] = [
        f"feat_{t}_dead" for t in tokens if f"feat_{t}_dead" in available
    ]
    # reward items
    METRIC_GROUPS["reward_items"] = [
        p for p in rwd_prefixes if p in available
    ]
    # action buttons + targets + attack targets + joint
    action_names = []
    for i in range(12):
        n = f"action_button_{i}"
        if n in available:
            action_names.append(n)
    for i in range(9):
        n = f"action_target_{i}"
        if n in available:
            action_names.append(n)
    for prefix in ["attack_target_", "attack_action_target_", "attack_button_"]:
        action_names.extend(
            sorted(n for n in available if n.startswith(prefix) and n not in action_names)
        )
    METRIC_GROUPS["action"] = action_names
    # training distribution
    METRIC_GROUPS["training_distribution"] = sorted(
        n for n in available
        if n.startswith(("opponent_", "hero_", "enemy_hero_", "mirror_"))
        and not n.startswith("enemy_tower")   # exclude enemy_tower_hp
    )
    # distance shaping (expanded): all out_of_range_* and attack_* shaping metrics
    METRIC_GROUPS["distance_shaping"] = sorted(
        n for n in available
        if n.startswith(("out_of_range_", "attack_action_", "last_hit_window_",
                         "frontline_presence_", "attack_in_range_",
                         "attack_near_out_", "attack_far_out_", "resolved_attack_",
                         "action_penalty_", "invalid_target_penalty_"))
    )
    # out of range breakdown (button + target)
    METRIC_GROUPS["out_of_range_breakdown"] = sorted(
        n for n in available
        if n.startswith("out_of_range_button_") or n.startswith("out_of_range_target_")
    )
    # direction head distribution
    METRIC_GROUPS["direction_head_distribution"] = sorted(
        n for n in available if n.startswith("action_head_")
    )
    # noop context
    METRIC_GROUPS["noop_context"] = sorted(
        n for n in available if n.startswith("noop_")
    )
    # action mask health
    METRIC_GROUPS["action_mask_health"] = sorted(
        n for n in available if n.startswith("button3_")
    )
    assigned = {
        name
        for group, names in METRIC_GROUPS.items()
        if group != "misc_metrics"
        for name in names
    }
    METRIC_GROUPS["misc_metrics"] = sorted(available - assigned)


def build_summary(
    task: dict, metrics_data: dict, available: set[str],
    start_iso: str, end_iso: str,
) -> dict:
    """Build a structured summary JSON."""
    _populate_dynamic_groups(available)

    def latest(name: str):
        md = metrics_data.get(name)
        return md["last"] if md and md["has_data"] != "no" else None

    def group_snapshot(group_name: str) -> dict[str, float | None]:
        names = METRIC_GROUPS.get(group_name, [])
        result = {}
        for n in names:
            val = latest(n)
            if val is not None:
                result[n] = val
        return result

    return {
        "task_id": task.get("id"),
        "task_name": task.get("name", ""),
        "task_status": task.get("status", ""),
        "time_range": {"start": start_iso, "end": end_iso},
        "total_metrics_available": len([n for n, m in metrics_data.items() if m["has_data"] != "no"]),
        "total_metrics_queried": len(metrics_data),
        "latest": {
            group: group_snapshot(group) for group in METRIC_GROUPS
        },
    }


def save_output(
    out_dir: Path, task: dict, metrics_data: dict, aligned: dict,
    error_logs: list[dict], start_iso: str, end_iso: str,
) -> None:
    """Write all output files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    available = set(metrics_data.keys())

    # 1. summary.json
    summary = build_summary(task, metrics_data, available, start_iso, end_iso)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), "utf-8"
    )
    print(f"  summary.json — {summary['total_metrics_available']}/{summary['total_metrics_queried']} metrics with data")

    # 2. metrics.csv — wide time series
    if aligned:
        all_names = sorted(available)
        csv_path = out_dir / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_step"] + all_names)
            for step in sorted(aligned.keys()):
                row_data = aligned[step]
                writer.writerow([step] + [row_data.get(n, "") for n in all_names])
        print(f"  metrics.csv — {len(aligned)} time steps × {len(all_names)} metrics")

    # 3. metrics.json — full data
    metrics_json = {
        "task_id": task.get("id"),
        "task_name": task.get("name", ""),
        "time_range": {"start": start_iso, "end": end_iso},
        "time_steps": max((m.get("points", 0) for m in metrics_data.values()), default=0),
        "metrics": {
            name: {
                k: v for k, v in m.items() if k != "values"  # don't duplicate raw arrays
            }
            for name, m in sorted(metrics_data.items())
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2), "utf-8"
    )
    print(f"  metrics.json — metadata for {len(metrics_data)} metrics")

    # 4. errors.jsonl
    errors_path = out_dir / "errors.jsonl"
    if error_logs:
        with open(errors_path, "w", encoding="utf-8") as f:
            for log in error_logs:
                if isinstance(log, str):
                    try:
                        log = json.loads(log)
                    except json.JSONDecodeError:
                        log = {"raw": log}
                f.write(json.dumps(log, ensure_ascii=False) + "\n")

        # error summary: group by message pattern
        msg_counts: dict[str, int] = {}
        for log in error_logs:
            msg = log.get("message", "") if isinstance(log, dict) else str(log)
            # truncate to first 120 chars as pattern
            pattern = msg[:120]
            msg_counts[pattern] = msg_counts.get(pattern, 0) + 1
        error_summary = {
            "total": len(error_logs),
            "by_level": {},
            "by_module": {},
            "top_patterns": sorted(msg_counts.items(), key=lambda x: -x[1])[:20],
        }
        for log in error_logs:
            lvl = log.get("level", "?") if isinstance(log, dict) else "?"
            mod = log.get("module", "?") if isinstance(log, dict) else "?"
            error_summary["by_level"][lvl] = error_summary["by_level"].get(lvl, 0) + 1
            error_summary["by_module"][mod] = error_summary["by_module"].get(mod, 0) + 1
        (out_dir / "errors_summary.json").write_text(
            json.dumps(error_summary, ensure_ascii=False, indent=2), "utf-8"
        )
        print(f"  errors.jsonl — {len(error_logs)} ERROR/WARNING logs")
    else:
        errors_path.write_text("", encoding="utf-8")
        print("  errors.jsonl — no errors (clean run)")


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull all meaningful training data from Kaiwu platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python script/pull_training_data.py --task-id 219006
  uv run python script/pull_training_data.py --task-id 219006 -o data/my-analysis
  uv run python script/pull_training_data.py --name train-diy-v0_94 --no-logs
        """,
    )
    # task identification
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--task-id", type=int, help="Train task ID")
    g.add_argument("--name", type=str, help="Train task name (finds most recent)")

    # context overrides
    parser.add_argument("--domain-type", default="", help="course / competition_stage")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--experiment-id", type=int, default=0)
    parser.add_argument("--team-id", type=int, default=0)

    # time range
    parser.add_argument("--start", default="", help="ISO8601 start (default: task start_time)")
    parser.add_argument("--end", default="", help="ISO8601 end (default: task end_time or now)")
    parser.add_argument("--step", type=int, default=15, help="Sampling step in seconds (default: 15)")

    # what to pull
    parser.add_argument("--no-logs", action="store_true", help="Skip error log pull")
    parser.add_argument("--no-standard", action="store_true",
                        help="Skip 37 standard platform metrics (pull only custom)")
    parser.add_argument("--monitor-path", default="",
                        help="Path to monitor_builder.py (default: agent_ppo/conf/monitor_builder.py)")

    # output
    parser.add_argument("-o", "--output", default="", help="Output directory (default: data/<task_name>)")

    args = parser.parse_args()

    # ── setup ──
    session = load_session()
    ctx = build_context(session, args)
    prefix = api_prefix(ctx)

    # ── find task ──
    print(f"Looking up task...")
    task = find_task(ctx, prefix, args.task_id, args.name)
    task_id = task["id"]
    task_name = task.get("name", f"task-{task_id}")
    start_iso, end_iso = get_task_timespan(task, args)
    print(f"  Task: {task_name} (id={task_id})")
    print(f"  Time: {start_iso} → {end_iso}")

    # ── discover metrics ──
    monitor_paths = [Path(args.monitor_path)] if args.monitor_path else default_monitor_paths()
    if args.no_standard:
        metrics = discover_monitor_metrics(monitor_paths)
    else:
        metrics = discover_all_metrics(monitor_paths)
    monitor_label = ", ".join(display_monitor_path(path) for path in monitor_paths)
    print(f"  Monitors: {monitor_label}")
    print(f"\nMetrics to pull: {len(metrics)} ({len(STANDARD_METRICS)} standard + {len(metrics) - len(STANDARD_METRICS)} custom)")

    # ── pull metrics ──
    print("\nPulling metrics via GetTrainMetricRange...")
    results = pull_metrics(ctx, prefix, task_id, start_iso, end_iso, metrics, args.step)

    # Build query lookup maps
    expr_map = {name: expr for name, expr in metrics.items()}
    queries_map = {f"{name}_{i}": name for i, name in enumerate(metrics.keys())}

    processed = process_metric_results(results, queries_map, expr_map)
    metrics_data = processed["metrics"]
    aligned = processed["aligned"]

    with_yes = sum(1 for m in metrics_data.values() if m["has_data"] == "yes")
    all_zero = sum(1 for m in metrics_data.values() if m["has_data"] == "all_zero")
    no_data = sum(1 for m in metrics_data.values() if m["has_data"] == "no")
    print(f"  {with_yes} have data, {all_zero} all-zero, {no_data} no-data, "
          f"{processed['time_steps']} time steps")

    # ── pull error logs ──
    error_logs: list[dict] = []
    if not args.no_logs:
        print("\nPulling ERROR/WARNING logs...")
        error_logs = pull_error_logs(ctx, prefix, task_id, start_iso, end_iso)
        if error_logs:
            levels: dict[str, int] = {}
            for log in error_logs:
                lvl = log.get("level", "?") if isinstance(log, dict) else "?"
                levels[lvl] = levels.get(lvl, 0) + 1
            print(f"  Found {len(error_logs)} log entries: {levels}")
        else:
            print("  No ERROR/WARNING logs — training is clean")

    # ── save ──
    out_dir = Path(args.output) if args.output else (ROOT / "logs" / task_name)
    print(f"\nSaving to {out_dir}/...")
    save_output(out_dir, task, metrics_data, aligned, error_logs, start_iso, end_iso)
    print("\nDone.")


if __name__ == "__main__":
    main()
