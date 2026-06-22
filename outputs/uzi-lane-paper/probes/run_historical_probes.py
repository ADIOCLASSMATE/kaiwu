#!/usr/bin/env python3
"""Run synthetic diagnostic probes on the current tree and selected commits.

The probes are intentionally small and deterministic. They are not game
evaluations; they diagnose whether a code version exposes or repairs specific
mechanics that matter for the paper narrative.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "outputs" / "uzi-lane-paper" / "probes"
RESULTS_DIR = OUT_DIR / "results"
WORKTREE_ROOT = Path(tempfile.gettempdir()) / "kaiwu-uzi-historical-probes"

COMMITS = [
    ("436d5ad", "ppo_diy_feature_swap"),
    ("7b86846", "hybrid_encoder"),
    ("f5ce14f", "ppo_owned_features_rewards"),
    ("4c07829", "training_observability_masks"),
    ("ba56882", "multi_hero_curriculum"),
    ("a824774", "recall_reward_shaping"),
    ("19fd508", "temporary_recall_exploration"),
    ("3463df5", "recall_button_legal"),
    ("357627a", "recall_ignition_monitoring"),
    ("840fd3e", "recall_channel_state"),
]

BUNDLED_PYTHON = Path(
    "/Users/wjx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
)
PYTHON = str(BUNDLED_PYTHON if BUNDLED_PYTHON.exists() else sys.executable)


INNER_PROBE = r'''
import copy
import json
import math
import os
import traceback
from pathlib import Path

out = {
    "cwd": os.getcwd(),
    "ok": True,
    "errors": [],
    "config": {},
    "probes": {},
}

def record_error(name, exc):
    out["errors"].append({
        "name": name,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
    })

def make_hero(runtime_id, camp, *, hp=1000, max_hp=1000, attack_range=5000, x=0, z=0, config_id=112):
    return {
        "runtime_id": runtime_id,
        "config_id": config_id,
        "actor_type": 0,
        "sub_type": 0,
        "camp": camp,
        "hp": hp,
        "max_hp": max_hp,
        "attack_range": attack_range,
        "location": {"x": x, "z": z},
        "camp_visible": [True, True],
        "skill_state": {"slot_states": []},
        "buff_state": {"buff_skills": [], "buff_marks": []},
    }

def make_tower(camp, *, hp=1000, x=0, z=0):
    return {
        "runtime_id": 1000 + camp,
        "actor_type": 2,
        "sub_type": 21,
        "camp": camp,
        "hp": hp,
        "max_hp": 1000,
        "attack_range": 5000,
        "attack_target": 0,
        "location": {"x": x, "z": z},
        "camp_visible": [True, True],
    }

def make_minion(runtime_id, camp, *, hp=1000, x=0, z=0, marks=None):
    return {
        "runtime_id": runtime_id,
        "actor_type": 1,
        "sub_type": 11,
        "camp": camp,
        "hp": hp,
        "max_hp": 1000,
        "kill_income": 40,
        "attack_target": 0,
        "location": {"x": x, "z": z},
        "camp_visible": [True, True],
        "buff_state": {"buff_skills": [], "buff_marks": marks or []},
    }

def make_frame(*, main=None, enemy=None, npcs=None, frame_no=0):
    return {
        "frame_no": frame_no,
        "hero_states": [
            main or make_hero(101, 1, x=-10000),
            enemy or make_hero(202, 2, x=10000),
        ],
        "npc_states": npcs or [make_tower(1, x=-15000), make_tower(2, x=15000)],
        "bullets": [],
        "cakes": [],
        "frame_action": {"dead_action": []},
        "map_state": {},
    }

try:
    from agent_ppo.conf.conf import Config, FeatureConfig, GameConfig
    out["config"]["label_size_list"] = list(getattr(Config, "LABEL_SIZE_LIST", []))
    out["config"]["legal_action_size_list"] = list(getattr(Config, "LEGAL_ACTION_SIZE_LIST", []))
    out["config"]["feature_dim"] = getattr(Config, "FEATURE_DIM", None)
    out["config"]["recall_enabled_default"] = getattr(GameConfig, "RECALL_ENABLED", None)
    out["config"]["recall_reward_weight_default"] = getattr(GameConfig, "REWARD_WEIGHT_DICT", {}).get("recall_recover")
    out["config"]["hero_config_ids"] = list(getattr(FeatureConfig, "HERO_CONFIG_IDS", []))
except Exception as exc:
    out["ok"] = False
    record_error("import_config", exc)
    print(json.dumps(out, sort_keys=True))
    raise SystemExit(0)

# Probe 1: Button3 target legality repair.
try:
    import numpy as np
    from agent_ppo.feature.action_mask import adjust_raw_legal_action_for_button_targets
    labels = list(Config.LABEL_SIZE_LIST)
    legal_sizes = list(Config.LEGAL_ACTION_SIZE_LIST)
    button_size = labels[0]
    target_size = labels[-1]
    raw_target_size = button_size * target_size
    legal_action = np.ones(sum(legal_sizes), dtype=np.float32)
    target_matrix = legal_action[-raw_target_size:].reshape(button_size, target_size)
    target_matrix[3, :] = 0
    target_matrix[3, 0] = 1
    target_matrix[3, 2] = 1
    adjusted, stats = adjust_raw_legal_action_for_button_targets(legal_action, return_stats=True)
    adjusted_matrix = adjusted[-raw_target_size:].reshape(button_size, target_size)
    out["probes"]["normal_attack_mask"] = {
        "available": True,
        "button3_after": float(adjusted[3]),
        "target0_after": float(adjusted_matrix[3, 0]),
        "target2_after": float(adjusted_matrix[3, 2]),
        "stats": {k: float(v) for k, v in dict(stats).items()},
    }
except Exception as exc:
    out["probes"]["normal_attack_mask"] = {"available": False}
    record_error("normal_attack_mask", exc)

# Probe 2: Soldier target slot order.
try:
    from agent_ppo.feature.targeting import target_slot_enemy_soldiers
    npcs = [
        make_tower(1, x=-15000),
        make_tower(2, x=15000),
        make_minion(30, 2, x=300),
        make_minion(10, 2, x=900),
        make_minion(20, 2, x=600),
    ]
    slots = target_slot_enemy_soldiers(npcs, (0, 0), 1, 4)
    out["probes"]["soldier_target_order"] = {
        "available": True,
        "input_runtime_ids": [30, 10, 20],
        "distance_order_runtime_ids": [30, 20, 10],
        "runtime_id_order": [10, 20, 30],
        "observed_slot_runtime_ids": [item["unit"]["runtime_id"] for item in slots],
    }
except Exception as exc:
    out["probes"]["soldier_target_order"] = {"available": False}
    record_error("soldier_target_order", exc)

# Probe 3: Recall reward in a safe low-HP state, with recall explicitly enabled
# for versions that support it.
try:
    from agent_ppo.feature.reward_process import GameRewardManager
    old_enabled = getattr(GameConfig, "RECALL_ENABLED", None)
    old_weight = GameConfig.REWARD_WEIGHT_DICT.get("recall_recover")
    if hasattr(GameConfig, "RECALL_ENABLED"):
        GameConfig.RECALL_ENABLED = True
    if "recall_recover" in GameConfig.REWARD_WEIGHT_DICT:
        GameConfig.REWARD_WEIGHT_DICT["recall_recover"] = 1.0
    manager = GameRewardManager(101)
    low_hp_safe = make_frame(
        main=make_hero(101, 1, hp=320, max_hp=1000, x=-14500),
        enemy=make_hero(202, 2, hp=1000, max_hp=1000, attack_range=5000, x=16000),
        npcs=[make_tower(1, x=-15000), make_tower(2, x=15000)],
        frame_no=100,
    )
    if hasattr(manager, "set_distance_penalty"):
        manager.set_distance_penalty([getattr(GameConfig, "RECALL_BUTTON", 9), 0, 0, 0, 0, 0], low_hp_safe)
    reward = manager.result(low_hp_safe)
    stats = manager.consume_monitor_stats() if hasattr(manager, "consume_monitor_stats") else {}
    out["probes"]["recall_low_hp_safe"] = {
        "available": True,
        "recall_button": getattr(GameConfig, "RECALL_BUTTON", 9),
        "recall_enabled_for_probe": getattr(GameConfig, "RECALL_ENABLED", None),
        "recall_reward_weight_for_probe": GameConfig.REWARD_WEIGHT_DICT.get("recall_recover"),
        "reward_sum": float(reward.get("reward_sum", 0.0)),
        "recall_recover": float(reward.get("recall_recover", 0.0)),
        "retreat_recover": float(reward.get("retreat_recover", 0.0)),
        "stats": {k: float(v) for k, v in dict(stats).items() if "recall" in k},
    }
    if old_enabled is not None:
        GameConfig.RECALL_ENABLED = old_enabled
    if old_weight is not None:
        GameConfig.REWARD_WEIGHT_DICT["recall_recover"] = old_weight
except Exception as exc:
    out["probes"]["recall_low_hp_safe"] = {"available": False}
    record_error("recall_low_hp_safe", exc)

# Probe 4: Arli mark feature availability from a real diagnostic frame.
try:
    from agent_ppo.feature.feature_process import FeatureProcess
    diag = Path(os.environ["KAIWU_PROBE_REPO_ROOT"]) / "diag_feature_probes" / "episode_03" / "frame_00914.json"
    data = json.loads(diag.read_text(encoding="utf-8"))
    obs = copy.deepcopy(next(iter(data["observation"].values())))
    process = FeatureProcess(obs.get("camp", 1))
    feature = process.process_feature(obs)
    result = {
        "available": True,
        "feature_len": len(feature),
        "has_arli_mark_field": False,
        "mark_values": [],
    }
    if hasattr(FeatureConfig, "MINION_FIELD_SLICES") and "arli_mark" in FeatureConfig.MINION_FIELD_SLICES:
        result["has_arli_mark_field"] = True
        mark_slice = FeatureConfig.MINION_FIELD_SLICES["arli_mark"]
        token_slices = getattr(FeatureConfig, "TOKEN_SLICES", {})
        for type_key in ("own_minions", "enemy_minions"):
            for idx, token_range in enumerate(token_slices.get(type_key, ())):
                token = feature[token_range]
                values = list(token[mark_slice])
                result["mark_values"].append({
                    "type": type_key,
                    "index": idx,
                    "values": [float(v) for v in values],
                    "nonzero": any(abs(float(v)) > 1e-9 for v in values),
                })
    out["probes"]["arli_mark_feature"] = result
except Exception as exc:
    out["probes"]["arli_mark_feature"] = {"available": False}
    record_error("arli_mark_feature", exc)

print(json.dumps(out, sort_keys=True))
'''


def run(cmd: list[str], cwd: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        **kwargs,
    )


def git_rev_parse(ref: str) -> str:
    proc = run(["git", "rev-parse", ref], REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def run_inner(cwd: Path) -> dict:
    env = os.environ.copy()
    env["KAIWU_PROBE_REPO_ROOT"] = str(REPO_ROOT)
    proc = subprocess.run(
        [PYTHON, "-c", INNER_PROBE],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    payload = {
        "returncode": proc.returncode,
        "stderr": proc.stderr,
    }
    stdout = proc.stdout.strip()
    if stdout:
        try:
            payload.update(json.loads(stdout.splitlines()[-1]))
        except json.JSONDecodeError:
            payload["parse_error"] = stdout
    else:
        payload["parse_error"] = "empty stdout"
    return payload


def add_worktree(commit: str, label: str) -> Path:
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    full = git_rev_parse(commit)
    path = WORKTREE_ROOT / f"{label}-{full[:12]}"
    if path.exists():
        proc = run(["git", "worktree", "remove", "--force", str(path)], REPO_ROOT)
        if proc.returncode != 0:
            shutil.rmtree(path, ignore_errors=True)
    proc = run(["git", "worktree", "add", "--detach", "--force", str(path), full], REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    current_result = run_inner(REPO_ROOT)
    current_result.update(
        {
            "kind": "working_tree",
            "commit": git_rev_parse("HEAD"),
            "label": "current_working_tree",
            "dirty": bool(run(["git", "status", "--short"], REPO_ROOT).stdout.strip()),
        }
    )
    all_results.append(current_result)
    (RESULTS_DIR / "current_working_tree.json").write_text(
        json.dumps(current_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for commit, label in COMMITS:
        try:
            worktree = add_worktree(commit, label)
            result = run_inner(worktree)
            result.update(
                {
                    "kind": "historical_commit",
                    "commit": git_rev_parse(commit),
                    "label": label,
                    "short_commit": commit,
                    "dirty": False,
                }
            )
        except Exception as exc:
            result = {
                "kind": "historical_commit",
                "label": label,
                "short_commit": commit,
                "ok": False,
                "errors": [
                    {
                        "name": "worktree_or_probe",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            }
        all_results.append(result)
        (RESULTS_DIR / f"{label}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "python": PYTHON,
        "results": all_results,
    }
    (RESULTS_DIR / "historical_probe_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
