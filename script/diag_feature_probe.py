#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
真实环境特征与 target 槽位主动诊断脚本。

在 Kaiwu WebIDE 终端执行：
  python3 script/diag_feature_probe.py

默认依次运行鲁班、狄仁杰、公孙离镜像对局，输出到：
  diag_feature_probes/<timestamp>/

主要产物：
  SUMMARY.json          汇总、字段出现次数、target 探针结论
  events.jsonl          abilities/buff/attack_target/revive/bullet 的变化事件
  target_probes.json    Soldier1-4 强制动作与 real_cmd actorID 对照
  episode_*/frame_*.json
                        周期采样和事件触发的原始 observation

常用环境变量：
  KAIWU_PROBE_LINEUPS=112:112,133:133,199:199
  KAIWU_PROBE_MAX_FRAME=20000
  KAIWU_PROBE_SAMPLE_GAP=300
  KAIWU_PROBE_TARGET_REPEATS=2
  KAIWU_PROBE_TARGET_MAX_ATTEMPTS=8
  KAIWU_PROBE_TARGET_TTL_STEPS=3
  KAIWU_PROBE_TARGET_COOLDOWN_STEPS=2
  KAIWU_PROBE_BUTTONS=3,4,5,6,8,10,11
  KAIWU_PROBE_OUTPUT_DIR=diag_feature_probes

说明：
  - target 探针只在 legal_action 与 sub_action_mask 都允许该 target 时执行。
  - 发出 Soldier 槽动作后，从后续帧自己的 real_cmd 中提取非零 actorID。
  - 每次探针同时保存敌方小兵的原始顺序、距离顺序和 runtime_id 顺序。
  - bullet 会解析 source_actor，并对同一 runtime_id 的连续观测估算位移速度。
"""

import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from kaiwudrl.common.utils.train_test_utils import run_train_test


HERO_IDS = {112: "Luban", 133: "DiRenjie", 199: "Arli"}
ABILITY_NAMES = [
    "NoControl",
    "NoMove",
    "NoSkill",
    "ImmuneNegative",
    "ImmuneControl",
    "NoMoveRotate",
    "ImmuneCrit",
    "Blindness",
    "MoveProtect",
    "NoRecoverEnergy",
    "Freeze",
    "DeadControl",
    "NoCollisionDetection",
    "NoJointSkill",
    "AbortMove",
    "ForbidSelect",
    "Renewal",
    "Sprint",
    "NoMoveButCanRotate",
    "ForbidSelectBySkillOrg",
    "ImmunePositiveAndPersistFromOtherOriginators",
    "Repressed",
    "ImmuneDeMoveSpeed",
]

LABEL_SIZES = [12, 16, 16, 16, 16, 9]
TARGET_MATRIX_OFFSET = sum(LABEL_SIZES[:-1])
TARGET_SIZE = LABEL_SIZES[-1]
BUTTON_SIZE = LABEL_SIZES[0]

OUTPUT_ROOT = Path(os.environ.get("KAIWU_PROBE_OUTPUT_DIR", "diag_feature_probes"))
OUTPUT_DIR = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
SAMPLE_GAP = int(os.environ.get("KAIWU_PROBE_SAMPLE_GAP", "300"))
MAX_FRAME = int(os.environ.get("KAIWU_PROBE_MAX_FRAME", "20000"))
TARGET_REPEATS = int(os.environ.get("KAIWU_PROBE_TARGET_REPEATS", "2"))
TARGET_MAX_ATTEMPTS = int(os.environ.get("KAIWU_PROBE_TARGET_MAX_ATTEMPTS", "8"))
TARGET_TTL_STEPS = int(os.environ.get("KAIWU_PROBE_TARGET_TTL_STEPS", "3"))
TARGET_COOLDOWN_STEPS = int(os.environ.get("KAIWU_PROBE_TARGET_COOLDOWN_STEPS", "2"))
PROBE_BUTTONS = [
    int(value)
    for value in os.environ.get("KAIWU_PROBE_BUTTONS", "3,4,5,6,8,10,11").split(",")
    if value.strip()
]


def _parse_lineups():
    raw = os.environ.get("KAIWU_PROBE_LINEUPS", "112:112,133:133,199:199")
    lineups = []
    for item in raw.split(","):
        blue, red = item.strip().split(":", 1)
        lineups.append([int(blue), int(red)])
    if not lineups:
        raise ValueError("KAIWU_PROBE_LINEUPS is empty")
    return lineups


LINEUPS = _parse_lineups()


def _safe_value(obj):
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {
            "__type__": type(obj).__name__,
            "length": len(obj),
        }
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return None if math.isnan(value) or math.isinf(value) else value
        if isinstance(obj, np.ndarray):
            return _safe_value(obj.tolist())
    if isinstance(obj, dict):
        return {str(key): _safe_value(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_value(value) for value in obj]
    return str(obj)


def _location_valid(entity):
    location = entity.get("location", {}) or {}
    return abs(location.get("x", 100000)) < 99999 and abs(location.get("z", 100000)) < 99999


def _visible_to(entity, camp):
    if entity.get("camp") == camp:
        return True
    visible = entity.get("camp_visible")
    if visible is None:
        return True
    try:
        return bool(visible[camp - 1])
    except (IndexError, TypeError):
        return True


def _actor_summary(actor):
    if actor is None:
        return None
    return {
        "runtime_id": actor.get("runtime_id"),
        "config_id": actor.get("config_id"),
        "actor_type": actor.get("actor_type"),
        "sub_type": actor.get("sub_type"),
        "camp": actor.get("camp"),
        "hp": actor.get("hp"),
        "location": actor.get("location"),
    }


def _actor_map(frame_state):
    actors = {}
    for actor in (frame_state.get("hero_states", []) or []) + (
        frame_state.get("npc_states", []) or []
    ):
        runtime_id = actor.get("runtime_id")
        if runtime_id is not None:
            actors[runtime_id] = actor
    return actors


def _main_hero(observation):
    frame_state = observation.get("frame_state", {}) or {}
    player_id = observation.get("player_id")
    camp = observation.get("camp")
    for hero in frame_state.get("hero_states", []) or []:
        if hero.get("runtime_id") == player_id:
            return hero
    for hero in frame_state.get("hero_states", []) or []:
        if hero.get("camp") == camp:
            return hero
    return None


def _ability_payload(abilities):
    abilities = list(abilities or [])
    true_indices = [index for index, value in enumerate(abilities) if value]
    return {
        "length": len(abilities),
        "true_indices": true_indices,
        "true_names": [
            ABILITY_NAMES[index] if index < len(ABILITY_NAMES) else f"UnknownAbility{index}"
            for index in true_indices
        ],
    }


def _buff_payload(actor):
    buff_state = actor.get("buff_state", {}) or {}
    return {
        "buff_skills": [
            {
                "config_id": buff.get("configId"),
                "times": buff.get("times"),
                "start_time": buff.get("startTime"),
            }
            for buff in (buff_state.get("buff_skills", []) or [])
        ],
        "buff_marks": [
            {
                "origin_actor_id": mark.get("origin_actorId"),
                "config_id": mark.get("configId"),
                "layer": mark.get("layer"),
            }
            for mark in (buff_state.get("buff_marks", []) or [])
        ],
    }


def _command_actor_ids(real_cmd):
    fields = ("attack_common", "attack_actor", "obj_skill", "dir_skill")
    found = []
    for command_index, command in enumerate(real_cmd or []):
        for field in fields:
            payload = command.get(field, {}) or {}
            actor_id = payload.get("actorID", 0) or 0
            if actor_id:
                found.append(
                    {
                        "command_index": command_index,
                        "command_type": command.get("command_type"),
                        "field": field,
                        "actor_id": actor_id,
                        "payload": payload,
                    }
                )
    return found


def _enemy_minion_orders(observation):
    frame_state = observation.get("frame_state", {}) or {}
    hero = _main_hero(observation)
    if hero is None or not _location_valid(hero):
        return {"raw": [], "distance": [], "runtime_id": []}

    hero_location = hero["location"]
    camp = observation.get("camp")
    raw = []
    for npc_index, npc in enumerate(frame_state.get("npc_states", []) or []):
        if (
            npc.get("actor_type") != 1
            or npc.get("sub_type") != 11
            or npc.get("camp") == camp
            or npc.get("hp", 0) <= 0
            or not _location_valid(npc)
            or not _visible_to(npc, camp)
        ):
            continue
        location = npc["location"]
        distance = math.hypot(
            location["x"] - hero_location["x"],
            location["z"] - hero_location["z"],
        )
        raw.append(
            {
                "npc_index": npc_index,
                "runtime_id": npc.get("runtime_id"),
                "config_id": npc.get("config_id"),
                "hp": npc.get("hp"),
                "location": location,
                "distance": distance,
            }
        )

    return {
        "raw": raw,
        "distance": sorted(raw, key=lambda item: (item["distance"], item["runtime_id"])),
        "runtime_id": sorted(raw, key=lambda item: item["runtime_id"]),
    }


def _target_matrix(legal_action):
    if legal_action is None or len(legal_action) != 184:
        return None
    flat = legal_action[TARGET_MATRIX_OFFSET:]
    return [
        list(flat[button * TARGET_SIZE:(button + 1) * TARGET_SIZE])
        for button in range(BUTTON_SIZE)
    ]


def _legal_direction_action(legal_action, button, target):
    action = [button]
    offset = BUTTON_SIZE
    for size in LABEL_SIZES[1:-1]:
        mask = legal_action[offset:offset + size]
        if len(mask) != size or not any(value > 0.5 for value in mask):
            action.append(size - 1)
        elif mask[size - 1] > 0.5:
            action.append(size - 1)
        else:
            action.append(next(index for index, value in enumerate(mask) if value > 0.5))
        offset += size
    action.append(target)
    return action


class ProbeRecorder:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.events_path = OUTPUT_DIR / "events.jsonl"
        self.events_file = self.events_path.open("w", encoding="utf-8")
        self.counts = Counter()
        self.state = {}
        self.bullet_previous = {}
        self.previous_bullet_ids = {}
        self.target_probes = []
        self.pending_probes = {}
        self.target_completed = Counter()
        self.target_attempted = Counter()
        self.last_probe_step = {}
        self.next_probe_id = 1
        self.saved_frames = set()
        self.episode_summaries = []
        self.field_stats = Counter()
        self.revive_time_values = []

    def emit(self, event_type, episode, agent_id, frame_no, **payload):
        event = {
            "event": event_type,
            "episode": episode,
            "agent_id": str(agent_id),
            "frame_no": frame_no,
            **payload,
        }
        self.events_file.write(
            json.dumps(_safe_value(event), ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.events_file.flush()
        self.counts[event_type] += 1

    def save_snapshot(self, episode, frame_no, observation, reason):
        key = (episode, frame_no)
        if key in self.saved_frames:
            return
        self.saved_frames.add(key)
        episode_dir = OUTPUT_DIR / f"episode_{episode:02d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "reason": reason,
            "observation": _safe_value(observation),
        }
        (episode_dir / f"frame_{frame_no:05d}.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _emit_actor_changes(self, episode, agent_id, observation, actors):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = frame_state.get("frame_no")
        camp = observation.get("camp")
        interesting = False
        tracked = (frame_state.get("hero_states", []) or []) + (
            frame_state.get("npc_states", []) or []
        )

        for actor in tracked:
            runtime_id = actor.get("runtime_id")
            is_hero = actor.get("actor_type") == 0
            is_outer_tower = actor.get("actor_type") == 2 and actor.get("sub_type") == 21
            actor_key = (episode, str(agent_id), runtime_id)

            abilities = tuple(bool(value) for value in (actor.get("abilities", []) or []))
            ability_key = actor_key + ("abilities",)
            previous = self.state.get(ability_key)
            should_track_ability = is_hero or any(abilities) or previous is not None
            if should_track_ability and previous != abilities:
                self.state[ability_key] = abilities
                ability_payload = _ability_payload(abilities)
                changed = []
                if previous is not None:
                    length = max(len(previous), len(abilities))
                    changed = [
                        index
                        for index in range(length)
                        if (previous[index] if index < len(previous) else False)
                        != (abilities[index] if index < len(abilities) else False)
                    ]
                self.emit(
                    "abilities_change",
                    episode,
                    agent_id,
                    frame_no,
                    actor=_actor_summary(actor),
                    visible_to_agent=_visible_to(actor, camp),
                    changed_indices=changed,
                    abilities=ability_payload,
                )
                for index in ability_payload["true_indices"]:
                    self.field_stats[f"ability_true:index={index}"] += 1
                interesting = interesting or any(abilities)

            buff_payload = _buff_payload(actor)
            buff_signature = json.dumps(buff_payload, sort_keys=True, separators=(",", ":"))
            buff_key = actor_key + ("buff",)
            previous = self.state.get(buff_key)
            has_buff = bool(buff_payload["buff_skills"] or buff_payload["buff_marks"])
            should_track_buff = is_hero or has_buff or previous is not None
            if should_track_buff and previous != buff_signature:
                self.state[buff_key] = buff_signature
                self.emit(
                    "buff_change",
                    episode,
                    agent_id,
                    frame_no,
                    actor=_actor_summary(actor),
                    visible_to_agent=_visible_to(actor, camp),
                    **buff_payload,
                )
                for buff in buff_payload["buff_skills"]:
                    self.field_stats[f"buff_skill:id={buff['config_id']}"] += 1
                for mark in buff_payload["buff_marks"]:
                    self.field_stats[f"buff_mark:id={mark['config_id']}"] += 1
                interesting = interesting or has_buff

            attack_target = actor.get("attack_target", 0) or 0
            attack_key = actor_key + ("attack_target",)
            previous = self.state.get(attack_key)
            should_track_attack = is_hero or is_outer_tower or attack_target != 0 or previous
            if should_track_attack and previous != attack_target:
                self.state[attack_key] = attack_target
                target = actors.get(attack_target)
                main_hero = _main_hero(observation)
                self.emit(
                    "attack_target_change",
                    episode,
                    agent_id,
                    frame_no,
                    source=_actor_summary(actor),
                    target=_actor_summary(target),
                    attack_target=attack_target,
                    target_is_main_hero=bool(
                        main_hero is not None
                        and attack_target == main_hero.get("runtime_id")
                    ),
                )
                if attack_target:
                    source_type = f"{actor.get('actor_type')}/{actor.get('sub_type')}"
                    self.field_stats[f"attack_target:source={source_type}"] += 1
                    if main_hero is not None and attack_target == main_hero.get("runtime_id"):
                        self.field_stats["attack_target:targets_main_hero"] += 1
                interesting = interesting or attack_target != 0

            if is_hero:
                revive_state = (actor.get("hp", 0), actor.get("revive_time", 0) or 0)
                revive_key = actor_key + ("revive",)
                previous = self.state.get(revive_key)
                if previous != revive_state:
                    self.state[revive_key] = revive_state
                    if (
                        previous is None
                        or revive_state[0] <= 0
                        or revive_state[1] != 0
                        or (previous and previous[0] <= 0)
                    ):
                        self.emit(
                            "revive_state",
                            episode,
                            agent_id,
                            frame_no,
                            actor=_actor_summary(actor),
                            revive_time=revive_state[1],
                            previous_hp=previous[0] if previous else None,
                            previous_revive_time=previous[1] if previous else None,
                        )
                        self.revive_time_values.append(revive_state[1])
                    interesting = interesting or revive_state[0] <= 0 or revive_state[1] != 0

        return interesting

    def _emit_bullets(self, episode, agent_id, observation, actors):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = frame_state.get("frame_no")
        camp = observation.get("camp")
        bullet_ids = set()
        saw_new_bullet = False

        for bullet in frame_state.get("bullets", []) or []:
            runtime_id = bullet.get("runtime_id")
            bullet_ids.add(runtime_id)
            source = actors.get(bullet.get("source_actor"))
            location = bullet.get("location", {}) or {}
            previous_key = (episode, str(agent_id), runtime_id)
            previous = self.bullet_previous.get(previous_key)
            is_new_bullet = previous is None
            saw_new_bullet = saw_new_bullet or is_new_bullet
            motion = None
            if previous is not None and frame_no != previous["frame_no"]:
                delta_frame = frame_no - previous["frame_no"]
                motion = {
                    "delta_frame": delta_frame,
                    "delta_x": location.get("x", 0) - previous["x"],
                    "delta_z": location.get("z", 0) - previous["z"],
                    "velocity_x_per_frame": (
                        location.get("x", 0) - previous["x"]
                    ) / delta_frame,
                    "velocity_z_per_frame": (
                        location.get("z", 0) - previous["z"]
                    ) / delta_frame,
                }
            self.bullet_previous[previous_key] = {
                "frame_no": frame_no,
                "x": location.get("x", 0),
                "z": location.get("z", 0),
            }
            self.emit(
                "bullet_observation",
                episode,
                agent_id,
                frame_no,
                bullet=bullet,
                source=_actor_summary(source),
                source_kind=(
                    "hero"
                    if source and source.get("actor_type") == 0
                    else "npc"
                    if source
                    else "unresolved"
                ),
                source_is_enemy=bool(source and source.get("camp") != camp),
                motion=motion,
            )
            source_kind = (
                "hero"
                if source and source.get("actor_type") == 0
                else "npc"
                if source
                else "unresolved"
            )
            relation = "enemy" if source and source.get("camp") != camp else "own_or_unknown"
            self.field_stats[f"bullet:source={source_kind}:{relation}"] += 1
            self.field_stats[
                f"bullet:slot={bullet.get('slot_type')}:skill={bullet.get('skill_id')}"
            ] += 1
            if is_new_bullet:
                self.field_stats["bullet:new_runtime_id"] += 1

        bullet_set_key = (episode, str(agent_id))
        previous_ids = self.previous_bullet_ids.get(bullet_set_key, set())
        for runtime_id in sorted(previous_ids - bullet_ids):
            self.emit(
                "bullet_disappeared",
                episode,
                agent_id,
                frame_no,
                bullet_runtime_id=runtime_id,
            )
        self.previous_bullet_ids[bullet_set_key] = bullet_ids
        return saw_new_bullet

    def record_observation(self, episode, agent_id, observation):
        if not isinstance(observation, dict):
            self.field_stats[
                f"observation:skipped_type={type(observation).__name__}"
            ] += 1
            return
        frame_state = observation.get("frame_state", {}) or {}
        if not isinstance(frame_state, dict):
            self.field_stats[
                f"frame_state:skipped_type={type(frame_state).__name__}"
            ] += 1
            return
        if not frame_state.get("hero_states"):
            return
        frame_no = frame_state.get("frame_no", 0)
        actors = _actor_map(frame_state)
        interesting = self._emit_actor_changes(
            episode, agent_id, observation, actors
        )
        interesting = self._emit_bullets(
            episode, agent_id, observation, actors
        ) or interesting
        self.resolve_pending_probe(episode, agent_id, observation)

        if interesting:
            self.save_snapshot(episode, frame_no, {str(agent_id): observation}, "event")

    def _probe_key(self, observation, target_slot):
        hero = _main_hero(observation)
        hero_id = hero.get("config_id") if hero else None
        return hero_id, target_slot

    def choose_probe_action(self, episode, agent_id, observation, step_index):
        pending_key = (episode, str(agent_id))
        if pending_key in self.pending_probes:
            return None
        if step_index - self.last_probe_step.get(pending_key, -100000) < TARGET_COOLDOWN_STEPS:
            return None

        hero = _main_hero(observation)
        if hero is None or hero.get("hp", 0) <= 0 or not _location_valid(hero):
            return None
        legal_action = observation.get("legal_action", []) or []
        matrix = _target_matrix(legal_action)
        if matrix is None:
            self.field_stats["target_probe:unexpected_legal_action_shape"] += 1
            return None
        sub_action_mask = observation.get("sub_action_mask", {}) or {}
        orders = _enemy_minion_orders(observation)
        self.field_stats[f"target_probe:visible_enemy_minions={len(orders['distance'])}"] += 1

        candidates = []
        for target_slot in range(3, 7):
            rank = target_slot - 3
            if rank >= len(orders["distance"]):
                continue
            completed = self.target_completed[self._probe_key(observation, target_slot)]
            attempted = self.target_attempted[self._probe_key(observation, target_slot)]
            if completed >= TARGET_REPEATS:
                continue
            if attempted >= TARGET_MAX_ATTEMPTS:
                continue
            legal_buttons = []
            for button in PROBE_BUTTONS:
                button_mask = sub_action_mask.get(
                    str(button), sub_action_mask.get(button, [])
                )
                target_is_used = len(button_mask) > 5 and button_mask[5] > 0.5
                if (
                    button < len(matrix)
                    and legal_action[button] > 0.5
                    and matrix[button][target_slot] > 0.5
                    and target_is_used
                ):
                    legal_buttons.append(button)
            if any(matrix[button][target_slot] > 0.5 for button in range(len(matrix))):
                self.field_stats[f"target_probe:mask_has_slot={target_slot}"] += 1
            if legal_buttons:
                self.field_stats[f"target_probe:actionable_slot={target_slot}"] += 1
            if legal_buttons:
                button = legal_buttons[attempted % len(legal_buttons)]
                candidates.append((completed, attempted, target_slot, button))

        if not candidates:
            return None

        _, _, target_slot, button = min(candidates)
        action = _legal_direction_action(legal_action, button, target_slot)
        frame_no = observation["frame_state"].get("frame_no")
        probe_id = self.next_probe_id
        self.next_probe_id += 1
        record = {
            "probe_id": probe_id,
            "episode": episode,
            "agent_id": str(agent_id),
            "hero": _actor_summary(hero),
            "issued_frame": frame_no,
            "issued_step": step_index,
            "action": action,
            "button": button,
            "target_slot": target_slot,
            "target_name": f"Soldier{target_slot - 2}",
            "legal_target_mask": matrix[button],
            "enemy_minion_orders": orders,
            "expected_by_distance": orders["distance"][target_slot - 3],
            "expected_by_raw_order": (
                orders["raw"][target_slot - 3]
                if target_slot - 3 < len(orders["raw"])
                else None
            ),
            "expected_by_runtime_id": (
                orders["runtime_id"][target_slot - 3]
                if target_slot - 3 < len(orders["runtime_id"])
                else None
            ),
            "observations_after_issue": [],
            "status": "pending",
        }
        self.target_attempted[self._probe_key(observation, target_slot)] += 1
        self.pending_probes[pending_key] = record
        self.last_probe_step[pending_key] = step_index
        self.emit(
            "target_probe_issued",
            episode,
            agent_id,
            frame_no,
            probe_id=record["probe_id"],
            action=action,
            target_slot=target_slot,
            target_name=record["target_name"],
            enemy_minion_orders=orders,
        )
        return action

    def resolve_pending_probe(self, episode, agent_id, observation):
        pending_key = (episode, str(agent_id))
        record = self.pending_probes.get(pending_key)
        if record is None:
            return

        frame_state = observation.get("frame_state", {}) or {}
        frame_no = frame_state.get("frame_no")
        hero = _main_hero(observation)
        real_cmd = hero.get("real_cmd", []) if hero else []
        actor_ids = _command_actor_ids(real_cmd)
        actors = _actor_map(frame_state)
        attack_target = hero.get("attack_target", 0) if hero else 0
        observation_record = {
            "frame_no": frame_no,
            "real_cmd": real_cmd,
            "command_actor_ids": actor_ids,
            "hero_attack_target": attack_target,
            "resolved_command_actors": [
                {
                    **item,
                    "actor": _actor_summary(actors.get(item["actor_id"])),
                }
                for item in actor_ids
            ],
        }
        record["observations_after_issue"].append(observation_record)

        unique_ids = sorted({item["actor_id"] for item in actor_ids})
        if unique_ids:
            record["status"] = "resolved"
            record["resolved_frame"] = frame_no
            record["actual_actor_ids"] = unique_ids
            record["actual_actors"] = [
                _actor_summary(actors.get(actor_id)) for actor_id in unique_ids
            ]
            expected_distance_id = record["expected_by_distance"]["runtime_id"]
            expected_raw = record.get("expected_by_raw_order")
            expected_runtime = record.get("expected_by_runtime_id")
            record["matches_distance_order"] = expected_distance_id in unique_ids
            record["matches_raw_order"] = bool(
                expected_raw and expected_raw["runtime_id"] in unique_ids
            )
            record["matches_runtime_id_order"] = bool(
                expected_runtime and expected_runtime["runtime_id"] in unique_ids
            )
            self._finish_probe(pending_key, record, observation)
            return

        if len(record["observations_after_issue"]) >= TARGET_TTL_STEPS:
            record["status"] = "unresolved"
            record["resolved_frame"] = frame_no
            record["actual_actor_ids"] = []
            record["actual_actors"] = []
            record["hero_attack_target_at_expiry"] = attack_target
            self._finish_probe(pending_key, record, observation)

    def _finish_probe(self, pending_key, record, observation):
        self.target_probes.append(record)
        if record["status"] == "resolved":
            self.target_completed[self._probe_key(observation, record["target_slot"])] += 1
        self.pending_probes.pop(pending_key, None)
        self.emit(
            "target_probe_finished",
            record["episode"],
            record["agent_id"],
            record.get("resolved_frame"),
            probe_id=record["probe_id"],
            status=record["status"],
            target_slot=record["target_slot"],
            actual_actor_ids=record.get("actual_actor_ids", []),
            matches_distance_order=record.get("matches_distance_order"),
            matches_raw_order=record.get("matches_raw_order"),
            matches_runtime_id_order=record.get("matches_runtime_id_order"),
        )
        self.save_snapshot(
            record["episode"],
            record.get("resolved_frame", record["issued_frame"]),
            {record["agent_id"]: observation},
            "target_probe",
        )

    def has_pending_probe(self, episode, agent_id):
        return (episode, str(agent_id)) in self.pending_probes

    def finish_episode(self, episode, lineup, frame_no, terminated, truncated):
        for pending_key, record in list(self.pending_probes.items()):
            if record["episode"] != episode:
                continue
            record["status"] = "episode_ended"
            record["resolved_frame"] = frame_no
            self.target_probes.append(record)
            self.pending_probes.pop(pending_key, None)
        self.episode_summaries.append(
            {
                "episode": episode,
                "lineup": lineup,
                "frame_no": frame_no,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    def close(self):
        self.events_file.close()
        (OUTPUT_DIR / "target_probes.json").write_text(
            json.dumps(_safe_value(self.target_probes), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        target_summary = {}
        for probe in self.target_probes:
            hero_id = (probe.get("hero") or {}).get("config_id")
            key = f"{hero_id}:{probe.get('target_name')}"
            bucket = target_summary.setdefault(
                key,
                {
                    "hero_id": hero_id,
                    "hero_name": HERO_IDS.get(hero_id, str(hero_id)),
                    "target_name": probe.get("target_name"),
                    "total": 0,
                    "resolved": 0,
                    "matches_distance_order": 0,
                    "matches_raw_order": 0,
                    "matches_runtime_id_order": 0,
                },
            )
            bucket["total"] += 1
            if probe.get("status") == "resolved":
                bucket["resolved"] += 1
            for field in (
                "matches_distance_order",
                "matches_raw_order",
                "matches_runtime_id_order",
            ):
                if probe.get(field):
                    bucket[field] += 1

        summary = {
            "output_dir": str(OUTPUT_DIR),
            "config": {
                "lineups": LINEUPS,
                "max_frame": MAX_FRAME,
                "sample_gap": SAMPLE_GAP,
                "target_repeats": TARGET_REPEATS,
                "target_max_attempts": TARGET_MAX_ATTEMPTS,
                "target_ttl_steps": TARGET_TTL_STEPS,
                "target_cooldown_steps": TARGET_COOLDOWN_STEPS,
                "probe_buttons": PROBE_BUTTONS,
            },
            "episodes": self.episode_summaries,
            "event_counts": dict(sorted(self.counts.items())),
            "field_stats": dict(sorted(self.field_stats.items())),
            "revive_time_values": self.revive_time_values,
            "target_probe_summary": target_summary,
            "interpretation": {
                "distance_order_confirmed_when": (
                    "同一 hero/slot 多次 resolved，且 matches_distance_order 持续为 true"
                ),
                "abilities": "查看 abilities_change 的 true_indices/true_names",
                "buffs": "查看 buff_change 中 buff_skills 与 buff_marks 的动态变化",
                "attack_target": (
                    "查看 attack_target_change，特别是 target_is_main_hero=true 的英雄/塔"
                ),
                "revive": "查看死亡期 revive_state 的 revive_time 序列",
                "bullets": (
                    "查看 bullet_observation 的 source_kind/source_is_enemy/motion"
                ),
            },
        }
        (OUTPUT_DIR / "SUMMARY.json").write_text(
            json.dumps(_safe_value(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main():
    import agent_diy.workflow.train_workflow as tw

    recorder = ProbeRecorder()

    def probe_workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
        env = envs[0]
        env_conf_manager = tw.EnvConfManager(
            config_path="agent_diy/conf/train_env_conf.toml",
            logger=logger,
        )
        runner = tw.EpisodeRunner(
            env=env,
            agents=agents,
            logger=logger,
            monitor=monitor,
            env_conf_manager=env_conf_manager,
            lineup_iterator=iter(LINEUPS),
        )

        try:
            for episode, lineup in enumerate(LINEUPS, start=1):
                usr_conf, is_eval, monitor_side = runner.env_conf_manager.update_config(lineup)
                runner._call_init_config(usr_conf)
                env_obs = env.reset(usr_conf=usr_conf)
                observation = env_obs["observation"]
                runner.reset_agents(observation)

                logger.info(
                    f"[feature_probe] episode={episode} lineup={lineup} "
                    f"monitor_side={monitor_side} is_eval={is_eval}"
                )

                frame_no = 0
                step_index = 0
                terminated = False
                truncated = False

                for agent_id in range(runner.agent_num):
                    obs = observation.get(str(agent_id))
                    if isinstance(obs, dict):
                        recorder.record_observation(episode, agent_id, obs)
                recorder.save_snapshot(episode, 0, observation, "episode_start")

                while True:
                    actions = [tw.NONE_ACTION] * runner.agent_num
                    for index, (do_predict, agent) in enumerate(
                        zip(runner.do_predicts, runner.agents)
                    ):
                        if not do_predict:
                            continue
                        agent_id = str(index)
                        obs = observation.get(agent_id, {})
                        if not (obs.get("frame_state", {}) or {}).get("hero_states"):
                            continue

                        # 保持模型 LSTM 状态正常推进；必要时仅覆盖最终环境动作。
                        model_action = agent.exploit(obs)
                        if recorder.has_pending_probe(episode, agent_id):
                            actions[index] = tw.NONE_ACTION
                            continue
                        probe_action = recorder.choose_probe_action(
                            episode, agent_id, obs, step_index
                        )
                        actions[index] = probe_action or model_action

                    _, env_obs = env.step(actions)
                    frame_no = env_obs["frame_no"]
                    observation = env_obs["observation"]
                    terminated = env_obs["terminated"]
                    truncated = env_obs["truncated"]
                    step_index += 1

                    for agent_id in range(runner.agent_num):
                        obs = observation.get(str(agent_id))
                        if isinstance(obs, dict):
                            recorder.record_observation(episode, agent_id, obs)

                    if SAMPLE_GAP > 0 and frame_no % SAMPLE_GAP == 0:
                        recorder.save_snapshot(
                            episode, frame_no, observation, "periodic"
                        )

                    reached_limit = MAX_FRAME > 0 and frame_no >= MAX_FRAME
                    if terminated or truncated or reached_limit:
                        recorder.save_snapshot(
                            episode, frame_no, observation, "episode_end"
                        )
                        recorder.finish_episode(
                            episode, lineup, frame_no, terminated, truncated
                        )
                        logger.info(
                            f"[feature_probe] episode={episode} ended frame={frame_no} "
                            f"terminated={terminated} truncated={truncated} "
                            f"limit={reached_limit}"
                        )
                        break
        finally:
            recorder.close()
            logger.info(f"[feature_probe] output={OUTPUT_DIR}")

        return

    tw.workflow = probe_workflow

    run_train_test(
        algorithm_name="diy",
        algorithm_name_list=["ppo", "diy"],
        env_vars={
            "replay_buffer_capacity": "128",
            "preload_ratio": "1.0",
            "reverb_remover": "reverb.selectors.Fifo",
            "reverb_sampler": "reverb.selectors.Uniform",
            "reverb_rate_limiter": "MinSize",
            "reverb_samples_per_insert": "8",
            "reverb_error_buffer": "8",
            "train_batch_size": "32",
            "dump_model_freq": "1000",
            "model_file_sync_per_minutes": "1",
            "modelpool_max_save_model_count": "1",
            "preload_model": "False",
            "preload_model_dir": "{agent_name}/ckpt",
            "preload_model_id": "1000",
        },
    )


if __name__ == "__main__":
    main()
