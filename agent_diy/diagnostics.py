#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Low-overhead training diagnostics saved beside model checkpoints.

The platform may not provide plotting libraries, so this module only writes
structured JSON and compressed JSONL artifacts. It is disabled by default and
is controlled with KAIWU_DIAG_* environment variables.
"""

import gzip
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from agent_diy.conf.conf import Config, FeatureConfig
from agent_diy.feature.action_mask import adjust_target_legal_for_button


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _to_numpy(value, dtype=np.float64):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _to_float(value, default=0.0):
    try:
        arr = _to_numpy(value).reshape(-1)
        if arr.size == 0:
            return default
        out = float(arr[0])
        return out if math.isfinite(out) else default
    except Exception:
        try:
            out = float(value)
            return out if math.isfinite(out) else default
        except Exception:
            return default


def _to_int_list(value):
    arr = _to_numpy(value, dtype=np.float64).reshape(-1)
    return [int(x) for x in arr.tolist()]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


@dataclass
class DiagnosticsConfig:
    enabled: bool = False
    frame_stride: int = 30
    episode_interval: int = 20
    max_episode_records: int = 2000
    sample_values: int = 20000

    @classmethod
    def from_env(cls):
        return cls(
            enabled=_env_bool("KAIWU_DIAG_ENABLE", bool(getattr(Config, "DIAG_ENABLE", False))),
            frame_stride=max(1, _env_int("KAIWU_DIAG_FRAME_STRIDE", getattr(Config, "DIAG_FRAME_STRIDE", 30))),
            episode_interval=max(
                1,
                _env_int("KAIWU_DIAG_EPISODE_INTERVAL", getattr(Config, "DIAG_EPISODE_INTERVAL", 20)),
            ),
            max_episode_records=max(0, _env_int("KAIWU_DIAG_MAX_RECORDS", getattr(Config, "DIAG_MAX_RECORDS", 2000))),
            sample_values=max(0, _env_int("KAIWU_DIAG_SAMPLE_VALUES", getattr(Config, "DIAG_SAMPLE_VALUES", 20000))),
        )


class ScalarStats:
    def __init__(self, sample_values=0):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = None
        self.max = None
        self.sample_values = sample_values
        self.samples = []

    def update(self, values):
        arr = _to_numpy(values).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sum_sq += float(np.square(arr).sum())
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        self.min = arr_min if self.min is None else min(self.min, arr_min)
        self.max = arr_max if self.max is None else max(self.max, arr_max)
        if self.sample_values > len(self.samples):
            take = min(self.sample_values - len(self.samples), arr.size)
            self.samples.extend(float(x) for x in arr[:take].tolist())

    def summary(self):
        if self.count == 0:
            return {}
        mean = self.sum / self.count
        var = max(0.0, self.sum_sq / self.count - mean * mean)
        out = {
            "count": self.count,
            "mean": round(mean, 6),
            "std": round(var ** 0.5, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
        }
        if self.samples:
            sample = np.asarray(self.samples, dtype=np.float64)
            out.update(
                {
                    "p01": round(float(np.percentile(sample, 1)), 6),
                    "p50": round(float(np.percentile(sample, 50)), 6),
                    "p99": round(float(np.percentile(sample, 99)), 6),
                }
            )
        return out


class FeatureSegmentStats:
    def __init__(self, dim, count, sample_values):
        self.dim = dim
        self.count = count
        self.frames = 0
        self.exists_sum = 0.0
        self.values = ScalarStats(sample_values=sample_values)
        self.dim_sum = np.zeros(dim * count, dtype=np.float64)
        self.dim_sum_sq = np.zeros(dim * count, dtype=np.float64)

    def update(self, segment):
        arr = _to_numpy(segment).reshape(self.count, self.dim)
        self.frames += 1
        self.exists_sum += float(arr[:, 0].sum())
        flat = arr.reshape(-1)
        self.values.update(flat)
        self.dim_sum += flat
        self.dim_sum_sq += flat * flat

    def summary(self):
        out = self.values.summary()
        if self.frames == 0:
            out.update({"exists_rate": 0.0, "dead_dims": 0})
            return out
        n = float(self.frames)
        dim_mean = self.dim_sum / n
        dim_var = self.dim_sum_sq / n - dim_mean * dim_mean
        dim_std = np.sqrt(np.maximum(dim_var, 0.0))
        out.update(
            {
                "exists_rate": round(self.exists_sum / (self.frames * self.count), 6),
                "dead_dims": int((dim_std < 1e-6).sum()),
            }
        )
        return out


class AgentDiagnostics:
    def __init__(self, config=None):
        self.config = config or DiagnosticsConfig.from_env()
        self._init_feature_stats()
        self.policy_decisions = 0
        self.policy_head_counts = [dict() for _ in Config.LABEL_SIZE_LIST]
        self.policy_entropy = [ScalarStats(sample_values=self.config.sample_values) for _ in Config.LABEL_SIZE_LIST]
        self.policy_legal_counts = [ScalarStats(sample_values=self.config.sample_values) for _ in Config.LABEL_SIZE_LIST]
        self.target_counts = {name: 0 for name, _ in FeatureConfig.TARGET_SLOT_DESC}
        self.greedy_diff_count = 0
        self.train_steps = 0
        self.train_metrics = {}
        self.train_reward = ScalarStats(sample_values=self.config.sample_values)
        self.train_advantage = ScalarStats(sample_values=self.config.sample_values)
        self.train_value = ScalarStats(sample_values=self.config.sample_values)
        self.train_grad_norms = {}
        self.episode_records = []

    @classmethod
    def from_env(cls):
        return cls(DiagnosticsConfig.from_env())

    @property
    def enabled(self):
        return self.config.enabled

    def _init_feature_stats(self):
        self.feature_frames = 0
        self.feature_nan = 0
        self.feature_inf = 0
        self.feature_neg = 0
        self.feature_gt_one = 0
        self.feature_values = ScalarStats(sample_values=self.config.sample_values)
        self.feature_segments = {}
        self._segment_offsets = []
        offset = 0
        for name, dim, count in FeatureConfig.TOKEN_SEGMENTS:
            length = dim * count
            self._segment_offsets.append((name, offset, offset + length))
            self.feature_segments[name] = FeatureSegmentStats(dim, count, self.config.sample_values)
            offset += length
        self.global_values = ScalarStats(sample_values=self.config.sample_values)
        self.global_dim_sum = np.zeros(FeatureConfig.GLOBAL_DIM, dtype=np.float64)
        self.global_dim_sum_sq = np.zeros(FeatureConfig.GLOBAL_DIM, dtype=np.float64)

    def record_feature(self, feature):
        if not self.enabled:
            return
        arr = _to_numpy(feature).reshape(-1)
        self.feature_frames += 1
        self.feature_nan += int(np.isnan(arr).sum())
        self.feature_inf += int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        self.feature_neg += int((finite < 0).sum())
        self.feature_gt_one += int((finite > 1.0).sum())
        self.feature_values.update(finite)
        for name, start, stop in self._segment_offsets:
            self.feature_segments[name].update(arr[start:stop])
        global_part = arr[FeatureConfig.TOKEN_FEATURE_DIM:FeatureConfig.TOKEN_FEATURE_DIM + FeatureConfig.GLOBAL_DIM]
        self.global_values.update(global_part)
        if global_part.size == FeatureConfig.GLOBAL_DIM:
            self.global_dim_sum += global_part
            self.global_dim_sum_sq += global_part * global_part

    def record_policy(self, logits, legal_action, prob, d_prob, action, d_action, value):
        if not self.enabled:
            return
        action = _to_int_list(action)
        d_action = _to_int_list(d_action)
        probs = self._prob_heads(prob)
        legal = self._compressed_legal_action(legal_action, action[0] if action else 0)
        legal_heads = self._split_heads(legal)

        self.policy_decisions += 1
        if action != d_action:
            self.greedy_diff_count += 1
        for index, label_size in enumerate(Config.LABEL_SIZE_LIST):
            action_index = action[index] if index < len(action) else -1
            counts = self.policy_head_counts[index]
            counts[str(action_index)] = counts.get(str(action_index), 0) + 1
            if index < len(probs):
                p = probs[index]
                p = p[np.isfinite(p)]
                if p.size > 0:
                    entropy = -float((p * np.log(np.maximum(p, 1e-12))).sum())
                    self.policy_entropy[index].update([entropy])
            if index < len(legal_heads):
                self.policy_legal_counts[index].update([float(np.asarray(legal_heads[index]).sum())])

        if len(action) >= len(Config.LABEL_SIZE_LIST):
            target_index = action[-1]
            if 0 <= target_index < len(FeatureConfig.TARGET_SLOT_DESC):
                target_name = FeatureConfig.TARGET_SLOT_DESC[target_index][0]
                self.target_counts[target_name] = self.target_counts.get(target_name, 0) + 1

    def record_train_step(self, metrics, reward=None, advantage=None, value=None, grad_norms=None):
        if not self.enabled:
            return
        self.train_steps += 1
        for key, raw_value in metrics.items():
            if isinstance(raw_value, (int, float, np.integer, np.floating)):
                self.train_metrics.setdefault(key, ScalarStats(sample_values=self.config.sample_values)).update([raw_value])
        if reward is not None:
            self.train_reward.update(reward)
        if advantage is not None:
            self.train_advantage.update(advantage)
        if value is not None:
            self.train_value.update(value)
        if grad_norms:
            for key, raw_value in grad_norms.items():
                self.train_grad_norms.setdefault(key, ScalarStats(sample_values=self.config.sample_values)).update([raw_value])

    def record_episode_step(
        self,
        episode,
        frame_no,
        observation,
        action,
        d_action,
        head_entropy=None,
        value=None,
        is_eval=False,
    ):
        if not self.enabled or self.config.max_episode_records <= 0:
            return
        if episode % self.config.episode_interval != 0:
            return
        if frame_no % self.config.frame_stride != 0:
            return
        if len(self.episode_records) >= self.config.max_episode_records:
            return

        record = self._episode_record(episode, frame_no, observation)
        record.update(
            {
                "is_eval": bool(is_eval),
                "action": _to_int_list(action),
                "greedy_action": _to_int_list(d_action),
                "head_entropy": [round(_to_float(x), 6) for x in (head_entropy or [])],
                "value": round(_to_float(value), 6),
            }
        )
        self.episode_records.append(record)

    def save_checkpoint(self, prefix, extra_meta=None):
        if not self.enabled:
            return
        meta = {
            "created_at": int(time.time()),
            "feature_dim": FeatureConfig.FEATURE_DIM,
            "label_size_list": list(Config.LABEL_SIZE_LIST),
            "lstm_time_steps": Config.LSTM_TIME_STEPS,
            "lstm_unit_size": Config.LSTM_UNIT_SIZE,
            "network_name": Config.NETWORK_NAME,
            "diagnostics": {
                "frame_stride": self.config.frame_stride,
                "episode_interval": self.config.episode_interval,
                "max_episode_records": self.config.max_episode_records,
            },
        }
        if extra_meta:
            meta.update(extra_meta)
        _write_json(prefix + ".meta.json", meta)
        _write_json(prefix + ".feature_stats.json", self.feature_summary())
        _write_json(prefix + ".policy_stats.json", self.policy_summary())
        _write_json(prefix + ".train_stats.json", self.train_summary())
        if self.episode_records:
            with gzip.open(prefix + ".episodes.jsonl.gz", "wt", encoding="utf-8") as handle:
                for row in self.episode_records:
                    handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")

    def feature_summary(self):
        global_dead = 0
        if self.feature_frames > 0:
            mean = self.global_dim_sum / self.feature_frames
            var = self.global_dim_sum_sq / self.feature_frames - mean * mean
            global_dead = int((np.sqrt(np.maximum(var, 0.0)) < 1e-6).sum())
        return {
            "frames": self.feature_frames,
            "nan": self.feature_nan,
            "inf": self.feature_inf,
            "neg": self.feature_neg,
            "gt_one": self.feature_gt_one,
            "all": self.feature_values.summary(),
            "segments": {name: stats.summary() for name, stats in self.feature_segments.items()},
            "global": {
                **self.global_values.summary(),
                "dead_dims": global_dead,
            },
        }

    def policy_summary(self):
        heads = []
        for index in range(len(Config.LABEL_SIZE_LIST)):
            heads.append(
                {
                    "action_counts": self.policy_head_counts[index],
                    "entropy": self.policy_entropy[index].summary(),
                    "legal_count": self.policy_legal_counts[index].summary(),
                }
            )
        target_slots = {}
        for name, count in self.target_counts.items():
            target_slots[name] = {
                "count": count,
                "rate": round(count / self.policy_decisions, 6) if self.policy_decisions else 0.0,
            }
        return {
            "decisions": self.policy_decisions,
            "sample_greedy_diff_rate": round(self.greedy_diff_count / self.policy_decisions, 6)
            if self.policy_decisions else 0.0,
            "heads": heads,
            "target_slots": target_slots,
        }

    def train_summary(self):
        return {
            "steps": self.train_steps,
            "metrics": {key: stats.summary() for key, stats in self.train_metrics.items()},
            "reward": self.train_reward.summary(),
            "advantage": self.train_advantage.summary(),
            "value": self.train_value.summary(),
            "grad_norms": {key: stats.summary() for key, stats in self.train_grad_norms.items()},
        }

    def _prob_heads(self, prob):
        if isinstance(prob, (list, tuple)):
            if len(prob) == 1:
                arr = _to_numpy(prob[0]).reshape(-1)
                return self._split_heads(arr[:Config.LABEL_SUM])
            if len(prob) == len(Config.LABEL_SIZE_LIST):
                return [_to_numpy(p).reshape(-1) for p in prob]
        arr = _to_numpy(prob).reshape(-1)
        if arr.size == Config.LABEL_SUM:
            return self._split_heads(arr)
        return self._split_heads(arr[:Config.LABEL_SUM])

    def _split_heads(self, flat):
        arr = _to_numpy(flat).reshape(-1)
        out = []
        offset = 0
        for size in Config.LABEL_SIZE_LIST:
            out.append(arr[offset:offset + size])
            offset += size
        return out

    def _compressed_legal_action(self, legal_action, action0):
        legal = _to_numpy(legal_action).reshape(-1)
        if legal.size == Config.LABEL_SUM:
            return legal
        raw_target_size = Config.LABEL_SIZE_LIST[-1] * Config.LABEL_SIZE_LIST[0]
        expected_raw = Config.LABEL_SUM - Config.LABEL_SIZE_LIST[-1] + raw_target_size
        if legal.size == expected_raw:
            fixed = legal[:-raw_target_size]
            target = legal[-raw_target_size:].reshape(Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1])
            action0 = int(np.clip(action0, 0, Config.LABEL_SIZE_LIST[0] - 1))
            target_row = adjust_target_legal_for_button(action0, target[action0])
            return np.concatenate([fixed, target_row], axis=0)
        return np.ones(Config.LABEL_SUM, dtype=np.float64)

    def _episode_record(self, episode, frame_no, observation):
        fs = observation.get("frame_state", {})
        camp = observation.get("camp")
        own, enemy = None, None
        for hero in fs.get("hero_states", []):
            if hero.get("camp") == camp:
                own = hero
            else:
                enemy = hero

        record = {
            "episode": int(episode),
            "frame_no": int(frame_no),
            "camp": camp,
            "reward": _to_float(observation.get("reward", {}).get("reward_sum", 0.0)),
        }
        if own is not None:
            max_hp = own.get("max_hp", 0) or 1
            loc = own.get("location", {}) or {}
            record.update(
                {
                    "hp_ratio": round(float(own.get("hp", 0)) / float(max_hp), 6),
                    "money": own.get("money", 0),
                    "level": own.get("level", 0),
                    "pos": [loc.get("x", 0), loc.get("z", 0)],
                }
            )
        if enemy is not None:
            max_hp = enemy.get("max_hp", 0) or 1
            loc = enemy.get("location", {}) or {}
            record.update(
                {
                    "enemy_hp_ratio": round(float(enemy.get("hp", 0)) / float(max_hp), 6),
                    "enemy_money": enemy.get("money", 0),
                    "enemy_level": enemy.get("level", 0),
                    "enemy_pos": [loc.get("x", 0), loc.get("z", 0)],
                }
            )
        return record
