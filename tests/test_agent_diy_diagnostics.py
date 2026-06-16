#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent_diy.conf.conf import Config, FeatureConfig as FC
from agent_diy.diagnostics import AgentDiagnostics, DiagnosticsConfig


class AgentDiyDiagnosticsTests(unittest.TestCase):
    def _diagnostics(self, **kwargs):
        config = DiagnosticsConfig(enabled=True, **kwargs)
        return AgentDiagnostics(config=config)

    def test_checkpoint_save_writes_structured_files_without_matplotlib(self):
        diag = self._diagnostics(frame_stride=1, max_episode_records=8)

        feature = np.zeros(FC.FEATURE_DIM, dtype=np.float32)
        feature[FC.TOKEN_SLICES["main_hero"][0].start] = 1.0
        feature[FC.TOKEN_SLICES["enemy_hero"][0].start] = 1.0
        feature[-1] = 1.25
        diag.record_feature(feature)

        logits = np.zeros(Config.LABEL_SUM, dtype=np.float32)
        legal_action = np.ones(Config.LABEL_SUM, dtype=np.float32)
        prob = [
            np.full(label_size, 1.0 / label_size, dtype=np.float32)
            for label_size in Config.LABEL_SIZE_LIST
        ]
        action = [3, 15, 15, 15, 15, 1]
        diag.record_policy(
            logits=logits,
            legal_action=legal_action,
            prob=prob,
            d_prob=prob,
            action=action,
            d_action=action,
            value=np.array([[0.5]], dtype=np.float32),
        )
        diag.record_train_step(
            {
                "total_loss": 1.2,
                "value_loss": 0.3,
                "policy_loss": -0.1,
                "entropy_loss": -1.7,
                "grad_norm": 0.4,
                "learning_rate": 0.001,
            },
            reward=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            advantage=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            value=np.array([0.8, 1.9, 2.7], dtype=np.float32),
            grad_norms={"model": 0.4, "model.lstm": 0.2},
        )
        diag.record_episode_step(
            episode=1,
            frame_no=30,
            observation={
                "camp": 1,
                "frame_state": {
                    "hero_states": [
                        {
                            "camp": 1,
                            "hp": 500,
                            "max_hp": 1000,
                            "money": 1200,
                            "level": 2,
                            "location": {"x": 100, "z": -50},
                        },
                        {
                            "camp": 2,
                            "hp": 300,
                            "max_hp": 1000,
                            "money": 1000,
                            "level": 2,
                            "location": {"x": 300, "z": 50},
                        },
                    ],
                    "npc_states": [],
                },
                "reward": {"reward_sum": 0.25},
            },
            action=action,
            d_action=action,
            head_entropy=[1.0, 2.0, 2.0, 2.0, 2.0, 1.5],
            value=0.5,
            is_eval=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "model.ckpt-7")
            diag.save_checkpoint(prefix, extra_meta={"model_id": "7"})

            expected = [
                "model.ckpt-7.meta.json",
                "model.ckpt-7.feature_stats.json",
                "model.ckpt-7.policy_stats.json",
                "model.ckpt-7.train_stats.json",
                "model.ckpt-7.episodes.jsonl.gz",
            ]
            for name in expected:
                self.assertTrue((Path(tmp) / name).exists(), name)

            meta = json.loads((Path(tmp) / "model.ckpt-7.meta.json").read_text())
            self.assertEqual(meta["model_id"], "7")
            self.assertEqual(meta["feature_dim"], FC.FEATURE_DIM)
            self.assertEqual(meta["label_size_list"], Config.LABEL_SIZE_LIST)

            feature_stats = json.loads((Path(tmp) / "model.ckpt-7.feature_stats.json").read_text())
            self.assertEqual(feature_stats["frames"], 1)
            self.assertEqual(feature_stats["gt_one"], 1)
            self.assertAlmostEqual(feature_stats["segments"]["main_hero"]["exists_rate"], 1.0)

            policy_stats = json.loads((Path(tmp) / "model.ckpt-7.policy_stats.json").read_text())
            self.assertEqual(policy_stats["decisions"], 1)
            self.assertEqual(policy_stats["heads"][0]["action_counts"]["3"], 1)
            self.assertEqual(policy_stats["target_slots"]["EnemyHero"]["count"], 1)

            train_stats = json.loads((Path(tmp) / "model.ckpt-7.train_stats.json").read_text())
            self.assertEqual(train_stats["steps"], 1)
            self.assertIn("advantage", train_stats)
            self.assertIn("grad_norms", train_stats)

            with gzip.open(Path(tmp) / "model.ckpt-7.episodes.jsonl.gz", "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["action"], action)
            self.assertAlmostEqual(rows[0]["hp_ratio"], 0.5)

    def test_disabled_diagnostics_are_noop(self):
        diag = AgentDiagnostics(config=DiagnosticsConfig(enabled=False))
        diag.record_feature(np.ones(FC.FEATURE_DIM, dtype=np.float32))
        diag.record_policy(
            logits=np.zeros(Config.LABEL_SUM, dtype=np.float32),
            legal_action=np.ones(Config.LABEL_SUM, dtype=np.float32),
            prob=[],
            d_prob=[],
            action=[],
            d_action=[],
            value=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "model.ckpt-1")
            diag.save_checkpoint(prefix)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
