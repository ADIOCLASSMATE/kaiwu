import importlib
import sys
import types
import unittest

from agent_diy.conf.conf import FeatureConfig, GameConfig
from agent_diy.feature.reward_process import GameRewardManager


class _FakeMonitorConfigBuilder:
    def __init__(self):
        self.metrics = []

    def title(self, *_args, **_kwargs):
        return self

    def add_group(self, *_args, **_kwargs):
        return self

    def end_group(self):
        return self

    def add_panel(self, *_args, **_kwargs):
        return self

    def end_panel(self):
        return self

    def add_metric(self, metrics_name, *_args, **_kwargs):
        self.metrics.append(metrics_name)
        return self

    def build(self):
        return {"metrics": list(self.metrics)}


def _install_fake_monitor_builder():
    sys.modules.setdefault("kaiwudrl", types.ModuleType("kaiwudrl"))
    sys.modules.setdefault("kaiwudrl.common", types.ModuleType("kaiwudrl.common"))
    sys.modules.setdefault("kaiwudrl.common.monitor", types.ModuleType("kaiwudrl.common.monitor"))
    builder_module = types.ModuleType("kaiwudrl.common.monitor.monitor_config_builder")
    builder_module.MonitorConfigBuilder = _FakeMonitorConfigBuilder
    sys.modules["kaiwudrl.common.monitor.monitor_config_builder"] = builder_module


def _monitor_metric_names():
    _install_fake_monitor_builder()
    module = importlib.import_module("agent_diy.conf.monitor_builder")
    module = importlib.reload(module)
    return set(module.build_monitor()["metrics"])


class AgentDiyMonitorTests(unittest.TestCase):
    def test_monitor_covers_every_feature_stat_key(self):
        metrics = _monitor_metric_names()
        expected = {"feat_nan", "feat_inf", "feat_neg", "feat_frames"}
        for key, _dim, _count in FeatureConfig.TOKEN_SEGMENTS:
            expected.update(
                {
                    f"feat_{key}_exists",
                    f"feat_{key}_mean",
                    f"feat_{key}_std",
                    f"feat_{key}_dead",
                }
            )
        expected.update({"feat_global_mean", "feat_global_std", "feat_global_dead"})

        self.assertEqual(expected - metrics, set())

    def test_reward_monitor_items_match_reward_manager_output_keys(self):
        metrics = _monitor_metric_names()
        reward_metrics = {name for name in metrics if name.startswith("rwd_")}
        expected_rewards = set(GameConfig.REWARD_WEIGHT_DICT)
        expected_rewards.update({"distance_penalty", "terminal"})

        self.assertEqual(reward_metrics, {f"rwd_{name}" for name in expected_rewards})

    def test_reward_manager_monitor_stats_match_health_monitor_items(self):
        metrics = _monitor_metric_names()
        expected = {
            "out_of_range_cnt",
            "out_of_range_rate",
            "out_of_range_sum",
            "attack_action_cnt",
            "idle_triggered",
            "idle_triggered_rate",
            "last_hit_window_cnt",
            "last_hit_window_attack_rate",
            "frontline_presence_rate",
        }
        expected.update({f"action_button_{idx}" for idx in range(12)})
        expected.update({f"action_target_{idx}" for idx in range(9)})
        expected.update(
            {
                "attack_target_none",
                "attack_target_enemy_hero",
                "attack_target_self",
                "attack_target_minion",
                "attack_target_tower",
                "attack_target_monster",
                "attack_target_other",
            }
        )
        manager = GameRewardManager(main_hero_runtime_id=101)

        self.assertEqual(expected - metrics, set())
        self.assertEqual(set(manager.consume_monitor_stats()), expected)

    def test_episode_level_metric_has_monitor_panel(self):
        metrics = _monitor_metric_names()

        self.assertIn("episode_cnt", metrics)


if __name__ == "__main__":
    unittest.main()
