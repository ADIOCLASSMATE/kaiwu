#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib.util
import sys
import tomllib
import types
import unittest
from pathlib import Path


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class _FakeEnvConfManager:
    def __init__(self, monitor_side=0, opponent_agent="selfplay"):
        self._monitor_side = monitor_side
        self._opponent_agent = opponent_agent

    def get_monitor_side(self):
        return self._monitor_side

    def get_opponent_agent(self):
        return self._opponent_agent


class _FakeRewardManager:
    def result(self, _frame_state):
        return {"reward_sum": 0.0}


class _FakeAgent:
    def __init__(self):
        self.loaded = []
        self.reset_observation = None
        self.reward_manager = _FakeRewardManager()

    def load_model(self, id):
        self.loaded.append(("model", id))

    def load_opponent_agent(self, id):
        self.loaded.append(("opponent", id))

    def reset(self, observation):
        self.reset_observation = observation


def _install_module(name, module):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    return previous


def _restore_modules(previous_modules):
    for name, previous in previous_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _load_workflow_module():
    previous_modules = {}

    feature_definition = types.ModuleType("agent_diy.feature.definition")
    feature_definition.sample_process = lambda frame_collector: frame_collector
    feature_definition.build_frame = lambda agent, observation: (agent, observation)
    feature_definition.FrameCollector = object
    feature_definition.NONE_ACTION = [0, 0, 0, 0, 0, 0]
    feature_definition.lineup_iterator_roundrobin_camp_heroes = lambda _heroes: iter(())
    previous_modules["agent_diy.feature.definition"] = _install_module(
        "agent_diy.feature.definition",
        feature_definition,
    )

    env_conf_manager = types.ModuleType("tools.env_conf_manager")

    class _StubEnvConfManager:
        @staticmethod
        def extract_hero_ids_from_usr_conf(_usr_conf):
            return [], []

        @staticmethod
        def inject_select_skills(_usr_conf, _camp_key, _select_skills):
            return None

    env_conf_manager.EnvConfManager = _StubEnvConfManager
    previous_modules["tools.env_conf_manager"] = _install_module(
        "tools.env_conf_manager",
        env_conf_manager,
    )

    model_pool_utils = types.ModuleType("tools.model_pool_utils")
    model_pool_utils.get_valid_model_pool = lambda _logger: [101, 102]
    previous_modules["tools.model_pool_utils"] = _install_module(
        "tools.model_pool_utils",
        model_pool_utils,
    )

    metrics_utils = types.ModuleType("tools.metrics_utils")
    metrics_utils.get_training_metrics = lambda: {}
    previous_modules["tools.metrics_utils"] = _install_module(
        "tools.metrics_utils",
        metrics_utils,
    )

    recovery = types.ModuleType("common_python.utils.workflow_disaster_recovery")
    recovery.handle_disaster_recovery = lambda _env_obs, _logger: False
    previous_modules["common_python.utils.workflow_disaster_recovery"] = _install_module(
        "common_python.utils.workflow_disaster_recovery",
        recovery,
    )

    try:
        path = Path(__file__).resolve().parents[1] / "agent_diy/workflow/train_workflow.py"
        spec = importlib.util.spec_from_file_location("_agent_diy_train_workflow_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(previous_modules)


class AgentDiyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow_module = _load_workflow_module()

    def _runner(self, monitor_side=0):
        runner = self.workflow_module.EpisodeRunner.__new__(
            self.workflow_module.EpisodeRunner
        )
        runner.logger = _FakeLogger()
        runner.env_conf_manager = _FakeEnvConfManager(monitor_side=monitor_side)
        runner.agents = [_FakeAgent(), _FakeAgent()]
        runner.train_opponent_mix = {
            "enable": True,
            "dynamic": False,
            "selfplay": 0.5,
            "common_ai": 0.5,
            "model_pool": 0.0,
            "low_win_selfplay": 0.4,
            "low_win_common_ai": 0.6,
            "mid_win_selfplay": 0.35,
            "mid_win_common_ai": 0.65,
            "high_win_selfplay": 0.5,
            "high_win_common_ai": 0.5,
            "common_ai_low_win_rate": 0.05,
            "common_ai_high_win_rate": 0.25,
        }
        return runner

    def test_prepare_episode_opponent_writes_selected_agent_before_reset(self):
        runner = self._runner()
        runner.train_opponent_mix["selfplay"] = 0.0
        runner.train_opponent_mix["common_ai"] = 1.0
        usr_conf = {"episode": {"opponent_agent": "selfplay"}}

        selected = runner._prepare_episode_opponent(
            usr_conf,
            is_eval=False,
            is_train_test=False,
        )

        self.assertEqual(selected, "common_ai")
        self.assertEqual(usr_conf["episode"]["opponent_agent"], "common_ai")

    def test_prepare_episode_opponent_keeps_eval_configuration(self):
        runner = self._runner()
        usr_conf = {"episode": {"opponent_agent": "selfplay"}}

        selected = runner._prepare_episode_opponent(
            usr_conf,
            is_eval=True,
            is_train_test=False,
        )

        self.assertEqual(selected, "selfplay")
        self.assertEqual(usr_conf["episode"]["opponent_agent"], "selfplay")

    def test_reset_agents_does_not_sample_random_selfplay_opponent(self):
        runner = self._runner(monitor_side=0)
        observation = {"0": {"frame_state": {}}, "1": {"frame_state": {}}}

        runner.reset_agents(
            observation,
            opponent_agent="selfplay",
            is_eval=False,
        )

        self.assertEqual(runner.agents[0].loaded, [("model", "latest")])
        self.assertEqual(runner.agents[1].loaded, [("model", "random")])
        self.assertEqual(runner.do_predicts, [True, True])
        self.assertEqual(runner.do_samples, [True, False])

    def test_reset_agents_disables_prediction_and_sampling_for_common_ai(self):
        runner = self._runner(monitor_side=0)
        observation = {"0": {"frame_state": {}}, "1": {"frame_state": {}}}

        runner.reset_agents(
            observation,
            opponent_agent="common_ai",
            is_eval=False,
        )

        self.assertEqual(runner.agents[0].loaded, [("model", "latest")])
        self.assertEqual(runner.agents[1].loaded, [])
        self.assertEqual(runner.do_predicts, [True, False])
        self.assertEqual(runner.do_samples, [True, False])

    def test_training_opponent_curriculum_anchors_on_common_ai_when_weak(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "agent_diy/conf/train_env_conf.toml"
        )
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        mix = config["episode"]["train_opponent_mix"]
        self.assertTrue(mix["enable"])
        self.assertTrue(mix["dynamic"])
        self.assertEqual(mix["selfplay"], 0.5)
        self.assertEqual(mix["common_ai"], 0.5)
        self.assertEqual(mix["low_win_selfplay"], 0.4)
        self.assertEqual(mix["low_win_common_ai"], 0.6)
        self.assertEqual(mix["mid_win_selfplay"], 0.35)
        self.assertEqual(mix["mid_win_common_ai"], 0.65)
        self.assertEqual(mix["high_win_selfplay"], 0.5)
        self.assertEqual(mix["high_win_common_ai"], 0.5)
        self.assertEqual(config["episode"]["eval_opponent_type"], "common_ai")

    def test_dynamic_opponent_mix_uses_common_ai_win_rate_bands(self):
        runner = self._runner()
        runner.train_opponent_mix["dynamic"] = True

        low = runner._effective_train_opponent_mix(
            {"env": {"common_ai": {"win_rate": 0.01}}}
        )
        mid = runner._effective_train_opponent_mix(
            {"env": {"common_ai": {"win_rate": 0.12}}}
        )
        high = runner._effective_train_opponent_mix(
            {"env": {"common_ai": {"win_rate": 0.31}}}
        )

        self.assertEqual((low["selfplay"], low["common_ai"]), (0.4, 0.6))
        self.assertEqual((mid["selfplay"], mid["common_ai"]), (0.35, 0.65))
        self.assertEqual((high["selfplay"], high["common_ai"]), (0.5, 0.5))

    def test_dynamic_opponent_mix_uses_baseline_when_metrics_are_missing(self):
        runner = self._runner()
        runner.train_opponent_mix["dynamic"] = True

        mix = runner._effective_train_opponent_mix({})

        self.assertEqual((mix["selfplay"], mix["common_ai"]), (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
