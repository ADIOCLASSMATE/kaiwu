import unittest
import sys
import types
from pathlib import Path
from types import SimpleNamespace

try:
    import torch

    from agent_ppo.conf.conf import Config, FeatureConfig, GameConfig
    from agent_ppo.feature.action_mask import (
        adjust_raw_legal_action_for_button_targets,
        adjust_target_legal_for_button,
    )
    from agent_ppo.feature.feature_process import FeatureProcess
    from agent_ppo.feature.reward_process import GameRewardManager
    from agent_ppo.model.model import Model
except ModuleNotFoundError:
    torch = None
    Config = None
    FeatureConfig = None
    GameConfig = None
    adjust_raw_legal_action_for_button_targets = None
    adjust_target_legal_for_button = None
    FeatureProcess = None
    GameRewardManager = None
    Model = None


MAIN_ID = 101
ENEMY_ID = 202


def make_hero(
    runtime_id,
    camp,
    *,
    hp=1000,
    max_hp=1000,
    attack_range=5000,
    x=0,
    z=0,
):
    return {
        "runtime_id": runtime_id,
        "camp": camp,
        "hp": hp,
        "max_hp": max_hp,
        "attack_range": attack_range,
        "location": {"x": x, "z": z},
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
    }


def make_minion(runtime_id, camp, *, hp=1000, x=0, z=0):
    return {
        "runtime_id": runtime_id,
        "actor_type": 1,
        "sub_type": 11,
        "camp": camp,
        "hp": hp,
        "max_hp": 1000,
        "kill_income": 40,
        "location": {"x": x, "z": z},
    }


def make_frame(*, main=None, enemy=None, npcs=None):
    return {
        "frame_no": 0,
        "hero_states": [
            main or make_hero(MAIN_ID, 1, x=-10000),
            enemy or make_hero(ENEMY_ID, 2, x=10000),
        ],
        "npc_states": npcs or [
            make_tower(1, x=-15000),
            make_tower(2, x=15000),
        ],
        "cakes": [],
        "frame_action": {"dead_action": []},
    }


@unittest.skipIf(torch is None, "torch is not installed")
class AgentPpoFeatureSwapTests(unittest.TestCase):
    def _enable_recall_for_test(self):
        old_enabled = GameConfig.RECALL_ENABLED
        old_weight = GameConfig.REWARD_WEIGHT_DICT["recall_recover"]
        GameConfig.RECALL_ENABLED = True
        GameConfig.REWARD_WEIGHT_DICT["recall_recover"] = 1.0
        self.addCleanup(
            lambda: (
                setattr(GameConfig, "RECALL_ENABLED", old_enabled),
                GameConfig.REWARD_WEIGHT_DICT.__setitem__("recall_recover", old_weight),
            )
        )

    @staticmethod
    def _install_base_agent_stub():
        common = sys.modules.setdefault("common_python", types.ModuleType("common_python"))
        utils = sys.modules.setdefault("common_python.utils", types.ModuleType("common_python.utils"))
        common_func = types.ModuleType("common_python.utils.common_func")

        def create_cls(name, **kwargs):
            return type(
                name,
                (),
                {
                    "__init__": lambda self, **values: self.__dict__.update(values),
                    "__repr__": lambda self: "{}({})".format(name, self.__dict__),
                },
            )

        class Frame:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        common_func.create_cls = create_cls
        common_func.Frame = Frame
        sys.modules["common_python.utils.common_func"] = common_func
        common.utils = utils
        utils.common_func = common_func

        kaiwudrl = sys.modules.setdefault("kaiwudrl", types.ModuleType("kaiwudrl"))
        interface = sys.modules.setdefault("kaiwudrl.interface", types.ModuleType("kaiwudrl.interface"))
        agent_module = types.ModuleType("kaiwudrl.interface.agent")

        class BaseAgent:
            def __init__(self, *args, **kwargs):
                pass

        agent_module.BaseAgent = BaseAgent
        sys.modules["kaiwudrl.interface.agent"] = agent_module
        kaiwudrl.interface = interface
        interface.agent = agent_module

    def test_ppo_owns_feature_process(self):
        self.assertEqual(FeatureProcess.__module__, "agent_ppo.feature.feature_process")
        self.assertEqual(Config.FEATURE_DIM, FeatureConfig.FEATURE_DIM)
        self.assertEqual(Config.SERI_VEC_SPLIT_SHAPE, [(FeatureConfig.FEATURE_DIM,), (85,)])

    def test_ppo_model_accepts_migrated_feature_dim(self):
        model = Model()
        model.set_eval_mode()
        feature = torch.zeros(1, Config.FEATURE_DIM)
        hidden = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        cell = torch.zeros(1, Config.LSTM_UNIT_SIZE)

        logits, value, next_cell, next_hidden = model([feature, hidden, cell], inference=True)

        self.assertEqual(tuple(logits.shape), (1, Config.LABEL_SUM))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(next_cell.shape), (1, 1, Config.LSTM_UNIT_SIZE))
        self.assertEqual(tuple(next_hidden.shape), (1, 1, Config.LSTM_UNIT_SIZE))

    def test_ppo_model_uses_diy_token_encoder(self):
        model = Model()
        feature = torch.zeros(2, Config.FEATURE_DIM)

        structured_state, entity_out = model._encode(feature)

        self.assertEqual(model.num_tokens, FeatureConfig.NUM_TOKENS)
        self.assertEqual(model.token_feature_dim, FeatureConfig.TOKEN_FEATURE_DIM)
        self.assertEqual(tuple(structured_state.shape), (2, Config.PPO_ENCODER_OUTPUT_DIM))
        self.assertEqual(tuple(entity_out.shape), (2, FeatureConfig.NUM_TOKENS, Config.EMBED_DIM))
        self.assertGreater(model.token_residual_gate.item(), 0.0)
        self.assertGreater(model.target_pointer_gate.item(), 0.0)
        self.assertGreater(model.lstm_residual_gate.item(), 0.0)

    def test_ppo_model_uses_button_conditioned_target_pointer(self):
        model = Model()
        model.set_train_mode()
        feature = torch.zeros(Config.LSTM_TIME_STEPS, Config.FEATURE_DIM)
        hidden = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        cell = torch.zeros(1, Config.LSTM_UNIT_SIZE)

        result_list = model([feature, hidden, cell], inference=False)

        target_logits_by_button = result_list[-2]
        self.assertEqual(
            tuple(target_logits_by_button.shape),
            (Config.LSTM_TIME_STEPS, Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1]),
        )
        self.assertEqual(tuple(model.lstm_cell_output.shape), (1, 1, Config.LSTM_UNIT_SIZE))
        self.assertEqual(tuple(model.lstm_hidden_output.shape), (1, 1, Config.LSTM_UNIT_SIZE))

    def test_ppo_feature_reuses_global_money_adv_as_retreat_need(self):
        process = FeatureProcess(1)
        observation = {
            "frame_state": make_frame(),
            "retreat_need_active": False,
        }
        retreat_offset = (
            FeatureConfig.TOKEN_FEATURE_DIM
            + FeatureConfig.GLOBAL_RETREAT_NEED_OFFSET
        )

        inactive_feature = process.process_feature(observation)
        observation["retreat_need_active"] = True
        active_feature = process.process_feature(observation)

        self.assertEqual(len(active_feature), FeatureConfig.FEATURE_DIM)
        self.assertEqual(inactive_feature[retreat_offset], 0.0)
        self.assertEqual(active_feature[retreat_offset], 1.0)

    def test_agent_injects_retreat_need_feature_from_reward_state(self):
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        agent = Agent.__new__(Agent)
        agent.reward_manager = GameRewardManager(MAIN_ID)
        agent.feature_processes = FeatureProcess(1)
        agent.lstm_cell = [0.0] * Config.LSTM_UNIT_SIZE
        agent.lstm_hidden = [0.0] * Config.LSTM_UNIT_SIZE
        retreat_offset = (
            FeatureConfig.TOKEN_FEATURE_DIM
            + FeatureConfig.GLOBAL_RETREAT_NEED_OFFSET
        )
        observation = {
            "frame_state": make_frame(
                main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=0),
                enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, attack_range=5000, x=1000),
            ),
            "legal_action": torch.ones(sum(Config.LEGAL_ACTION_SIZE_LIST)).numpy(),
        }

        obs_data = agent.observation_process(observation)

        self.assertTrue(observation["retreat_need_active"])
        self.assertEqual(obs_data.feature[retreat_offset], 1.0)

    def test_agent_samples_target_from_selected_button_logits(self):
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        agent = Agent.__new__(Agent)
        agent.label_size_list = Config.LABEL_SIZE_LIST
        agent.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        logits = torch.zeros(Config.LABEL_SUM).numpy()
        logits[4] = 10.0
        legal_action = torch.ones(sum(Config.LEGAL_ACTION_SIZE_LIST)).numpy()
        target_logits_by_button = torch.zeros(Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1]).numpy()
        target_logits_by_button[4, 7] = 10.0

        _, _, _, d_action = agent._sample_masked_action(
            logits,
            legal_action,
            target_logits_by_button=target_logits_by_button,
        )

        self.assertEqual(d_action[0], 4)
        self.assertEqual(d_action[-1], 7)

    def test_ppo_reward_has_terminal_bonus(self):
        reward = {"reward_sum": 1.25}
        manager = GameRewardManager(main_hero_runtime_id=101)

        bonus = manager.apply_terminal_outcome(reward, {"hero_states": []}, win=1)

        self.assertEqual(reward["terminal"], 1.0)
        self.assertEqual(bonus, GameConfig.TERMINAL_WIN_REWARD * GameConfig.TERMINAL_WIN_MIN_QUALITY)
        self.assertEqual(reward["reward_sum"], 1.25 + bonus)
        self.assertEqual(manager.apply_terminal_outcome(reward, {"hero_states": []}, win=1), 0.0)

    def test_agent_ppo_does_not_import_agent_diy(self):
        repo = Path(__file__).resolve().parents[1]
        ppo_files = [
            path
            for path in (repo / "agent_ppo").rglob("*.py")
            if "__pycache__" not in path.parts
        ]
        offenders = [
            str(path.relative_to(repo))
            for path in ppo_files
            if "agent_diy" in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(offenders, [])

    def test_ppo_action_mask_blocks_normal_attack_without_entity_target(self):
        legal_action = torch.ones(sum(Config.LEGAL_ACTION_SIZE_LIST)).numpy()
        target_size = Config.LABEL_SIZE_LIST[-1]
        button_size = Config.LABEL_SIZE_LIST[0]
        target_matrix = legal_action[-target_size * button_size:].reshape(button_size, target_size)
        target_matrix[3, :] = 0
        target_matrix[3, 0] = 1
        target_matrix[3, 2] = 1

        adjusted, stats = adjust_raw_legal_action_for_button_targets(
            legal_action,
            return_stats=True,
        )

        adjusted_target = adjusted[-target_size * button_size:].reshape(button_size, target_size)
        self.assertEqual(adjusted[3], 0)
        self.assertEqual(adjusted_target[3, 0], 0)
        self.assertEqual(adjusted_target[3, 2], 0)
        self.assertEqual(stats["button3_no_entity_target_legal_cnt"], 1)
        self.assertEqual(stats["button3_masked_no_entity_target_cnt"], 1)

    def test_ppo_action_mask_bans_recall_button_by_default(self):
        legal_action = torch.ones(sum(Config.LEGAL_ACTION_SIZE_LIST)).numpy()
        target_size = Config.LABEL_SIZE_LIST[-1]
        button_size = Config.LABEL_SIZE_LIST[0]
        legal_action[GameConfig.RECALL_BUTTON] = 1
        target_matrix = legal_action[-target_size * button_size:].reshape(button_size, target_size)
        target_matrix[GameConfig.RECALL_BUTTON, :] = 1

        adjusted = adjust_raw_legal_action_for_button_targets(legal_action)

        adjusted_target = adjusted[-target_size * button_size:].reshape(button_size, target_size)
        self.assertEqual(adjusted[GameConfig.RECALL_BUTTON], 0)
        self.assertEqual(adjusted_target[GameConfig.RECALL_BUTTON].sum(), 0)

    def test_ppo_eval_sampling_cannot_choose_recall_when_banned(self):
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        agent = Agent.__new__(Agent)
        agent.label_size_list = Config.LABEL_SIZE_LIST
        agent.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        logits = torch.zeros(Config.LABEL_SUM).numpy()
        logits[GameConfig.RECALL_BUTTON] = 10.0
        legal_action = torch.ones(sum(Config.LEGAL_ACTION_SIZE_LIST)).numpy()
        target_size = Config.LABEL_SIZE_LIST[-1]
        button_size = Config.LABEL_SIZE_LIST[0]
        legal_action[GameConfig.RECALL_BUTTON] = 1
        target_matrix = legal_action[-target_size * button_size:].reshape(button_size, target_size)
        target_matrix[GameConfig.RECALL_BUTTON, :] = 1

        _, _, _, d_action = agent._sample_masked_action(logits, legal_action)

        self.assertNotEqual(d_action[0], GameConfig.RECALL_BUTTON)

    def test_ppo_action_quality_monitor_stats_are_reported(self):
        manager = GameRewardManager(MAIN_ID)
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=900),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 2, hp=150, x=800),
            ],
        )

        manager.set_distance_penalty([1, 7, 8, 9, 10, 0], frame)
        manager.set_distance_penalty([3, 1, 2, 3, 4, 3], frame)
        manager.result(frame)
        stats = manager.consume_monitor_stats()

        self.assertEqual(stats["noop_cnt"], 1)
        self.assertEqual(stats["noop_enemy_in_range_cnt"], 1)
        self.assertEqual(stats["noop_last_hit_window_cnt"], 1)
        self.assertEqual(stats["action_button_1_rate"], 0.5)
        self.assertEqual(stats["action_head_1_7"], 1)
        self.assertEqual(stats["action_head_4_10"], 1)
        self.assertEqual(stats["resolved_attack_cnt"], 1)
        self.assertEqual(stats["attack_in_range_rate"], 1.0)
        self.assertEqual(stats["out_of_range_button_3_cnt"], 0)

    def test_ppo_out_of_range_monitor_breaks_down_button_and_target(self):
        manager = GameRewardManager(MAIN_ID)
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=5000),
        )

        manager.set_distance_penalty([3, 0, 0, 0, 0, 1], frame)
        manager.result(frame)
        stats = manager.consume_monitor_stats()

        self.assertEqual(stats["out_of_range_button_3_cnt"], 1)
        self.assertEqual(stats["out_of_range_button_3_rate"], 1.0)
        self.assertEqual(stats["out_of_range_target_enemy_hero_cnt"], 1)
        self.assertEqual(stats["out_of_range_target_enemy_hero_rate"], 1.0)
        self.assertEqual(stats["attack_far_out_rate"], 1.0)

    def test_ppo_out_of_range_penalty_scales_by_distance(self):
        manager = GameRewardManager(MAIN_ID)
        near_frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=1100),
        )
        mid_frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=1300),
        )
        far_frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=1800),
        )

        self.assertAlmostEqual(
            manager.out_of_range_penalty([3, 0, 0, 0, 0, 1], near_frame),
            -GameConfig.OUT_OF_RANGE_PENALTY * GameConfig.OUT_OF_RANGE_NEAR_MULT,
        )
        self.assertAlmostEqual(
            manager.out_of_range_penalty([3, 0, 0, 0, 0, 1], mid_frame),
            -GameConfig.OUT_OF_RANGE_PENALTY * GameConfig.OUT_OF_RANGE_MID_MULT,
        )
        self.assertAlmostEqual(
            manager.out_of_range_penalty([3, 0, 0, 0, 0, 1], far_frame),
            -GameConfig.OUT_OF_RANGE_PENALTY * GameConfig.OUT_OF_RANGE_FAR_MULT,
        )

    def test_ppo_noop_opportunity_penalty_is_small_and_safe_gated(self):
        manager = GameRewardManager(MAIN_ID)
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=900),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 2, hp=150, x=800),
            ],
        )

        manager.set_distance_penalty([1, 0, 0, 0, 0, 0], frame)
        reward = manager.result(frame)
        stats = manager.consume_monitor_stats()

        self.assertAlmostEqual(reward["distance_penalty"], -GameConfig.NOOP_MAX_PENALTY)
        self.assertAlmostEqual(stats["noop_opportunity_penalty_sum"], -GameConfig.NOOP_MAX_PENALTY)
        self.assertAlmostEqual(stats["action_penalty_sum"], -GameConfig.NOOP_MAX_PENALTY)
        self.assertEqual(stats["out_of_range_cnt"], 0)

        danger_manager = GameRewardManager(MAIN_ID)
        danger_frame = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, x=900),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 2, hp=150, x=800),
            ],
        )

        danger_manager.set_distance_penalty([1, 0, 0, 0, 0, 0], danger_frame)
        danger_reward = danger_manager.result(danger_frame)

        self.assertEqual(danger_reward["distance_penalty"], 0.0)

    def test_ppo_invalid_normal_attack_target_penalty_is_narrow(self):
        manager = GameRewardManager(MAIN_ID)
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=900),
        )

        manager.set_distance_penalty([3, 0, 0, 0, 0, 2], frame)
        reward = manager.result(frame)
        stats = manager.consume_monitor_stats()

        self.assertAlmostEqual(
            reward["distance_penalty"],
            -GameConfig.INVALID_NORMAL_ATTACK_TARGET_PENALTY,
        )
        self.assertAlmostEqual(
            stats["invalid_target_penalty_sum"],
            -GameConfig.INVALID_NORMAL_ATTACK_TARGET_PENALTY,
        )
        self.assertEqual(stats["out_of_range_cnt"], 0)

    def test_ppo_monitor_builder_declares_observability_panels(self):
        repo = Path(__file__).resolve().parents[1]
        monitor_source = (repo / "agent_ppo/conf/monitor_builder.py").read_text(encoding="utf-8")
        for marker in [
            "feat_%s_exists",
            "rwd_tower_hp_point",
            "attack_action_target_%s_rate",
            "button3_entity_target_legal_rate",
            "entropy_\" + en_suffix",
            "adv_mean",
            "grad_norm",
            "win",
            "action_button_%d_rate",
            "out_of_range_button_%d_rate",
            "action_head_%d_%d_rate",
            "noop_enemy_in_range_rate",
            "attack_far_out_rate",
            "action_penalty_sum",
            "noop_opportunity_penalty_sum",
            "invalid_target_penalty_sum",
            "rwd_recall_recover",
            "recall_button_rate_when_needed",
            "recall_interrupt_penalty_sum",
            "recall_explore_button9_prob_avg",
            "recall_explore_forced_legal_cnt",
            "recall_hold_model_keep_rate",
            "recall_hold_assist_cnt",
        ]:
            self.assertIn(marker, monitor_source)

    def test_ppo_recall_exploration_overrides_to_legal_recall_button(self):
        self._enable_recall_for_test()
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        old_enabled = GameConfig.RECALL_EXPLORATION_ENABLED
        old_prob = GameConfig.RECALL_EXPLORATION_PROB
        old_max = GameConfig.RECALL_EXPLORATION_MAX_STARTS_PER_EPISODE
        try:
            GameConfig.RECALL_EXPLORATION_ENABLED = True
            GameConfig.RECALL_EXPLORATION_PROB = 1.0
            GameConfig.RECALL_EXPLORATION_MAX_STARTS_PER_EPISODE = 1
            agent = self._recall_explore_agent()
            observation = self._recall_explore_observation()
            prob = [0.0] * Config.LABEL_SUM
            prob[GameConfig.RECALL_BUTTON] = 0.0007
            act_data = SimpleNamespace(action=[2, 0, 0, 0, 0, 0], prob=[prob])

            action = agent.action_process(observation, act_data, True)
            stats = agent.consume_action_mask_stats()

            self.assertEqual(action[0], GameConfig.RECALL_BUTTON)
            self.assertEqual(act_data.action[0], GameConfig.RECALL_BUTTON)
            self.assertEqual(stats["recall_explore_need_cnt"], 1)
            self.assertEqual(stats["recall_explore_legal_cnt"], 1)
            self.assertEqual(stats["recall_explore_override_cnt"], 1)
            self.assertAlmostEqual(stats["recall_explore_button9_prob_avg"], 0.0007)
        finally:
            GameConfig.RECALL_EXPLORATION_ENABLED = old_enabled
            GameConfig.RECALL_EXPLORATION_PROB = old_prob
            GameConfig.RECALL_EXPLORATION_MAX_STARTS_PER_EPISODE = old_max

    def test_ppo_recall_exploration_uses_adjusted_recall_legal_mask(self):
        self._enable_recall_for_test()
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        old_enabled = GameConfig.RECALL_EXPLORATION_ENABLED
        old_prob = GameConfig.RECALL_EXPLORATION_PROB
        old_force = GameConfig.RECALL_EXPLORATION_FORCE_LEGAL
        try:
            GameConfig.RECALL_EXPLORATION_ENABLED = True
            GameConfig.RECALL_EXPLORATION_PROB = 1.0
            GameConfig.RECALL_EXPLORATION_FORCE_LEGAL = True
            agent = self._recall_explore_agent()
            observation = self._recall_explore_observation()
            observation["legal_action"][GameConfig.RECALL_BUTTON] = 0.0
            target_size = Config.LABEL_SIZE_LIST[-1]
            button_size = Config.LABEL_SIZE_LIST[0]
            target_matrix = observation["legal_action"][-target_size * button_size:].reshape(
                button_size,
                target_size,
            )
            target_matrix[GameConfig.RECALL_BUTTON, :] = 0.0
            act_data = SimpleNamespace(
                action=[2, 0, 0, 0, 0, 0],
                prob=[[0.0] * Config.LABEL_SUM],
            )

            action = agent.action_process(observation, act_data, True)
            stats = agent.consume_action_mask_stats()

            self.assertEqual(action[0], GameConfig.RECALL_BUTTON)
            self.assertEqual(observation["legal_action"][GameConfig.RECALL_BUTTON], 1.0)
            self.assertEqual(
                observation["legal_action"][-target_size * button_size:].reshape(
                    button_size,
                    target_size,
                )[GameConfig.RECALL_BUTTON, 0],
                1.0,
            )
            self.assertEqual(stats["recall_explore_need_cnt"], 1)
            self.assertEqual(stats["recall_explore_legal_cnt"], 1)
            self.assertEqual(stats["recall_explore_forced_legal_cnt"], 0)
            self.assertEqual(stats["recall_explore_override_cnt"], 1)
        finally:
            GameConfig.RECALL_EXPLORATION_ENABLED = old_enabled
            GameConfig.RECALL_EXPLORATION_PROB = old_prob
            GameConfig.RECALL_EXPLORATION_FORCE_LEGAL = old_force

    def test_ppo_recall_exploration_ignores_force_toggle_after_mask_adjustment(self):
        self._enable_recall_for_test()
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        old_enabled = GameConfig.RECALL_EXPLORATION_ENABLED
        old_prob = GameConfig.RECALL_EXPLORATION_PROB
        old_force = GameConfig.RECALL_EXPLORATION_FORCE_LEGAL
        try:
            GameConfig.RECALL_EXPLORATION_ENABLED = True
            GameConfig.RECALL_EXPLORATION_PROB = 1.0
            GameConfig.RECALL_EXPLORATION_FORCE_LEGAL = False
            agent = self._recall_explore_agent()
            observation = self._recall_explore_observation()
            observation["legal_action"][GameConfig.RECALL_BUTTON] = 0.0
            act_data = SimpleNamespace(
                action=[2, 0, 0, 0, 0, 0],
                prob=[[0.0] * Config.LABEL_SUM],
            )

            action = agent.action_process(observation, act_data, True)
            stats = agent.consume_action_mask_stats()

            self.assertEqual(action[0], GameConfig.RECALL_BUTTON)
            self.assertEqual(stats["recall_explore_forced_legal_cnt"], 0)
            self.assertEqual(stats["recall_explore_override_cnt"], 1)
        finally:
            GameConfig.RECALL_EXPLORATION_ENABLED = old_enabled
            GameConfig.RECALL_EXPLORATION_PROB = old_prob
            GameConfig.RECALL_EXPLORATION_FORCE_LEGAL = old_force

    def test_ppo_recall_exploration_holds_active_channel_with_noop(self):
        self._enable_recall_for_test()
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        old_enabled = GameConfig.RECALL_EXPLORATION_ENABLED
        old_hold_assist = GameConfig.RECALL_HOLD_ASSIST_ENABLED
        GameConfig.RECALL_EXPLORATION_ENABLED = True
        GameConfig.RECALL_HOLD_ASSIST_ENABLED = False
        agent = self._recall_explore_agent()
        try:
            agent.reward_manager._recall_channel_steps = 10
            observation = self._recall_explore_observation()
            act_data = SimpleNamespace(
                action=[3, 0, 0, 0, 0, 1],
                prob=[[0.0] * Config.LABEL_SUM],
            )

            action = agent.action_process(observation, act_data, True)
            stats = agent.consume_action_mask_stats()

            self.assertEqual(action[0], GameConfig.RECALL_NOOP_BUTTON)
            self.assertEqual(stats["recall_explore_override_cnt"], 1)
            self.assertEqual(stats["recall_explore_hold_cnt"], 1)
        finally:
            GameConfig.RECALL_EXPLORATION_ENABLED = old_enabled
            GameConfig.RECALL_HOLD_ASSIST_ENABLED = old_hold_assist

    def test_ppo_recall_hold_assist_works_when_exploration_is_disabled(self):
        self._enable_recall_for_test()
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        old_enabled = GameConfig.RECALL_EXPLORATION_ENABLED
        old_hold_assist = GameConfig.RECALL_HOLD_ASSIST_ENABLED
        old_hold_prob = GameConfig.RECALL_HOLD_ASSIST_PROB
        try:
            GameConfig.RECALL_EXPLORATION_ENABLED = False
            GameConfig.RECALL_HOLD_ASSIST_ENABLED = True
            GameConfig.RECALL_HOLD_ASSIST_PROB = 1.0
            agent = self._recall_explore_agent()
            agent.reward_manager._recall_channel_steps = 10
            observation = self._recall_explore_observation()
            act_data = SimpleNamespace(
                action=[3, 0, 0, 0, 0, 1],
                d_action=[3, 0, 0, 0, 0, 1],
                prob=[[0.0] * Config.LABEL_SUM],
            )

            action = agent.action_process(observation, act_data, True)
            stats = agent.consume_action_mask_stats()

            self.assertEqual(action[0], GameConfig.RECALL_NOOP_BUTTON)
            self.assertEqual(stats["recall_hold_active_cnt"], 1)
            self.assertEqual(stats["recall_hold_model_keep_cnt"], 0)
            self.assertEqual(stats["recall_hold_model_keep_rate"], 0.0)
            self.assertEqual(stats["recall_hold_assist_cnt"], 1)
            self.assertEqual(stats["recall_explore_override_cnt"], 0)
            self.assertEqual(stats["recall_explore_hold_cnt"], 0)
        finally:
            GameConfig.RECALL_EXPLORATION_ENABLED = old_enabled
            GameConfig.RECALL_HOLD_ASSIST_ENABLED = old_hold_assist
            GameConfig.RECALL_HOLD_ASSIST_PROB = old_hold_prob

    def test_ppo_recall_hold_assist_can_be_disabled(self):
        self._enable_recall_for_test()
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        old_enabled = GameConfig.RECALL_EXPLORATION_ENABLED
        old_hold_assist = GameConfig.RECALL_HOLD_ASSIST_ENABLED
        try:
            GameConfig.RECALL_EXPLORATION_ENABLED = False
            GameConfig.RECALL_HOLD_ASSIST_ENABLED = False
            agent = self._recall_explore_agent()
            agent.reward_manager._recall_channel_steps = 10
            observation = self._recall_explore_observation()
            act_data = SimpleNamespace(
                action=[3, 0, 0, 0, 0, 1],
                d_action=[3, 0, 0, 0, 0, 1],
                prob=[[0.0] * Config.LABEL_SUM],
            )

            action = agent.action_process(observation, act_data, True)
            stats = agent.consume_action_mask_stats()

            self.assertEqual(action[0], 3)
            self.assertEqual(stats["recall_hold_assist_cnt"], 0)
        finally:
            GameConfig.RECALL_EXPLORATION_ENABLED = old_enabled
            GameConfig.RECALL_HOLD_ASSIST_ENABLED = old_hold_assist

    def test_ppo_recall_hold_assist_reports_model_kept_channel(self):
        self._enable_recall_for_test()
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        old_enabled = GameConfig.RECALL_EXPLORATION_ENABLED
        old_hold_assist = GameConfig.RECALL_HOLD_ASSIST_ENABLED
        try:
            GameConfig.RECALL_EXPLORATION_ENABLED = False
            GameConfig.RECALL_HOLD_ASSIST_ENABLED = True
            agent = self._recall_explore_agent()
            agent.reward_manager._recall_channel_steps = 10
            observation = self._recall_explore_observation()
            act_data = SimpleNamespace(
                action=[GameConfig.RECALL_NOOP_BUTTON, 0, 0, 0, 0, 0],
                d_action=[GameConfig.RECALL_NOOP_BUTTON, 0, 0, 0, 0, 0],
                prob=[[0.0] * Config.LABEL_SUM],
            )

            action = agent.action_process(observation, act_data, True)
            stats = agent.consume_action_mask_stats()

            self.assertEqual(action[0], GameConfig.RECALL_NOOP_BUTTON)
            self.assertEqual(stats["recall_hold_active_cnt"], 1)
            self.assertEqual(stats["recall_hold_model_keep_cnt"], 1)
            self.assertEqual(stats["recall_hold_model_keep_rate"], 1.0)
            self.assertEqual(stats["recall_hold_assist_cnt"], 0)
        finally:
            GameConfig.RECALL_EXPLORATION_ENABLED = old_enabled
            GameConfig.RECALL_HOLD_ASSIST_ENABLED = old_hold_assist

    def _recall_explore_agent(self):
        self._install_base_agent_stub()
        from agent_ppo.agent import Agent

        agent = Agent.__new__(Agent)
        agent.label_size_list = Config.LABEL_SIZE_LIST
        agent.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        agent.reward_manager = GameRewardManager(MAIN_ID)
        agent._action_mask_stats = {}
        agent._reset_recall_exploration_state()
        return agent

    def _recall_explore_observation(self):
        return {
            "frame_state": make_frame(
                main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
                enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
            ),
            "legal_action": torch.ones(sum(Config.LEGAL_ACTION_SIZE_LIST)).numpy(),
        }

    def test_ppo_recall_reward_encourages_start_and_successful_recovery(self):
        self._enable_recall_for_test()
        manager = GameRewardManager(MAIN_ID)
        low_hp_lane = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )
        recovered_backfield = make_frame(
            main=make_hero(MAIN_ID, 1, hp=760, max_hp=1000, x=-15000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )

        manager.result(low_hp_lane)
        manager.set_distance_penalty([GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0], low_hp_lane)
        reward = manager.result(recovered_backfield)
        stats = manager.consume_monitor_stats()

        expected = GameConfig.RECALL_START_REWARD + GameConfig.RECALL_SUCCESS_REWARD
        self.assertAlmostEqual(reward["recall_recover"], expected)
        self.assertEqual(reward["lane_progress"], 0.0)
        self.assertGreater(reward["reward_sum"], 0.0)
        self.assertGreater(reward["recall_recover"], GameConfig.RECALL_INTERRUPT_PENALTY)
        self.assertLess(reward["recall_recover"], GameConfig.REWARD_WEIGHT_DICT["kill"])
        self.assertEqual(stats["recall_need_cnt"], 1)
        self.assertEqual(stats["recall_start_cnt"], 1)
        self.assertEqual(stats["recall_success_cnt"], 1)
        self.assertAlmostEqual(stats["recall_success_reward_sum"], GameConfig.RECALL_SUCCESS_REWARD)
        self.assertEqual(stats["recall_button_rate_when_needed"], 1.0)

    def test_ppo_recall_success_reward_scales_down_when_enemy_tower_is_low(self):
        self._enable_recall_for_test()
        manager = GameRewardManager(MAIN_ID)
        low_hp_lane = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, hp=50, x=15000),
            ],
        )
        recovered_backfield = make_frame(
            main=make_hero(MAIN_ID, 1, hp=760, max_hp=1000, x=-15000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, hp=50, x=15000),
            ],
        )

        manager.result(low_hp_lane)
        manager.set_distance_penalty([GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0], low_hp_lane)
        reward = manager.result(recovered_backfield)
        stats = manager.consume_monitor_stats()

        discounted_success = (
            GameConfig.RECALL_SUCCESS_REWARD
            * GameConfig.RECALL_SUCCESS_MIN_TOWER_FACTOR
        )
        expected = GameConfig.RECALL_START_REWARD + discounted_success
        self.assertAlmostEqual(reward["recall_recover"], expected)
        self.assertAlmostEqual(stats["recall_success_reward_sum"], discounted_success)
        self.assertLess(reward["recall_recover"], GameConfig.RECALL_START_REWARD + GameConfig.RECALL_SUCCESS_REWARD)

    def test_ppo_recall_reward_penalizes_ignoring_or_interrupting_channel(self):
        self._enable_recall_for_test()
        manager = GameRewardManager(MAIN_ID)
        low_hp_lane = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )

        manager.result(low_hp_lane)
        manager.set_distance_penalty([GameConfig.RECALL_NOOP_BUTTON, 0, 0, 0, 0, 0], low_hp_lane)
        miss_reward = manager.result(low_hp_lane)

        manager.set_distance_penalty([GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0], low_hp_lane)
        start_reward = manager.result(low_hp_lane)

        manager.set_distance_penalty([3, 0, 0, 0, 0, 1], low_hp_lane)
        interrupt_reward = manager.result(low_hp_lane)
        stats = manager.consume_monitor_stats()

        self.assertAlmostEqual(miss_reward["recall_recover"], -GameConfig.RECALL_MISS_PENALTY)
        self.assertAlmostEqual(start_reward["recall_recover"], GameConfig.RECALL_START_REWARD)
        self.assertAlmostEqual(
            interrupt_reward["recall_recover"],
            -GameConfig.RECALL_INTERRUPT_PENALTY,
        )
        self.assertLess(interrupt_reward["recall_recover"], miss_reward["recall_recover"])
        self.assertEqual(stats["recall_need_cnt"], 3)
        self.assertEqual(stats["recall_miss_cnt"], 1)
        self.assertEqual(stats["recall_interrupt_cnt"], 1)
        self.assertAlmostEqual(
            stats["recall_interrupt_penalty_sum"],
            -GameConfig.RECALL_INTERRUPT_PENALTY,
        )

    def test_ppo_recall_need_penalizes_walking_back_instead_of_recall(self):
        self._enable_recall_for_test()
        manager = GameRewardManager(MAIN_ID)
        low_hp_safe = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )

        manager.result(low_hp_safe)
        manager.set_distance_penalty([2, 0, 0, 0, 0, 0], low_hp_safe)
        reward = manager.result(low_hp_safe)
        stats = manager.consume_monitor_stats()

        self.assertAlmostEqual(reward["recall_recover"], -GameConfig.RECALL_MISS_PENALTY)
        self.assertEqual(stats["recall_need_cnt"], 1)
        self.assertEqual(stats["recall_miss_cnt"], 1)

    def test_ppo_safe_low_hp_heal_without_recall_does_not_get_retreat_reward(self):
        manager = GameRewardManager(MAIN_ID)
        low_hp_backfield = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-20000),
            enemy=make_hero(ENEMY_ID, 2, hp=900, max_hp=1000, attack_range=5000, x=1000),
        )
        recovered_backfield = make_frame(
            main=make_hero(MAIN_ID, 1, hp=760, max_hp=1000, x=-20000),
            enemy=make_hero(ENEMY_ID, 2, hp=900, max_hp=1000, attack_range=5000, x=1000),
        )

        manager.result(low_hp_backfield)
        reward = manager.result(recovered_backfield)

        self.assertEqual(reward["lane_progress"], 0.0)
        self.assertAlmostEqual(reward["retreat_recover"], 0.0)
        self.assertAlmostEqual(reward["recall_recover"], 0.0)

    def test_ppo_danger_low_hp_retreat_move_beats_staying_in_threat(self):
        retreat_manager = GameRewardManager(MAIN_ID)
        stay_manager = GameRewardManager(MAIN_ID)
        threatened_low_hp = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=900, max_hp=1000, attack_range=5000, x=1000),
        )
        safer_low_hp = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=900, max_hp=1000, attack_range=5000, x=1000),
        )

        retreat_manager.result(threatened_low_hp)
        retreat_reward = retreat_manager.result(safer_low_hp)

        stay_manager.result(threatened_low_hp)
        stay_reward = stay_manager.result(threatened_low_hp)

        self.assertGreater(retreat_reward["retreat_recover"], 0.0)
        self.assertGreater(retreat_reward["reward_sum"], stay_reward["reward_sum"])

    def test_ppo_recall_success_does_not_double_count_retreat_heal_reward(self):
        self._enable_recall_for_test()
        manager = GameRewardManager(MAIN_ID)
        threatened_low_hp = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=900, max_hp=1000, attack_range=5000, x=1000),
        )
        safe_low_hp = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-15000),
            enemy=make_hero(ENEMY_ID, 2, hp=900, max_hp=1000, attack_range=5000, x=1000),
        )
        recovered_backfield = make_frame(
            main=make_hero(MAIN_ID, 1, hp=760, max_hp=1000, x=-15000),
            enemy=make_hero(ENEMY_ID, 2, hp=900, max_hp=1000, attack_range=5000, x=1000),
        )

        manager.result(threatened_low_hp)
        manager.set_distance_penalty([GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0], safe_low_hp)
        manager.result(safe_low_hp)
        reward = manager.result(recovered_backfield)

        self.assertEqual(reward["lane_progress"], 0.0)
        self.assertAlmostEqual(reward["retreat_recover"], 0.0)
        self.assertGreater(reward["recall_recover"], 0.0)

    def test_ppo_recall_success_beats_natural_heal_in_total_reward(self):
        self._enable_recall_for_test()
        recall_manager = GameRewardManager(MAIN_ID)
        natural_manager = GameRewardManager(MAIN_ID)
        low_hp_safe = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )
        recovered_backfield = make_frame(
            main=make_hero(MAIN_ID, 1, hp=760, max_hp=1000, x=-15000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )

        recall_manager.result(low_hp_safe)
        recall_manager.set_distance_penalty(
            [GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0],
            low_hp_safe,
        )
        recall_reward = recall_manager.result(recovered_backfield)

        natural_manager.result(low_hp_safe)
        natural_manager.set_distance_penalty([2, 0, 0, 0, 0, 0], low_hp_safe)
        natural_reward = natural_manager.result(recovered_backfield)

        self.assertGreater(recall_reward["reward_sum"], natural_reward["reward_sum"])
        self.assertGreater(recall_reward["reward_sum"], 0.0)
        self.assertLess(natural_reward["reward_sum"], 0.0)

    def test_ppo_recall_need_allows_low_hp_under_own_tower(self):
        self._enable_recall_for_test()
        manager = GameRewardManager(MAIN_ID)
        low_hp_behind_tower = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-15000),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, attack_range=5000, x=-12000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
            ],
        )

        manager.result(low_hp_behind_tower)
        manager.set_distance_penalty(
            [GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0],
            low_hp_behind_tower,
        )
        reward = manager.result(low_hp_behind_tower)
        stats = manager.consume_monitor_stats()

        self.assertAlmostEqual(reward["recall_recover"], GameConfig.RECALL_START_REWARD)
        self.assertEqual(stats["recall_need_cnt"], 1)
        self.assertEqual(stats["recall_start_cnt"], 1)

    def test_ppo_recall_reward_does_not_compete_with_safe_tower_push(self):
        self._enable_recall_for_test()
        manager = GameRewardManager(MAIN_ID)
        low_hp_with_wave = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, attack_range=2000, x=14000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 1, hp=1000, x=14800),
            ],
        )

        manager.result(low_hp_with_wave)
        manager.set_distance_penalty([GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0], low_hp_with_wave)
        reward = manager.result(low_hp_with_wave)
        stats = manager.consume_monitor_stats()

        self.assertAlmostEqual(reward["recall_recover"], -GameConfig.RECALL_UNNEEDED_PENALTY)
        self.assertEqual(stats["recall_need_cnt"], 0)
        self.assertEqual(stats["recall_unneeded_cnt"], 1)

    def test_ppo_major_reward_scale_hierarchy_is_consistent(self):
        base = make_frame()

        def transition_reward(next_frame, *, action=None):
            manager = GameRewardManager(MAIN_ID)
            manager.result(base)
            if action is not None:
                manager.set_distance_penalty(action, base)
            return manager.result(next_frame)["reward_sum"]

        tower_hit = make_frame(npcs=[
            make_tower(1, x=-15000),
            make_tower(2, hp=900, x=15000),
        ])
        hero_damage = make_frame()
        hero_damage["hero_states"][0]["total_hurt_to_hero"] = 1000
        kill = make_frame()
        kill["hero_states"][0]["kill_cnt"] = 1
        death = make_frame()
        death["hero_states"][0]["dead_cnt"] = 1
        money = make_frame()
        money["hero_states"][0]["money_cnt"] = 40
        exp = make_frame()
        exp["hero_states"][0]["exp"] = 100
        last_hit = make_frame()
        last_hit["frame_action"] = {
            "dead_action": [
                {"death": {"sub_type": 11, "camp": 2}, "killer": {"runtime_id": MAIN_ID}},
            ],
        }
        monster = make_frame()
        monster["frame_action"] = {
            "dead_action": [
                {"death": {"sub_type": 12, "camp": 0}, "killer": {"runtime_id": MAIN_ID}},
            ],
        }

        kill_reward = transition_reward(kill)
        hero_damage_reward = transition_reward(hero_damage)
        last_hit_reward = transition_reward(last_hit)
        monster_reward = transition_reward(monster)
        tower_reward = transition_reward(tower_hit)
        exp_reward = transition_reward(exp)
        money_reward = transition_reward(money)
        death_reward = transition_reward(death)

        self.assertGreater(kill_reward, hero_damage_reward)
        self.assertGreater(hero_damage_reward, tower_reward)
        self.assertGreater(tower_reward, last_hit_reward)
        self.assertGreater(last_hit_reward, monster_reward)
        self.assertGreater(monster_reward, exp_reward)
        self.assertGreater(exp_reward, money_reward)
        self.assertLess(death_reward, -kill_reward)

    def test_ppo_composite_reward_prefers_objective_safe_fighting(self):
        def run_episode(frames, *, actions=None, terminal_win=None):
            manager = GameRewardManager(MAIN_ID)
            total = 0.0
            actions = actions or [None] * len(frames)
            for idx, frame in enumerate(frames):
                if idx > 0 and actions[idx] is not None:
                    manager.set_distance_penalty(actions[idx], frames[idx - 1])
                total += manager.result(frame)["reward_sum"]
            reward_dict = {"reward_sum": total}
            total += manager.apply_terminal_outcome(
                reward_dict,
                frames[-1],
                win=terminal_win,
            )
            return total

        base = make_frame()
        safe_trade = make_frame()
        safe_trade["hero_states"][0]["total_hurt_to_hero"] = 1000
        even_trade = make_frame()
        even_trade["hero_states"][0]["total_hurt_to_hero"] = 1000
        even_trade["hero_states"][1]["total_hurt_to_hero"] = 700
        bad_trade = make_frame()
        bad_trade["hero_states"][0]["total_hurt_to_hero"] = 1000
        bad_trade["hero_states"][1]["total_hurt_to_hero"] = 1500
        taking_damage = make_frame()
        taking_damage["hero_states"][1]["total_hurt_to_hero"] = 900

        farm_monster = make_frame()
        farm_monster["frame_action"] = {
            "dead_action": [
                {"death": {"sub_type": 12, "camp": 0}, "killer": {"runtime_id": MAIN_ID}},
            ],
        }

        bad_recall = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, x=10000),
        )

        own_tower_hit_while_farming = make_frame(npcs=[
            make_tower(1, hp=800, x=-15000),
            make_tower(2, hp=1000, x=15000),
        ])
        own_tower_hit_while_farming["frame_action"] = farm_monster["frame_action"]

        pushed_enemy_tower = make_frame(npcs=[
            make_tower(1, hp=1000, x=-15000),
            make_tower(2, hp=800, x=15000),
            make_minion(401, 1, hp=1000, x=14500),
        ])

        greedy_death_after_fight = make_frame()
        greedy_death_after_fight["hero_states"][0]["total_hurt_to_hero"] = 5000
        greedy_death_after_fight["hero_states"][0]["kill_cnt"] = 1
        greedy_death_after_fight["hero_states"][0]["dead_cnt"] = 1

        lost_tower_after_fight = make_frame(npcs=[
            make_tower(1, hp=0, x=-15000),
            make_tower(2, hp=1000, x=15000),
        ])
        lost_tower_after_fight["hero_states"][0]["total_hurt_to_hero"] = 5000
        lost_tower_after_fight["hero_states"][0]["kill_cnt"] = 1
        lost_tower_after_fight["hero_states"][0]["dead_cnt"] = 1

        safe_trade_score = run_episode([base, safe_trade])
        even_trade_score = run_episode([base, even_trade])
        bad_trade_score = run_episode([base, bad_trade])
        taking_damage_score = run_episode([base, taking_damage])
        monster_score = run_episode([base, farm_monster])
        bad_recall_score = run_episode(
            [bad_recall, bad_recall],
            actions=[None, [GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0]],
        )
        ignore_tower_score = run_episode([base, own_tower_hit_while_farming])
        push_tower_score = run_episode([base, pushed_enemy_tower])
        greedy_death_score = run_episode([base, greedy_death_after_fight])
        lost_tower_score = run_episode([base, lost_tower_after_fight], terminal_win=0)

        self.assertGreater(safe_trade_score, even_trade_score)
        self.assertGreater(even_trade_score, bad_trade_score)
        self.assertGreater(bad_trade_score, taking_damage_score)
        self.assertLess(taking_damage_score, 0.0)
        self.assertGreater(safe_trade_score, monster_score)
        self.assertLess(bad_trade_score, monster_score)
        self.assertGreater(monster_score, bad_recall_score)
        self.assertGreater(push_tower_score, safe_trade_score)
        self.assertLess(greedy_death_score, safe_trade_score)
        self.assertLess(greedy_death_score, 0.0)
        self.assertLess(ignore_tower_score, monster_score)
        self.assertLess(lost_tower_score, greedy_death_score)

    def test_ppo_conservative_laning_rewards_safe_trade_not_damage_taken(self):
        base = make_frame()

        def transition_reward(next_frame):
            manager = GameRewardManager(MAIN_ID)
            manager.result(base)
            return manager.result(next_frame)["reward_sum"]

        safe_trade = make_frame()
        safe_trade["hero_states"][0]["total_hurt_to_hero"] = 1000
        safe_trade["hero_states"][1]["total_hurt_to_hero"] = 400

        even_trade = make_frame()
        even_trade["hero_states"][0]["total_hurt_to_hero"] = 600
        even_trade["hero_states"][1]["total_hurt_to_hero"] = 600

        damage_taken = make_frame()
        damage_taken["hero_states"][1]["total_hurt_to_hero"] = 600

        self.assertGreater(transition_reward(safe_trade), 0.0)
        self.assertLess(transition_reward(even_trade), 0.0)
        self.assertLess(transition_reward(damage_taken), transition_reward(even_trade))

    def test_ppo_low_hp_retreat_need_penalizes_aggressive_targets(self):
        low_hp_threat = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, attack_range=5000, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, attack_range=5000, x=3000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
            ],
        )

        def transition_reward(action):
            manager = GameRewardManager(MAIN_ID)
            manager.result(low_hp_threat)
            manager.set_distance_penalty(action, low_hp_threat)
            return manager.result(low_hp_threat)

        noop_reward = transition_reward([GameConfig.RECALL_NOOP_BUTTON, 0, 0, 0, 0, 0])
        attack_hero_reward = transition_reward([3, 0, 0, 0, 0, 1])
        attack_tower_reward = transition_reward([3, 0, 0, 0, 0, 7])

        self.assertLess(attack_hero_reward["distance_penalty"], 0.0)
        self.assertLess(attack_tower_reward["distance_penalty"], 0.0)
        self.assertLess(attack_hero_reward["reward_sum"], noop_reward["reward_sum"])
        self.assertLess(attack_tower_reward["reward_sum"], noop_reward["reward_sum"])

    def test_ppo_low_hp_retreat_prefers_under_tower_not_behind_tower(self):
        manager = GameRewardManager(MAIN_ID)
        under_tower = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-15000),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, x=12000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
            ],
        )
        behind_tower = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-22000),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, x=12000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
            ],
        )

        under_state = manager.calculate_retreat_recover_state(
            under_tower,
            1,
            under_tower["hero_states"][0],
            under_tower["npc_states"][0],
            under_tower["hero_states"][1],
            under_tower["npc_states"][1],
        )
        behind_state = manager.calculate_retreat_recover_state(
            behind_tower,
            1,
            behind_tower["hero_states"][0],
            behind_tower["npc_states"][0],
            behind_tower["hero_states"][1],
            behind_tower["npc_states"][1],
        )

        self.assertTrue(under_state["in_retreat_zone"])
        self.assertFalse(behind_state["in_retreat_zone"])

        under_reward_manager = GameRewardManager(MAIN_ID)
        behind_reward_manager = GameRewardManager(MAIN_ID)
        under_reward = under_reward_manager.result(under_tower)
        behind_reward = behind_reward_manager.result(behind_tower)

        self.assertEqual(under_reward["danger_penalty"], 0.0)
        self.assertGreater(behind_reward["danger_penalty"], 0.0)
        self.assertLess(behind_reward["reward_sum"], under_reward["reward_sum"])

    def test_ppo_long_horizon_reward_orders_common_strategies(self):
        def frame(
            *,
            frame_no,
            main_hp=1000,
            enemy_hp=1000,
            main_hurt=0,
            enemy_hurt=0,
            main_kill=0,
            main_dead=0,
            main_money=0,
            main_exp=0,
            own_tower_hp=1000,
            enemy_tower_hp=1000,
            main_x=-10000,
            enemy_x=10000,
            own_minion_x=None,
            enemy_minion_x=None,
            enemy_minion_hp=1000,
            dead_actions=None,
        ):
            f = make_frame(
                main=make_hero(MAIN_ID, 1, hp=main_hp, max_hp=1000, attack_range=5000, x=main_x),
                enemy=make_hero(ENEMY_ID, 2, hp=enemy_hp, max_hp=1000, attack_range=5000, x=enemy_x),
                npcs=[
                    make_tower(1, hp=own_tower_hp, x=-15000),
                    make_tower(2, hp=enemy_tower_hp, x=15000),
                ],
            )
            f["frame_no"] = frame_no
            f["hero_states"][0]["total_hurt_to_hero"] = main_hurt
            f["hero_states"][1]["total_hurt_to_hero"] = enemy_hurt
            f["hero_states"][0]["kill_cnt"] = main_kill
            f["hero_states"][0]["dead_cnt"] = main_dead
            f["hero_states"][0]["money_cnt"] = main_money
            f["hero_states"][0]["exp"] = main_exp
            if own_minion_x is not None:
                f["npc_states"].append(make_minion(401, 1, hp=1000, x=own_minion_x))
            if enemy_minion_x is not None and enemy_minion_hp > 0:
                f["npc_states"].append(make_minion(501, 2, hp=enemy_minion_hp, x=enemy_minion_x))
            f["frame_action"] = {"dead_action": dead_actions or []}
            return f

        def run_episode(frames, actions=None, terminal_win=None):
            manager = GameRewardManager(MAIN_ID)
            actions = actions or [None] * len(frames)
            total = 0.0
            for index, frame_data in enumerate(frames):
                if index > 0 and actions[index] is not None:
                    manager.set_distance_penalty(actions[index], frames[index - 1])
                total += manager.result(frame_data)["reward_sum"]
            reward_dict = {"reward_sum": total}
            total += manager.apply_terminal_outcome(
                reward_dict,
                frames[-1],
                win=terminal_win,
            )
            return total

        attack_hero = [3, 0, 0, 0, 0, 1]
        attack_minion = [3, 0, 0, 0, 0, 3]
        attack_monster = [3, 0, 0, 0, 0, 8]
        attack_tower = [3, 0, 0, 0, 0, 7]
        noop = [GameConfig.RECALL_NOOP_BUTTON, 0, 0, 0, 0, 0]
        recall = [GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0]

        safe_trade = [
            frame(frame_no=i, main_hurt=i * 18, enemy_hurt=i * 4, main_x=0, enemy_x=3500)
            for i in range(180)
        ]
        even_trade = [
            frame(frame_no=i, main_hurt=i * 14, enemy_hurt=i * 12, main_x=0, enemy_x=3500)
            for i in range(180)
        ]
        bad_trade = [
            frame(frame_no=i, main_hurt=i * 8, enemy_hurt=i * 16, main_x=0, enemy_x=3500)
            for i in range(180)
        ]
        pure_damage_taken = [
            frame(frame_no=i, main_hurt=0, enemy_hurt=i * 18, main_x=0, enemy_x=3000)
            for i in range(120)
        ]
        jungle = [
            frame(
                frame_no=i,
                main_money=(i // 60) * 30,
                main_exp=(i // 60) * 20,
                dead_actions=[
                    {
                        "death": {"runtime_id": 800 + i, "camp": 0, "sub_type": 12},
                        "killer": {"runtime_id": MAIN_ID},
                    }
                ]
                if i > 0 and i % 60 == 0
                else None,
            )
            for i in range(240)
        ]
        safe_push = [
            frame(
                frame_no=i,
                enemy_tower_hp=max(0, 1000 - i * 3),
                main_x=14500,
                enemy_x=9000,
                own_minion_x=14500,
            )
            for i in range(180)
        ]
        win_push = [
            frame(
                frame_no=i,
                enemy_tower_hp=max(0, 1000 - i * 5),
                main_hurt=i * 8,
                main_kill=1 if i > 100 else 0,
                main_x=14500,
                enemy_x=9000,
                own_minion_x=14500,
            )
            for i in range(240)
        ]
        ignore_tower = [
            frame(
                frame_no=i,
                own_tower_hp=max(0, 1000 - i * 3),
                main_money=(i // 60) * 30,
                main_exp=(i // 60) * 20,
                dead_actions=[
                    {
                        "death": {"runtime_id": 900 + i, "camp": 0, "sub_type": 12},
                        "killer": {"runtime_id": MAIN_ID},
                    }
                ]
                if i > 0 and i % 60 == 0
                else None,
            )
            for i in range(240)
        ]
        kill_no_death = [
            frame(
                frame_no=i,
                main_hurt=i * 30,
                enemy_hurt=i * 5,
                main_kill=1 if i >= 80 else 0,
                main_x=0,
                enemy_x=3500,
            )
            for i in range(120)
        ]
        kill_then_death = [
            frame(
                frame_no=i,
                main_hurt=i * 30,
                enemy_hurt=i * 20,
                main_kill=1 if i >= 80 else 0,
                main_dead=1 if i >= 100 else 0,
                main_x=0,
                enemy_x=3500,
            )
            for i in range(120)
        ]
        effective_defense = [
            frame(
                frame_no=i,
                own_tower_hp=max(880, 1000 - i),
                main_hurt=i * 6,
                enemy_hurt=i * 2,
                main_x=-12000,
                enemy_x=-9000,
                enemy_minion_x=-11000,
                enemy_minion_hp=max(0, 1000 - i * 9),
                dead_actions=[
                    {
                        "death": {"runtime_id": 501, "camp": 2, "sub_type": 11},
                        "killer": {"runtime_id": MAIN_ID},
                    }
                ]
                if i == 112
                else None,
            )
            for i in range(180)
        ]
        bad_defense = [
            frame(
                frame_no=i,
                own_tower_hp=max(700, 1000 - i * 2),
                main_hurt=i * 4,
                enemy_hurt=i * 12,
                main_x=-12000,
                enemy_x=-9000,
                enemy_minion_x=-11000,
            )
            for i in range(180)
        ]
        low_hp_recall = [
            frame(
                frame_no=i,
                main_hp=300 if i < 80 else 800,
                enemy_hp=0,
                main_x=-10000 if i < 80 else -22000,
                enemy_x=20000,
            )
            for i in range(100)
        ]
        low_hp_idle = [
            frame(frame_no=i, main_hp=300, enemy_hp=0, main_x=-10000, enemy_x=20000)
            for i in range(100)
        ]
        full_hp_recall = [
            frame(frame_no=i, main_hp=1000, main_x=-10000, enemy_x=12000)
            for i in range(120)
        ]
        lose_tower = [
            frame(
                frame_no=i,
                own_tower_hp=max(0, 1000 - i * 5),
                main_hurt=i * 20,
                enemy_hurt=i * 8,
                main_kill=1 if i > 90 else 0,
                main_dead=1 if i > 150 else 0,
                main_x=0,
                enemy_x=3500,
            )
            for i in range(240)
        ]

        score = {
            "win_push": run_episode(win_push, [attack_tower] * 240, terminal_win=1),
            "safe_push": run_episode(safe_push, [attack_tower] * 180),
            "safe_trade": run_episode(safe_trade, [attack_hero] * 180),
            "even_trade": run_episode(even_trade, [attack_hero] * 180),
            "bad_trade": run_episode(bad_trade, [attack_hero] * 180),
            "pure_damage_taken": run_episode(pure_damage_taken, [noop] * 120),
            "jungle": run_episode(jungle, [attack_monster] * 240),
            "ignore_tower": run_episode(ignore_tower, [attack_monster] * 240),
            "kill_no_death": run_episode(kill_no_death, [attack_hero] * 120),
            "kill_then_death": run_episode(kill_then_death, [attack_hero] * 120),
            "effective_defense": run_episode(effective_defense, [attack_minion] * 180),
            "bad_defense": run_episode(bad_defense, [attack_hero] * 180),
            "low_hp_idle": run_episode(low_hp_idle, [noop] * 100),
            "full_hp_recall": run_episode(full_hp_recall, [recall] * 120),
            "lose_tower": run_episode(lose_tower, [attack_hero] * 240, terminal_win=0),
        }

        self.assertGreater(score["win_push"], score["safe_push"])
        self.assertGreater(score["safe_push"], score["safe_trade"])
        self.assertGreater(score["safe_trade"], score["even_trade"])
        self.assertGreater(score["even_trade"], score["bad_trade"])
        self.assertGreater(score["bad_trade"], score["pure_damage_taken"])
        self.assertGreater(score["safe_trade"], score["jungle"])
        self.assertGreater(score["jungle"], score["full_hp_recall"])
        self.assertGreater(score["kill_no_death"], score["kill_then_death"])
        self.assertGreater(score["effective_defense"], score["bad_defense"])
        self.assertGreater(score["bad_defense"], score["ignore_tower"])
        self.assertGreater(score["ignore_tower"], score["lose_tower"])

    def test_ppo_lane_guidance_is_capped_below_kill_but_above_small_shaping(self):
        manager = GameRewardManager(MAIN_ID)
        fountain = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, max_hp=1000, x=-22500),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, x=20000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
            ],
        )
        toward_lane = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, max_hp=1000, x=-18000),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, x=20000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
            ],
        )
        kill = make_frame()
        kill["hero_states"][0]["kill_cnt"] = 1
        kill_manager = GameRewardManager(MAIN_ID)

        manager.result(fountain)
        lane_reward = manager.result(toward_lane)["reward_sum"]
        kill_manager.result(make_frame())
        kill_reward = kill_manager.result(kill)["reward_sum"]

        self.assertGreater(lane_reward, 0.0)
        self.assertLess(lane_reward, kill_reward)
        self.assertLessEqual(
            GameConfig.LANE_PROGRESS_MAX_PER_EPISODE
            * GameConfig.REWARD_WEIGHT_DICT["lane_progress"],
            kill_reward,
        )

    def test_ppo_action_shaping_prefers_correct_actions_without_dominating_objectives(self):
        low_hp_minion_window = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, max_hp=1000, attack_range=5000, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, x=12000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(601, 2, hp=100, x=1000),
            ],
        )

        def action_reward(action):
            manager = GameRewardManager(MAIN_ID)
            manager.result(low_hp_minion_window)
            manager.set_distance_penalty(action, low_hp_minion_window)
            return manager.result(low_hp_minion_window)["reward_sum"]

        correct_last_hit = action_reward([3, 0, 0, 0, 0, 3])
        noop = action_reward([GameConfig.RECALL_NOOP_BUTTON, 0, 0, 0, 0, 0])
        wrong_target = action_reward([3, 0, 0, 0, 0, 1])

        self.assertGreater(correct_last_hit, noop)
        self.assertGreater(noop, wrong_target)
        self.assertLess(correct_last_hit, 1.0)

    def test_ppo_terminal_reward_dominates_dense_objectives(self):
        win_reward = {"reward_sum": 0.0}
        loss_reward = {"reward_sum": 0.0}
        win_manager = GameRewardManager(MAIN_ID)
        loss_manager = GameRewardManager(MAIN_ID)
        win_bonus = win_manager.apply_terminal_outcome(win_reward, make_frame(), win=1)
        loss_bonus = loss_manager.apply_terminal_outcome(loss_reward, make_frame(), win=0)

        self.assertGreater(win_bonus, GameConfig.REWARD_WEIGHT_DICT["kill"])
        self.assertLess(loss_bonus, -GameConfig.REWARD_WEIGHT_DICT["kill"])
        self.assertEqual(win_reward["reward_sum"], win_bonus)
        self.assertEqual(loss_reward["reward_sum"], loss_bonus)


if __name__ == "__main__":
    unittest.main()
