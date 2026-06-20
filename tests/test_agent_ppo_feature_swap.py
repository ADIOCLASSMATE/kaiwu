import unittest
from pathlib import Path

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

    def test_agent_samples_target_from_selected_button_logits(self):
        try:
            from agent_ppo.agent import Agent
        except ModuleNotFoundError as exc:
            self.skipTest("agent framework dependency is unavailable: {}".format(exc))

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
        ]:
            self.assertIn(marker, monitor_source)

    def test_ppo_recall_reward_encourages_start_and_successful_recovery(self):
        manager = GameRewardManager(MAIN_ID)
        low_hp_lane = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )
        recovered_backfield = make_frame(
            main=make_hero(MAIN_ID, 1, hp=760, max_hp=1000, x=-20000),
            enemy=make_hero(ENEMY_ID, 2, hp=0, max_hp=1000, x=10000),
        )

        manager.result(low_hp_lane)
        manager.set_distance_penalty([GameConfig.RECALL_BUTTON, 0, 0, 0, 0, 0], low_hp_lane)
        reward = manager.result(recovered_backfield)
        stats = manager.consume_monitor_stats()

        expected = GameConfig.RECALL_START_REWARD + GameConfig.RECALL_SUCCESS_REWARD
        self.assertAlmostEqual(reward["recall_recover"], expected)
        self.assertGreater(reward["recall_recover"], GameConfig.RECALL_INTERRUPT_PENALTY)
        self.assertLess(reward["recall_recover"], GameConfig.REWARD_WEIGHT_DICT["kill"])
        self.assertEqual(stats["recall_need_cnt"], 1)
        self.assertEqual(stats["recall_start_cnt"], 1)
        self.assertEqual(stats["recall_success_cnt"], 1)
        self.assertEqual(stats["recall_button_rate_when_needed"], 1.0)

    def test_ppo_recall_reward_penalizes_ignoring_or_interrupting_channel(self):
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

    def test_ppo_recall_reward_does_not_compete_with_safe_tower_push(self):
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


if __name__ == "__main__":
    unittest.main()
