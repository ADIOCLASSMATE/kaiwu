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
        ]:
            self.assertIn(marker, monitor_source)


if __name__ == "__main__":
    unittest.main()
