import unittest

try:
    import torch

    from agent_diy.conf.conf import FeatureConfig as DiyFeatureConfig
    from agent_diy.feature.feature_process import FeatureProcess as DiyFeatureProcess
    from agent_ppo.conf.conf import Config
    from agent_ppo.feature.feature_process import FeatureProcess
    from agent_ppo.model.model import Model
except ModuleNotFoundError:
    torch = None
    DiyFeatureConfig = None
    DiyFeatureProcess = None
    Config = None
    FeatureProcess = None
    Model = None


@unittest.skipIf(torch is None, "torch is not installed")
class AgentPpoFeatureSwapTests(unittest.TestCase):
    def test_ppo_uses_diy_feature_process(self):
        self.assertIs(FeatureProcess, DiyFeatureProcess)
        self.assertEqual(Config.FEATURE_DIM, DiyFeatureConfig.FEATURE_DIM)
        self.assertEqual(Config.SERI_VEC_SPLIT_SHAPE, [(DiyFeatureConfig.FEATURE_DIM,), (85,)])

    def test_old_ppo_model_accepts_diy_feature_dim(self):
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


if __name__ == "__main__":
    unittest.main()
