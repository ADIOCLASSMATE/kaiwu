import unittest

try:
    import torch

    from agent_ppo.conf.conf import Config
    from agent_ppo.model.model import Model
except ModuleNotFoundError:
    torch = None
    Config = None
    Model = None


@unittest.skipIf(torch is None, "torch is not installed")
class AgentPpoDiyModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()

    def _feature_batch(self, batch_size):
        feature = torch.zeros(batch_size, Config.SERI_VEC_SPLIT_SHAPE[0][0])
        feature[:, 0] = 1.0
        feature[:, 3] = 1.0
        feature[:, 9] = 1.0
        return feature

    def test_forward_shapes_for_training_and_inference(self):
        self.model.set_eval_mode()
        feature = self._feature_batch(1)
        hidden = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        cell = torch.zeros(1, Config.LSTM_UNIT_SIZE)

        logits, value, next_cell, next_hidden = self.model(
            [feature, hidden, cell],
            inference=True,
        )

        self.assertEqual(tuple(logits.shape), (1, sum(Config.LABEL_SIZE_LIST)))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(next_cell.shape), (1, 1, Config.LSTM_UNIT_SIZE))
        self.assertEqual(tuple(next_hidden.shape), (1, 1, Config.LSTM_UNIT_SIZE))
        self.assertEqual(
            tuple(self.model.target_logits_by_button.shape),
            (1, Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1]),
        )

    def test_target_pointer_is_conditioned_by_button(self):
        self.model.set_eval_mode()
        feature = self._feature_batch(1)
        hidden = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        cell = torch.zeros(1, Config.LSTM_UNIT_SIZE)

        self.model([feature, hidden, cell], inference=True)
        target_logits = self.model.target_logits_by_button[0]

        self.assertFalse(torch.allclose(target_logits[3], target_logits[8]))

    def test_ppo_loss_backward_with_original_feature_layout(self):
        self.model.set_train_mode()
        time_steps = Config.LSTM_TIME_STEPS
        feature = self._feature_batch(time_steps)
        legal_action = torch.ones(time_steps, sum(Config.LABEL_SIZE_LIST))
        seri_vec = torch.cat([feature, legal_action], dim=1)

        data_list = [
            seri_vec,
            torch.linspace(0.0, 1.0, time_steps).unsqueeze(1),
            torch.linspace(-1.0, 1.0, time_steps).unsqueeze(1),
        ]
        data_list.extend(torch.zeros(time_steps, 1) for _ in Config.LABEL_SIZE_LIST)
        data_list.extend(
            torch.full((time_steps, label_size), 1.0 / label_size)
            for label_size in Config.LABEL_SIZE_LIST
        )
        data_list.extend(torch.ones(time_steps, 1) for _ in Config.LABEL_SIZE_LIST)
        data_list.extend(
            [
                torch.ones(time_steps, 1),
                torch.zeros(Config.LSTM_UNIT_SIZE),
                torch.zeros(Config.LSTM_UNIT_SIZE),
            ]
        )

        result = self.model([feature, data_list[-1], data_list[-2]], inference=False)
        loss, _ = self.model.compute_loss(data_list, result)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        grad_sum = sum(
            p.grad.abs().sum().item()
            for p in self.model.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)


if __name__ == "__main__":
    unittest.main()
