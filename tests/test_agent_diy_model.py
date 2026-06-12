import unittest

from agent_diy.conf.conf import Config, FeatureConfig as FC


try:
    import torch

    from agent_diy.model.model import Model
except ModuleNotFoundError:
    torch = None
    Model = None


@unittest.skipIf(torch is None, "torch is not installed")
class ActorAdapterTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.model.set_eval_mode()

    def _feature_batch(self, hero_ids):
        feature = torch.zeros(len(hero_ids), FC.FEATURE_DIM)
        main_range = FC.TOKEN_SLICES["main_hero"][0]
        hero_id_slice = FC.HERO_FIELD_SLICES["hero_id"]
        feature[:, main_range.start] = 1.0
        for row, hero_id in enumerate(hero_ids):
            hero_index = FC.HERO_CONFIG_IDS.index(hero_id)
            feature[
                row,
                main_range.start + hero_id_slice.start + hero_index,
            ] = 1.0
        return feature

    def test_zero_initialized_adapters_preserve_shared_logits(self):
        hidden = torch.randn(3, Config.LSTM_UNIT_SIZE)
        feature = self._feature_batch(FC.HERO_CONFIG_IDS)
        shared = self.model._shared_actor_logits(hidden)
        adapted = self.model._actor_logits(hidden, feature)
        for shared_head, adapted_head in zip(shared, adapted):
            torch.testing.assert_close(shared_head, adapted_head)

    def test_each_adapter_only_changes_its_hero(self):
        hidden = torch.randn(3, Config.LSTM_UNIT_SIZE)
        feature = self._feature_batch(FC.HERO_CONFIG_IDS)
        baseline = torch.cat(self.model._actor_logits(hidden, feature), dim=1)

        with torch.no_grad():
            self.model.actor_adapters["112"].up.bias.fill_(1.0)
        changed = torch.cat(self.model._actor_logits(hidden, feature), dim=1)

        torch.testing.assert_close(changed[0], baseline[0] + 1.0)
        torch.testing.assert_close(changed[1:], baseline[1:])

    def test_unknown_hero_does_not_activate_an_adapter(self):
        hidden = torch.randn(1, Config.LSTM_UNIT_SIZE)
        feature = torch.zeros(1, FC.FEATURE_DIM)
        main_range = FC.TOKEN_SLICES["main_hero"][0]
        hero_id_slice = FC.HERO_FIELD_SLICES["hero_id"]
        feature[0, main_range.start] = 1.0
        feature[
            0,
            main_range.start + hero_id_slice.stop - 1,
        ] = 1.0

        with torch.no_grad():
            for adapter in self.model.actor_adapters.values():
                adapter.up.bias.fill_(1.0)

        shared = self.model._shared_actor_logits(hidden)
        adapted = self.model._actor_logits(hidden, feature)
        for shared_head, adapted_head in zip(shared, adapted):
            torch.testing.assert_close(shared_head, adapted_head)

    def test_gradient_routes_to_matching_adapter(self):
        hidden = torch.randn(1, Config.LSTM_UNIT_SIZE, requires_grad=True)
        feature = self._feature_batch([199])
        logits = torch.cat(self.model._actor_logits(hidden, feature), dim=1)
        logits.sum().backward()

        for hero_id in FC.HERO_CONFIG_IDS:
            grad = self.model.actor_adapters[str(hero_id)].up.bias.grad
            self.assertIsNotNone(grad)
            if hero_id == 199:
                self.assertGreater(grad.abs().sum().item(), 0.0)
            else:
                self.assertEqual(grad.abs().sum().item(), 0.0)

    def test_forward_shapes_for_training_and_inference(self):
        self.model.set_eval_mode()
        feature = self._feature_batch([112])
        hidden = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        cell = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        logits, value, next_cell, next_hidden = self.model(
            [feature, hidden, cell], inference=True
        )
        self.assertEqual(tuple(logits.shape), (1, Config.LABEL_SUM))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(next_cell.shape), (1, 1, Config.LSTM_UNIT_SIZE))
        self.assertEqual(tuple(next_hidden.shape), (1, 1, Config.LSTM_UNIT_SIZE))

    def test_ppo_loss_backward_with_new_feature_layout(self):
        self.model.set_train_mode()
        time_steps = Config.LSTM_TIME_STEPS
        feature = self._feature_batch([199] * time_steps)
        legal_action = torch.ones(time_steps, Config.LEGAL_ACTION_DIM)
        seri_vec = torch.cat([feature, legal_action], dim=1)

        data_list = [
            seri_vec,
            torch.linspace(0.0, 1.0, time_steps).unsqueeze(1),
            torch.linspace(-1.0, 1.0, time_steps).unsqueeze(1),
        ]
        data_list.extend(
            torch.zeros(time_steps, 1) for _ in Config.LABEL_SIZE_LIST
        )
        data_list.extend(
            torch.full((time_steps, label_size), 1.0 / label_size)
            for label_size in Config.LABEL_SIZE_LIST
        )
        data_list.extend(
            torch.ones(time_steps, 1) for _ in Config.LABEL_SIZE_LIST
        )
        data_list.extend(
            [
                torch.ones(time_steps, 1),
                torch.zeros(Config.LSTM_UNIT_SIZE),
                torch.zeros(Config.LSTM_UNIT_SIZE),
            ]
        )

        result = self.model(
            [feature, data_list[-1], data_list[-2]],
            inference=False,
        )
        loss, _ = self.model.compute_loss(data_list, result)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        selected_grad = self.model.actor_adapters["199"].up.weight.grad
        self.assertIsNotNone(selected_grad)
        self.assertGreater(selected_grad.abs().sum().item(), 0.0)
        for hero_id in (112, 133):
            grad = self.model.actor_adapters[str(hero_id)].up.weight.grad
            self.assertIsNotNone(grad)
            self.assertEqual(grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
