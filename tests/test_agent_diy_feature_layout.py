import math
import unittest

from agent_diy.conf.conf import Config, FeatureConfig as FC
from agent_diy.feature.feature_process import FeatureProcess
from agent_diy.feature.feature_process.builder import FeatureBuilder
from tests.feature_test_utils import iter_probe_observations, load_obs


class FeatureLayoutTests(unittest.TestCase):
    def test_feature_dimensions_and_token_count(self):
        self.assertEqual(FC.HERO_DIM, 124)
        self.assertEqual(FC.MINION_DIM, 22)
        self.assertEqual(FC.CAKE_DIM, 6)
        self.assertEqual(FC.NUM_TOKENS, 19)
        self.assertEqual(FC.TOKEN_FEATURE_DIM, 542)
        self.assertEqual(FC.FEATURE_DIM, 551)
        self.assertEqual(Config.SERI_VEC_SPLIT_SHAPE, [(551,), (85,)])
        self.assertEqual(Config.DATA_SPLIT_SHAPE[0], 636)
        self.assertEqual(
            Config.data_shapes[0],
            [Config.DATA_SPLIT_SHAPE[0] * Config.LSTM_TIME_STEPS],
        )
        self.assertEqual(
            Config.SAMPLE_DIM,
            sum(Config.DATA_SPLIT_SHAPE[:-2]) * Config.LSTM_TIME_STEPS
            + sum(Config.DATA_SPLIT_SHAPE[-2:]),
        )

    def test_field_layouts_are_contiguous_and_complete(self):
        layouts = [
            (FC.HERO_FIELDS, FC.HERO_FIELD_SLICES, FC.HERO_DIM),
            (FC.MINION_FIELDS, FC.MINION_FIELD_SLICES, FC.MINION_DIM),
            (FC.CAKE_FIELDS, FC.CAKE_FIELD_SLICES, FC.CAKE_DIM),
        ]
        for fields, slices, expected_dim in layouts:
            with self.subTest(expected_dim=expected_dim):
                cursor = 0
                for name, width in fields:
                    field_slice = slices[name]
                    self.assertEqual(field_slice.start, cursor)
                    self.assertEqual(field_slice.stop, cursor + width)
                    cursor += width
                self.assertEqual(cursor, expected_dim)

    def test_key_field_offsets_are_explicit(self):
        expected = {
            "skills": (38, 73),
            "private_state": (73, 83),
            "range_flags": (83, 85),
            "abilities": (85, 95),
            "attack_target": (95, 100),
            "equipment": (100, 124),
        }
        for name, (start, stop) in expected.items():
            with self.subTest(name=name):
                field_slice = FC.HERO_FIELD_SLICES[name]
                self.assertEqual((field_slice.start, field_slice.stop), (start, stop))

        self.assertEqual(
            (
                FC.MINION_FIELD_SLICES["attack_target"].start,
                FC.MINION_FIELD_SLICES["attack_target"].stop,
            ),
            (8, 13),
        )
        self.assertEqual(
            (
                FC.MINION_FIELD_SLICES["minion_type"].start,
                FC.MINION_FIELD_SLICES["minion_type"].stop,
            ),
            (15, 22),
        )

    def test_token_layout_is_contiguous_and_complete(self):
        cursor = 0
        token_count = 0
        for type_key, dim, count in FC.TOKEN_SEGMENTS:
            ranges = FC.TOKEN_SLICES[type_key]
            self.assertEqual(len(ranges), count)
            for token_range in ranges:
                self.assertEqual(token_range.start, cursor)
                self.assertEqual(token_range.stop, cursor + dim)
                cursor += dim
                token_count += 1
        self.assertEqual(cursor, FC.TOKEN_FEATURE_DIM)
        self.assertEqual(token_count, FC.NUM_TOKENS)

    def test_all_probe_frames_build_finite_features(self):
        for path, observation in iter_probe_observations():
            with self.subTest(path=path):
                feature = FeatureBuilder(observation["camp"]).build(
                    observation["frame_state"]
                )
                self.assertEqual(len(feature), FC.FEATURE_DIM)
                self.assertTrue(all(math.isfinite(value) for value in feature))

    def test_feature_process_stats_include_new_token_group(self):
        observation = load_obs("episode_03/frame_01874.json")
        process = FeatureProcess(observation["camp"])
        feature = process.process_feature(observation)
        stats = process.get_stats()

        self.assertEqual(len(feature), FC.FEATURE_DIM)
        self.assertEqual(stats["feat_frames"], 1)
        self.assertEqual(stats["feat_cakes_exists"], 1.0)
        self.assertNotIn("feat_cakes_exists", process.get_stats())


if __name__ == "__main__":
    unittest.main()
