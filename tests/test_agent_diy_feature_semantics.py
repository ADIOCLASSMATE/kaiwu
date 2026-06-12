import unittest

from agent_diy.conf.conf import FeatureConfig as FC
from agent_diy.feature.feature_process.builder import FeatureBuilder, _rel_to01
from tests.feature_test_utils import (
    cake_field,
    copied_frame,
    enemy_hero,
    hero_field,
    load_obs,
    main_hero,
    minion_field,
    set_slot_config,
    token_slice,
)


class HeroPrivateStateTests(unittest.TestCase):
    CASES = [
        (112, 11204, 4, (11215, 11220, 11231), [1.0, 0.0, 1.0]),
        (133, 13302, 2, (13310, 13325, 13330), [0.0, 1.0, 0.0]),
        (199, 19903, 3, (19915, 19921, 19931), [1.0, 1.0, 1.0]),
    ]

    def test_three_heroes_share_one_private_state_layout(self):
        observation = load_obs("episode_02/frame_01802.json")
        for hero_id, passive_id, phase, active_ids, expected_variants in self.CASES:
            with self.subTest(hero_id=hero_id):
                frame = copied_frame(observation)
                hero = main_hero(frame, observation["camp"])
                hero["config_id"] = hero_id
                set_slot_config(hero, 0, passive_id)
                for slot_type, config_id in enumerate(active_ids, start=1):
                    set_slot_config(hero, slot_type, config_id)

                feature = FeatureBuilder(observation["camp"]).build(frame)
                token = token_slice(feature, "main_hero")
                private = hero_field(token, "private_state")

                self.assertEqual(private[phase], 1.0)
                self.assertEqual(sum(private[:6]), 1.0)
                self.assertEqual(private[6:9], expected_variants)
                self.assertEqual(private[9], 0.0)

    def test_unknown_private_state_uses_unknown_bit(self):
        observation = load_obs("episode_02/frame_01802.json")
        frame = copied_frame(observation)
        hero = main_hero(frame, observation["camp"])
        hero["config_id"] = 133
        set_slot_config(hero, 0, 13999)

        feature = FeatureBuilder(observation["camp"]).build(frame)
        private = hero_field(token_slice(feature, "main_hero"), "private_state")

        self.assertEqual(private[:5], [0.0] * 5)
        self.assertEqual(private[5], 1.0)
        self.assertEqual(private[9], 1.0)

    def test_hidden_enemy_private_state_is_zero(self):
        observation = load_obs("episode_02/frame_01292.json")
        frame = copied_frame(observation)
        hero = enemy_hero(frame, observation["camp"])
        set_slot_config(hero, 0, 13302)
        set_slot_config(hero, 1, 13315)

        feature = FeatureBuilder(observation["camp"]).build(frame)
        private = hero_field(token_slice(feature, "enemy_hero"), "private_state")

        self.assertEqual(private, [0.0] * FC.HERO_PRIVATE_DIM)

    def test_private_state_matches_real_probe_frames(self):
        cases = [
            ("episode_02/frame_01214.json", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]),
            ("episode_03/frame_01874.json", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]),
        ]
        for relative_path, expected_phase, expected_variants in cases:
            with self.subTest(relative_path=relative_path):
                observation = load_obs(relative_path)
                feature = FeatureBuilder(observation["camp"]).build(
                    observation["frame_state"]
                )
                private = hero_field(
                    token_slice(feature, "main_hero"),
                    "private_state",
                )
                self.assertEqual(private[:6], expected_phase)
                self.assertEqual(private[6:9], expected_variants)
                self.assertEqual(private[9], 0.0)


class CakeFeatureTests(unittest.TestCase):
    def test_cakes_are_sorted_by_distance_and_padded(self):
        observation = load_obs("episode_02/frame_01802.json")
        frame = copied_frame(observation)
        hero = main_hero(frame, observation["camp"])
        hx = hero["location"]["x"]
        hz = hero["location"]["z"]
        frame["cakes"] = [
            {"configId": 5, "collider": {"location": {"x": hx + 8000, "z": hz}}},
            {"configId": 5, "collider": {"location": {"x": hx + 1000, "z": hz}}},
        ]

        feature = FeatureBuilder(observation["camp"]).build(frame)
        first = token_slice(feature, "cakes", 0)
        second = token_slice(feature, "cakes", 1)

        self.assertEqual(cake_field(first, "exists"), [1.0])
        self.assertEqual(cake_field(second, "exists"), [1.0])
        self.assertAlmostEqual(
            cake_field(first, "relative_position")[0],
            _rel_to01(1000, FC.ENGAGE_SCALE),
        )
        self.assertAlmostEqual(
            cake_field(second, "relative_position")[0],
            _rel_to01(8000, FC.ENGAGE_SCALE),
        )

        frame["cakes"] = frame["cakes"][:1]
        feature = FeatureBuilder(observation["camp"]).build(frame)
        self.assertEqual(token_slice(feature, "cakes", 1), [0.0] * FC.CAKE_DIM)

    def test_cake_coordinates_follow_camp_mirroring(self):
        observation = load_obs("episode_03/frame_00848.json")
        frame = copied_frame(observation)
        hero = main_hero(frame, observation["camp"])
        hx = hero["location"]["x"]
        hz = hero["location"]["z"]
        frame["cakes"] = [
            {"configId": 5, "collider": {"location": {"x": hx + 1000, "z": hz - 500}}}
        ]

        feature = FeatureBuilder(observation["camp"]).build(frame)
        cake = token_slice(feature, "cakes", 0)

        self.assertAlmostEqual(
            cake_field(cake, "relative_position")[0],
            _rel_to01(-1000, FC.ENGAGE_SCALE),
        )
        self.assertAlmostEqual(
            cake_field(cake, "relative_position")[1],
            _rel_to01(500, FC.ENGAGE_SCALE),
        )

    def test_real_probe_exposes_two_cake_tokens(self):
        observation = load_obs("episode_03/frame_01874.json")
        feature = FeatureBuilder(observation["camp"]).build(
            observation["frame_state"]
        )
        cakes = [token_slice(feature, "cakes", index) for index in range(2)]

        self.assertEqual([cake[0] for cake in cakes], [1.0, 1.0])
        self.assertLess(
            cake_field(cakes[0], "distance")[0],
            cake_field(cakes[1], "distance")[0],
        )


class MinionTypeTests(unittest.TestCase):
    def test_known_and_unknown_minion_types(self):
        observation = load_obs("episode_02/frame_01802.json")
        frame = copied_frame(observation)
        own_minions = [
            npc
            for npc in frame["npc_states"]
            if npc.get("actor_type") == 1
            and npc.get("sub_type") == 11
            and npc.get("camp") == observation["camp"]
        ]

        for minion in own_minions:
            minion["config_id"] = 6805
        feature = FeatureBuilder(observation["camp"]).build(frame)
        known_types = minion_field(
            token_slice(feature, "own_minions", 0), "minion_type"
        )
        self.assertEqual(known_types, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        for minion in own_minions:
            minion["config_id"] = 6899
        feature = FeatureBuilder(observation["camp"]).build(frame)
        unknown_types = minion_field(
            token_slice(feature, "own_minions", 0), "minion_type"
        )
        self.assertEqual(unknown_types, [0.0] * 6 + [1.0])


if __name__ == "__main__":
    unittest.main()
