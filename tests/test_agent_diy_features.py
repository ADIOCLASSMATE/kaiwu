#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import copy
import json
import math
import unittest
from pathlib import Path

from agent_diy.conf.conf import Config, FeatureConfig as FC
from agent_diy.feature.feature_process.builder import FeatureBuilder, _rel_to01
from agent_diy.feature.reward_process import GameRewardManager
from agent_diy.feature.targeting import target_slot_enemy_soldiers


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "diag_feature_probes"


def load_obs(rel_path):
    data = json.loads((PROBES / rel_path).read_text(encoding="utf-8"))
    observations = data["observation"]
    return next(iter(observations.values()))


def token_layout():
    layout = []
    offset = 0
    for type_key, dim, count in FC.TOKEN_SEGMENTS:
        for index in range(count):
            layout.append((type_key, index, offset, dim))
            offset += dim
    return layout


def token_slice(feature, type_key, index=0):
    for key, idx, offset, dim in token_layout():
        if key == type_key and idx == index:
            return feature[offset:offset + dim]
    raise AssertionError(f"missing token {type_key}[{index}]")


def main_pos(obs, mirror=False):
    hero = next(h for h in obs["frame_state"]["hero_states"] if h["camp"] == obs["camp"])
    x = hero["location"]["x"]
    z = hero["location"]["z"]
    return (-x, -z) if mirror else (x, z)


def ordered_own_soldier_ids(obs):
    mx, mz = main_pos(obs)
    own = []
    for npc in obs["frame_state"]["npc_states"]:
        if (
            npc.get("actor_type") == 1
            and npc.get("sub_type") == 11
            and npc.get("camp") == obs["camp"]
            and npc.get("hp", 0) > 0
        ):
            loc = npc["location"]
            distance = math.hypot(loc["x"] - mx, loc["z"] - mz)
            own.append((distance, npc["runtime_id"]))
    return [runtime_id for _, runtime_id in sorted(own)]


class AgentDiyFeatureProbeTests(unittest.TestCase):
    def test_soldier3_slot_uses_runtime_id_order_at_probe_25(self):
        obs = load_obs("episode_02/frame_01802.json")
        ordered = target_slot_enemy_soldiers(
            obs["frame_state"]["npc_states"],
            main_pos(obs),
            obs["camp"],
            FC.N_MINION_PER_CAMP,
        )
        self.assertEqual([item["unit"]["runtime_id"] for item in ordered], [101, 106, 110])

        feature = FeatureBuilder(obs["camp"]).build(obs["frame_state"])
        soldier3 = token_slice(feature, "enemy_minions", 2)
        expected = next(n for n in obs["frame_state"]["npc_states"] if n["runtime_id"] == 110)
        mx, mz = main_pos(obs)
        self.assertAlmostEqual(soldier3[3], _rel_to01(expected["location"]["x"] - mx, FC.ENGAGE_SCALE))
        self.assertAlmostEqual(soldier3[4], _rel_to01(expected["location"]["z"] - mz, FC.ENGAGE_SCALE))

        reward = GameRewardManager(obs["player_id"])
        self.assertEqual(reward._nth_enemy_minion_pos(obs["frame_state"], 2), (-3561, -3362))

    def test_soldier3_slot_uses_runtime_id_order_at_probe_38(self):
        obs = load_obs("episode_02/frame_05264.json")
        ordered = target_slot_enemy_soldiers(
            obs["frame_state"]["npc_states"],
            main_pos(obs),
            obs["camp"],
            FC.N_MINION_PER_CAMP,
        )
        self.assertEqual([item["unit"]["runtime_id"] for item in ordered], [474, 479, 486])

        reward = GameRewardManager(obs["player_id"])
        self.assertEqual(reward._nth_enemy_minion_pos(obs["frame_state"], 2), (-9323, -7094))

    def test_hero_abilities_include_unknown_bits_and_respect_fog(self):
        obs = load_obs("episode_02/frame_01214.json")
        feature = FeatureBuilder(obs["camp"]).build(obs["frame_state"])
        enemy = token_slice(feature, "enemy_hero")
        ability_start = FC.HERO_DIM - FC.HERO_ABILITY_DIM - FC.ATTACK_TARGET_DIM - 1
        ability_values = enemy[ability_start:ability_start + FC.HERO_ABILITY_DIM]
        by_bit = dict(zip(FC.HERO_ABILITY_BITS, ability_values))
        self.assertEqual(by_bit[1], 1.0)
        self.assertEqual(by_bit[31], 1.0)

        hidden_obs = load_obs("episode_03/frame_01874.json")
        mutated = copy.deepcopy(hidden_obs["frame_state"])
        hidden_enemy = next(h for h in mutated["hero_states"] if h["camp"] != hidden_obs["camp"])
        hidden_enemy["camp_visible"][hidden_obs["camp"] - 1] = False
        hidden_enemy["abilities"][33] = True
        feature = FeatureBuilder(hidden_obs["camp"]).build(mutated)
        enemy = token_slice(feature, "enemy_hero")
        ability_values = enemy[ability_start:ability_start + FC.HERO_ABILITY_DIM]
        self.assertEqual(sum(ability_values), 0.0)

    def test_arli_mark_layers_on_enemy_minions(self):
        cases = [
            ("episode_03/frame_00848.json", 20, 1.0 / 3.0),
            ("episode_03/frame_00914.json", 20, 1.0),
            ("episode_03/frame_01118.json", 22, 0.0),
        ]
        for rel_path, runtime_id, layer_ratio in cases:
            with self.subTest(rel_path=rel_path):
                obs = load_obs(rel_path)
                marked = next(
                    n for n in obs["frame_state"]["npc_states"]
                    if n["runtime_id"] == runtime_id
                )
                if marked["camp"] == obs["camp"]:
                    type_key = "own_minions"
                    slot = ordered_own_soldier_ids(obs).index(runtime_id)
                else:
                    type_key = "enemy_minions"
                    ordered = target_slot_enemy_soldiers(
                        obs["frame_state"]["npc_states"],
                        main_pos(obs),
                        obs["camp"],
                        FC.N_MINION_PER_CAMP,
                    )
                    slot = [item["unit"]["runtime_id"] for item in ordered].index(runtime_id)
                feature = FeatureBuilder(obs["camp"]).build(obs["frame_state"])
                minion = token_slice(feature, type_key, slot)
                mark_start = FC.MINION_DIM - FC.ARLI_MARK_DIM
                self.assertEqual(minion[mark_start], 1.0)
                self.assertAlmostEqual(minion[mark_start + 1], layer_ratio)

    def test_attack_target_semantics_from_frame_5258(self):
        obs = load_obs("episode_02/frame_05258.json")
        ordered = target_slot_enemy_soldiers(
            obs["frame_state"]["npc_states"],
            main_pos(obs),
            obs["camp"],
            FC.N_MINION_PER_CAMP,
        )
        slot = [item["unit"]["runtime_id"] for item in ordered].index(479)
        feature = FeatureBuilder(obs["camp"]).build(obs["frame_state"])

        minion = token_slice(feature, "enemy_minions", slot)
        attack_start = FC.MINION_DIM - FC.ARLI_MARK_DIM - FC.ATTACK_TARGET_DIM
        has_target, targets_me, *_ = minion[attack_start:attack_start + FC.ATTACK_TARGET_DIM]
        self.assertEqual(has_target, 1.0)
        self.assertEqual(targets_me, 1.0)

        hero = token_slice(feature, "enemy_hero")
        hero_attack_start = FC.HERO_DIM - FC.ATTACK_TARGET_DIM - 1
        hero_bits = hero[hero_attack_start:hero_attack_start + FC.ATTACK_TARGET_DIM]
        self.assertEqual(hero_bits[0], 1.0)
        self.assertEqual(hero_bits[3], 1.0)

    def test_hero_bullet_source_and_mirrored_velocity(self):
        obs1 = load_obs("episode_03/frame_01874.json")
        obs2 = load_obs("episode_03/frame_01880.json")
        builder = FeatureBuilder(obs1["camp"])
        builder.build(obs1["frame_state"])
        feature = builder.build(obs2["frame_state"])

        bullet_tokens = [token_slice(feature, "bullets", i) for i in range(FC.N_BULLETS)]
        slot_start = 1 + 1 + FC.HERO_ID_ONEHOT_DIM
        velocity_start = FC.BULLET_DIM - 2
        arli_index = FC.HERO_CONFIG_IDS.index(199)

        matching = []
        for token in bullet_tokens:
            if token[0] <= 0.5:
                continue
            source_onehot = token[2:2 + FC.HERO_ID_ONEHOT_DIM]
            slot_onehot = token[slot_start:slot_start + FC.BULLET_SLOT_ONEHOT_DIM]
            if source_onehot[arli_index] == 1.0 and slot_onehot[2] == 1.0:
                matching.append(token)

        self.assertEqual(len(matching), 1)
        token = matching[0]
        self.assertEqual(token[1], 1.0)  # source is enemy from camp2 perspective.
        expected_vx = -(2810 - 1952) / 6.0
        expected_vz = -(7710 - 7298) / 6.0
        self.assertAlmostEqual(token[velocity_start], _rel_to01(expected_vx, FC.BULLET_VEL_SCALE))
        self.assertAlmostEqual(token[velocity_start + 1], _rel_to01(expected_vz, FC.BULLET_VEL_SCALE))

    def test_feature_length_exists_masks_and_no_monster_target_dim(self):
        obs = load_obs("episode_02/frame_01802.json")
        feature = FeatureBuilder(obs["camp"]).build(obs["frame_state"])
        self.assertEqual(len(feature), FC.FEATURE_DIM)
        self.assertEqual(FC.MONSTER_DIM, 8)
        for type_key, index, _, _ in token_layout():
            token = token_slice(feature, type_key, index)
            self.assertIn(token[0], (0.0, 1.0))

    def test_model_forward_shapes_when_torch_is_available(self):
        try:
            import torch
            from agent_diy.model.model import Model
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this local environment")

        model = Model()
        model.set_eval_mode()
        feature = torch.zeros(1, FC.FEATURE_DIM)
        feature[0, 0] = 1.0
        hidden = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        cell = torch.zeros(1, Config.LSTM_UNIT_SIZE)
        logits, value, next_cell, next_hidden = model([feature, hidden, cell], inference=True)
        self.assertEqual(tuple(logits.shape), (1, Config.LABEL_SUM))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(next_cell.shape), (1, 1, Config.LSTM_UNIT_SIZE))
        self.assertEqual(tuple(next_hidden.shape), (1, 1, Config.LSTM_UNIT_SIZE))


if __name__ == "__main__":
    unittest.main()
