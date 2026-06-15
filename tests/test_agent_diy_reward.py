import copy
import unittest

from agent_diy.conf.conf import GameConfig
from agent_diy.feature.reward_process import GameRewardManager
from tests.feature_test_utils import load_obs


MAIN_ID = 101
ENEMY_ID = 202


def make_hero(
    runtime_id,
    camp,
    *,
    hp=1000,
    max_hp=1000,
    ep=100,
    max_ep=100,
    level=1,
    exp=0,
    money=0,
    money_cnt=0,
    kill_cnt=0,
    dead_cnt=0,
    attack_range=5000,
    x=0,
    z=0,
):
    return {
        "runtime_id": runtime_id,
        "camp": camp,
        "hp": hp,
        "max_hp": max_hp,
        "ep": ep,
        "max_ep": max_ep,
        "level": level,
        "exp": exp,
        "money": money,
        "money_cnt": money_cnt,
        "kill_cnt": kill_cnt,
        "dead_cnt": dead_cnt,
        "total_hurt_to_hero": 0,
        "attack_range": attack_range,
        "location": {"x": x, "z": z},
    }


def make_tower(camp, *, hp=1000, attack_target=0, x=0, z=0):
    return {
        "runtime_id": 1000 + camp,
        "actor_type": 2,
        "sub_type": 21,
        "camp": camp,
        "hp": hp,
        "max_hp": 1000,
        "attack_range": 5000,
        "attack_target": attack_target,
        "location": {"x": x, "z": z},
    }


def make_frame(
    *,
    frame_no=0,
    main=None,
    enemy=None,
    own_tower=None,
    enemy_tower=None,
    dead_actions=None,
):
    return {
        "frame_no": frame_no,
        "hero_states": [
            main or make_hero(MAIN_ID, 1, x=-10000),
            enemy or make_hero(ENEMY_ID, 2, x=10000),
        ],
        "npc_states": [
            own_tower or make_tower(1, x=-15000),
            enemy_tower or make_tower(2, x=15000),
        ],
        "frame_action": {"dead_action": dead_actions or []},
    }


class RewardDesignTests(unittest.TestCase):
    def setUp(self):
        self.manager = GameRewardManager(MAIN_ID)

    def test_reward_configuration_prioritizes_objectives_without_duplicates(self):
        self.assertEqual(
            set(GameConfig.REWARD_WEIGHT_DICT),
            {
                "tower_hp_point",
                "hp_point",
                "kill",
                "money",
                "exp",
                "last_hit",
                "kill_monster",
                "retreat_penalty",
                "idle_penalty",
            },
        )
        self.assertGreater(GameConfig.TERMINAL_WIN_REWARD, 0)
        self.assertGreater(
            GameConfig.TERMINAL_WIN_REWARD,
            GameConfig.REWARD_WEIGHT_DICT["tower_hp_point"],
        )

    def test_first_frame_has_no_artificial_shaping_reward(self):
        reward = self.manager.result(make_frame())

        self.assertAlmostEqual(reward["reward_sum"], 0.0)
        for name in GameConfig.REWARD_WEIGHT_DICT:
            self.assertAlmostEqual(reward[name], 0.0)

    def test_missing_towers_on_first_observation_are_safe(self):
        frame = make_frame()
        frame["npc_states"] = []

        reward = self.manager.result(frame)

        self.assertEqual(reward["tower_hp_point"], 0.0)
        self.assertTrue(all(value == value for value in reward.values()))

    def test_purchase_does_not_create_negative_money_reward(self):
        initial = make_frame(
            main=make_hero(MAIN_ID, 1, money=1000, money_cnt=1200, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, money=500, money_cnt=900, x=10000),
        )
        after_purchase = copy.deepcopy(initial)
        after_purchase["frame_no"] = 1
        after_purchase["hero_states"][0]["money"] = 100

        self.manager.result(initial)
        reward = self.manager.result(after_purchase)

        self.assertAlmostEqual(reward["money"], 0.0)

    def test_level_up_uses_monotonic_total_experience(self):
        initial = make_frame(
            main=make_hero(MAIN_ID, 1, level=1, exp=150, x=-10000),
            enemy=make_hero(ENEMY_ID, 2, level=1, exp=100, x=10000),
        )
        leveled = copy.deepcopy(initial)
        leveled["frame_no"] = 1
        leveled["hero_states"][0]["level"] = 2
        leveled["hero_states"][0]["exp"] = 10

        self.manager.result(initial)
        reward = self.manager.result(leveled)

        self.assertAlmostEqual(reward["exp"], 0.02)

    def test_last_hit_and_monster_control_use_death_event_attribution(self):
        initial = make_frame()
        event_frame = make_frame(
            frame_no=1,
            dead_actions=[
                {
                    "death": {
                        "runtime_id": 301,
                        "camp": 2,
                        "sub_type": "ACTOR_SUB_SOLDIER",
                    },
                    "killer": {"runtime_id": MAIN_ID},
                },
                {
                    "death": {
                        "runtime_id": 302,
                        "camp": "PLAYERCAMP_MID",
                        "sub_type": "ACTOR_SUB_MONSTER",
                    },
                    "killer": {"runtime_id": MAIN_ID},
                },
            ],
        )

        self.manager.result(initial)
        reward = self.manager.result(event_frame)

        self.assertEqual(reward["last_hit"], 1.0)
        self.assertEqual(reward["kill_monster"], 1.0)

    def test_enemy_objective_events_are_negative(self):
        initial = make_frame()
        event_frame = make_frame(
            frame_no=1,
            dead_actions=[
                {
                    "death": {
                        "runtime_id": 301,
                        "camp": 1,
                        "sub_type": "ACTOR_SUB_SOLDIER",
                    },
                    "killer": {"runtime_id": ENEMY_ID},
                },
                {
                    "death": {
                        "runtime_id": 302,
                        "camp": "PLAYERCAMP_MID",
                        "sub_type": "ACTOR_SUB_MONSTER",
                    },
                    "killer": {"runtime_id": ENEMY_ID},
                },
            ],
        )

        self.manager.result(initial)
        reward = self.manager.result(event_frame)

        self.assertEqual(reward["last_hit"], -1.0)
        self.assertEqual(reward["kill_monster"], -1.0)

    def test_real_probe_dead_action_is_attributed_to_the_enemy(self):
        observation = load_obs("episode_03/frame_01874.json")
        manager = GameRewardManager(observation["player_id"])

        reward = manager.result(observation["frame_state"])

        self.assertEqual(reward["last_hit"], -1.0)

    def test_retreat_penalty_only_targets_high_hp_turtling(self):
        behind_tower = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, x=-18000),
        )
        normal_lane = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, hp=1000, x=-5000),
        )
        low_hp_retreat = make_frame(
            frame_no=2,
            main=make_hero(MAIN_ID, 1, hp=300, x=-18000),
        )

        turtling_reward = self.manager.result(behind_tower)
        lane_reward = self.manager.result(normal_lane)
        retreat_reward = self.manager.result(low_hp_retreat)

        self.assertGreater(turtling_reward["retreat_penalty"], 0.0)
        self.assertEqual(lane_reward["retreat_penalty"], 0.0)
        self.assertEqual(retreat_reward["retreat_penalty"], 0.0)

    def test_tower_damage_is_discounted_when_diving(self):
        initial = make_frame()
        safe_damage = make_frame(
            frame_no=1,
            enemy_tower=make_tower(2, hp=900, x=15000),
        )

        self.manager.result(initial)
        safe_reward = self.manager.result(safe_damage)["tower_hp_point"]

        diving_manager = GameRewardManager(MAIN_ID)
        diving_manager.result(initial)
        diving_damage = make_frame(
            frame_no=1,
            enemy_tower=make_tower(
                2,
                hp=900,
                attack_target=MAIN_ID,
                x=15000,
            ),
        )
        diving_reward = diving_manager.result(diving_damage)["tower_hp_point"]

        self.assertAlmostEqual(safe_reward, 0.1)
        self.assertAlmostEqual(
            diving_reward,
            safe_reward * GameConfig.TOWER_DIVE_DISCOUNT,
        )

    def test_terminal_outcome_is_strong_symmetric_and_idempotent(self):
        reward = {"reward_sum": 1.25}
        win_bonus = self.manager.apply_terminal_outcome(
            reward,
            make_frame(enemy_tower=make_tower(2, hp=0, x=15000)),
            win=None,
        )

        self.assertEqual(win_bonus, GameConfig.TERMINAL_WIN_REWARD)
        self.assertEqual(reward["terminal"], 1.0)
        self.assertEqual(
            reward["reward_sum"],
            1.25 + GameConfig.TERMINAL_WIN_REWARD,
        )
        self.assertEqual(
            self.manager.apply_terminal_outcome(reward, make_frame(), win=1),
            0.0,
        )

        loss_reward = {"reward_sum": 0.0}
        loss_manager = GameRewardManager(MAIN_ID)
        loss_bonus = loss_manager.apply_terminal_outcome(
            loss_reward,
            make_frame(own_tower=make_tower(1, hp=0, x=-15000)),
            win=None,
        )
        self.assertEqual(loss_bonus, -GameConfig.TERMINAL_WIN_REWARD)
        self.assertEqual(loss_reward["terminal"], -1.0)

        draw_reward = {"reward_sum": 0.0}
        draw_manager = GameRewardManager(MAIN_ID)
        self.assertEqual(
            draw_manager.apply_terminal_outcome(
                draw_reward,
                make_frame(),
                win=-1,
            ),
            0.0,
        )
        self.assertEqual(draw_reward["terminal"], 0.0)

    def test_out_of_range_penalty_is_one_shot_and_action_conditioned(self):
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, attack_range=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, x=5000),
        )
        attack_enemy = [3, 0, 0, 0, 0, 1]

        self.assertEqual(
            self.manager.out_of_range_penalty([0, 0, 0, 0, 0, 1], frame),
            0.0,
        )
        self.manager.set_distance_penalty(attack_enemy, frame)
        reward = self.manager.result(frame)
        next_reward = self.manager.result(frame)

        self.assertEqual(
            reward["distance_penalty"],
            -GameConfig.OUT_OF_RANGE_PENALTY,
        )
        self.assertEqual(next_reward["distance_penalty"], 0.0)

    def test_idle_penalty_has_bounded_per_frame_scale(self):
        frame = make_frame()
        self.manager.result(frame)
        self.manager._inactive_frames = (
            GameConfig.IDLE_GRACE_FRAMES + GameConfig.IDLE_RAMP_FRAMES
        )

        reward = self.manager.result(frame)

        self.assertEqual(
            reward["idle_penalty"],
            GameConfig.IDLE_FRAME_SCALE,
        )
        self.assertGreater(
            reward["idle_penalty"] * GameConfig.REWARD_WEIGHT_DICT["idle_penalty"],
            -0.01,
        )


if __name__ == "__main__":
    unittest.main()
