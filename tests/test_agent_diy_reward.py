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
    total_hurt_to_hero=0,
    attack_range=5000,
    attack_target=0,
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
        "total_hurt_to_hero": total_hurt_to_hero,
        "attack_range": attack_range,
        "attack_target": attack_target,
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


def make_cake(x, z):
    return {
        "configId": 5,
        "collider": {
            "location": {"x": x, "y": 48, "z": z},
            "radius": 0,
        },
    }


def make_frame(
    *,
    frame_no=0,
    main=None,
    enemy=None,
    own_tower=None,
    enemy_tower=None,
    npcs=None,
    cakes=None,
    dead_actions=None,
):
    return {
        "frame_no": frame_no,
        "hero_states": [
            main or make_hero(MAIN_ID, 1, x=-10000),
            enemy or make_hero(ENEMY_ID, 2, x=10000),
        ],
        "npc_states": npcs or [
            own_tower or make_tower(1, x=-15000),
            enemy_tower or make_tower(2, x=15000),
        ],
        "cakes": cakes or [],
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
                "lane_progress",
                "hp_point",
                "danger_penalty",
                "kill",
                "death",
                "money",
                "exp",
                "last_hit",
                "minion_hp_point",
                "kill_monster",
                "idle_penalty",
            },
        )
        self.assertGreater(GameConfig.TERMINAL_WIN_REWARD, 0)
        self.assertGreater(
            GameConfig.TERMINAL_WIN_REWARD,
            GameConfig.REWARD_WEIGHT_DICT["tower_hp_point"],
        )
        self.assertLess(
            GameConfig.REWARD_WEIGHT_DICT["minion_hp_point"],
            GameConfig.REWARD_WEIGHT_DICT["last_hit"],
        )
        self.assertLess(
            GameConfig.REWARD_WEIGHT_DICT["minion_hp_point"],
            GameConfig.REWARD_WEIGHT_DICT["hp_point"],
        )

    def test_first_frame_has_no_artificial_shaping_reward(self):
        reward = self.manager.result(
            make_frame(main=make_hero(MAIN_ID, 1, hp=980, max_hp=1000))
        )

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

    def test_hp_point_uses_only_hero_damage_attribution(self):
        initial = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, total_hurt_to_hero=0),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, total_hurt_to_hero=0),
        )
        non_hero_damage = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, hp=1000, total_hurt_to_hero=0),
            enemy=make_hero(ENEMY_ID, 2, hp=500, total_hurt_to_hero=0),
        )
        hero_damage = make_frame(
            frame_no=2,
            main=make_hero(MAIN_ID, 1, hp=1000, total_hurt_to_hero=300),
            enemy=make_hero(ENEMY_ID, 2, hp=500, total_hurt_to_hero=0),
        )
        enemy_trade = make_frame(
            frame_no=3,
            main=make_hero(MAIN_ID, 1, hp=800, total_hurt_to_hero=300),
            enemy=make_hero(ENEMY_ID, 2, hp=500, total_hurt_to_hero=150),
        )

        self.manager.result(initial)
        non_hero_reward = self.manager.result(non_hero_damage)
        hero_reward = self.manager.result(hero_damage)
        trade_reward = self.manager.result(enemy_trade)

        self.assertEqual(non_hero_reward["hp_point"], 0.0)
        self.assertAlmostEqual(
            hero_reward["hp_point"],
            300 / GameConfig.HERO_DAMAGE_REWARD_SCALE,
        )
        self.assertAlmostEqual(
            trade_reward["hp_point"],
            -150 / GameConfig.HERO_DAMAGE_REWARD_SCALE,
        )

    def test_death_penalty_tracks_only_main_hero_deaths(self):
        initial = make_frame(
            main=make_hero(MAIN_ID, 1, dead_cnt=0),
            enemy=make_hero(ENEMY_ID, 2, dead_cnt=0),
        )
        main_dead = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, dead_cnt=1),
            enemy=make_hero(ENEMY_ID, 2, dead_cnt=0),
        )
        enemy_dead = make_frame(
            frame_no=2,
            main=make_hero(MAIN_ID, 1, dead_cnt=1),
            enemy=make_hero(ENEMY_ID, 2, dead_cnt=1),
        )

        self.manager.result(initial)
        death_reward = self.manager.result(main_dead)
        enemy_death_reward = self.manager.result(enemy_dead)

        self.assertEqual(death_reward["death"], 1.0)
        self.assertLess(
            death_reward["death"] * GameConfig.REWARD_WEIGHT_DICT["death"],
            0.0,
        )
        self.assertEqual(enemy_death_reward["death"], 0.0)

    def test_tower_suicide_still_receives_death_penalty(self):
        initial = make_frame(
            main=make_hero(MAIN_ID, 1, dead_cnt=0, x=14500),
            enemy=make_hero(ENEMY_ID, 2, dead_cnt=0, x=9000),
            enemy_tower=make_tower(2, attack_target=MAIN_ID, x=15000),
        )
        tower_death = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, dead_cnt=1, hp=0, x=14500),
            enemy=make_hero(ENEMY_ID, 2, dead_cnt=0, x=9000),
            enemy_tower=make_tower(2, attack_target=MAIN_ID, x=15000),
        )

        self.manager.result(initial)
        reward = self.manager.result(tower_death)

        self.assertEqual(reward["death"], 1.0)
        self.assertLess(
            reward["death"] * GameConfig.REWARD_WEIGHT_DICT["death"],
            0.0,
        )

    def test_minion_hp_point_does_not_reward_clearing_enemy_minions(self):
        initial = make_frame(
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(301, 2, hp=1000, x=1000),
            ],
        )
        enemy_minion_damaged = make_frame(
            frame_no=1,
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(301, 2, hp=700, x=1000),
                make_minion(302, 2, hp=1000, x=1500),
            ],
        )

        self.manager.result(initial)
        reward = self.manager.result(enemy_minion_damaged)

        self.assertAlmostEqual(reward["minion_hp_point"], 0.0)

    def test_minion_hp_point_penalizes_visible_enemy_hero_hitting_own_minions(self):
        initial = make_frame(
            enemy=make_hero(ENEMY_ID, 2, attack_target=401, x=1000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 1, hp=1000, x=-1000),
            ],
        )
        own_minion_damaged = make_frame(
            frame_no=1,
            enemy=make_hero(ENEMY_ID, 2, attack_target=401, x=1000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 1, hp=800, x=-1000),
            ],
        )

        self.manager.result(initial)
        reward = self.manager.result(own_minion_damaged)

        self.assertAlmostEqual(reward["minion_hp_point"], -0.2)
        self.assertAlmostEqual(
            reward["minion_hp_point"] * GameConfig.REWARD_WEIGHT_DICT["minion_hp_point"],
            -0.02,
        )

    def test_minion_hp_point_ignores_non_hero_lane_damage_to_own_minions(self):
        initial = make_frame(
            enemy=make_hero(ENEMY_ID, 2, attack_target=0, x=1000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 1, hp=1000, x=-1000),
            ],
        )
        own_minion_damaged_by_wave = make_frame(
            frame_no=1,
            enemy=make_hero(ENEMY_ID, 2, attack_target=0, x=1000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, x=15000),
                make_minion(401, 1, hp=800, x=-1000),
            ],
        )

        self.manager.result(initial)
        reward = self.manager.result(own_minion_damaged_by_wave)

        self.assertAlmostEqual(reward["minion_hp_point"], 0.0)

    def test_real_probe_dead_action_is_attributed_to_the_enemy(self):
        observation = load_obs("episode_03/frame_01874.json")
        manager = GameRewardManager(observation["player_id"])

        reward = manager.result(observation["frame_state"])

        self.assertEqual(reward["last_hit"], -1.0)

    def test_full_hp_lane_guidance_decays_from_fountain_to_own_cake_then_center(self):
        cakes = [
            make_cake(-15200, 0),
            make_cake(15200, 0),
        ]
        fountain = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, x=-23000),
            cakes=cakes,
        )
        own_cake = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, hp=1000, x=-15200),
            cakes=cakes,
        )
        between_cake_and_center = make_frame(
            frame_no=2,
            main=make_hero(MAIN_ID, 1, hp=1000, x=-7600),
            cakes=cakes,
        )
        center = make_frame(
            frame_no=3,
            main=make_hero(MAIN_ID, 1, hp=1000, x=0),
            cakes=cakes,
        )

        fountain_reward = self.manager.result(fountain)
        cake_reward = self.manager.result(own_cake)
        weak_reward = self.manager.result(between_cake_and_center)
        center_reward = self.manager.result(center)

        self.assertAlmostEqual(fountain_reward["lane_progress"], 1.0)
        self.assertLess(cake_reward["lane_progress"], 0.06)
        self.assertGreater(cake_reward["lane_progress"], 0.0)
        self.assertLess(weak_reward["lane_progress"], cake_reward["lane_progress"])
        self.assertGreater(weak_reward["lane_progress"], 0.0)
        self.assertEqual(center_reward["lane_progress"], 0.0)

        lane_weight = GameConfig.REWARD_WEIGHT_DICT["lane_progress"]
        self.assertLess(fountain_reward["lane_progress"] * lane_weight, 0.0)

    def test_lane_guidance_only_applies_at_near_full_hp(self):
        cakes = [make_cake(-15200, 0)]
        low_hp_fountain = make_frame(
            main=make_hero(MAIN_ID, 1, hp=980, max_hp=1000, x=-23000),
            cakes=cakes,
        )

        reward = self.manager.result(low_hp_fountain)

        self.assertEqual(reward["lane_progress"], 0.0)

    def test_low_hp_in_enemy_threat_area_is_penalized(self):
        initial = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, attack_range=5000, x=3000),
        )
        danger = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, hp=300, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, attack_range=5000, x=3000),
        )
        safe_retreat = make_frame(
            frame_no=2,
            main=make_hero(MAIN_ID, 1, hp=300, x=-18000),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, attack_range=5000, x=3000),
        )

        self.manager.result(initial)
        danger_reward = self.manager.result(danger)
        retreat_reward = self.manager.result(safe_retreat)

        self.assertGreater(danger_reward["danger_penalty"], 0.0)
        self.assertLess(
            danger_reward["danger_penalty"]
            * GameConfig.REWARD_WEIGHT_DICT["danger_penalty"],
            0.0,
        )
        self.assertEqual(retreat_reward["danger_penalty"], 0.0)

    def test_invisible_enemy_hero_does_not_trigger_danger_penalty(self):
        enemy = make_hero(ENEMY_ID, 2, hp=1000, attack_range=5000, x=3000)
        enemy["camp_visible"] = [False, True]
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, x=0),
            enemy=enemy,
        )

        reward = self.manager.result(frame)

        self.assertEqual(reward["danger_penalty"], 0.0)

    def test_enemy_hero_danger_penalty_scales_with_enemy_hp(self):
        full_hp_enemy = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, max_hp=1000, attack_range=5000, x=3000),
        )
        lower_but_still_dangerous_enemy = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, x=0),
            enemy=make_hero(ENEMY_ID, 2, hp=250, max_hp=1000, attack_range=5000, x=3000),
        )

        full_hp_reward = GameRewardManager(MAIN_ID).result(full_hp_enemy)
        low_hp_reward = GameRewardManager(MAIN_ID).result(lower_but_still_dangerous_enemy)

        self.assertGreater(full_hp_reward["danger_penalty"], 0.0)
        self.assertGreater(low_hp_reward["danger_penalty"], 0.0)
        self.assertLess(low_hp_reward["danger_penalty"], full_hp_reward["danger_penalty"])

    def test_much_lower_hp_enemy_hero_does_not_trigger_danger_penalty(self):
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=0),
            enemy=make_hero(
                ENEMY_ID,
                2,
                hp=200,
                max_hp=1000,
                attack_range=5000,
                x=3000,
            ),
        )

        reward = self.manager.result(frame)

        self.assertEqual(reward["danger_penalty"], 0.0)

    def test_slightly_lower_hp_enemy_hero_still_triggers_danger_penalty(self):
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=0),
            enemy=make_hero(
                ENEMY_ID,
                2,
                hp=280,
                max_hp=1000,
                attack_range=5000,
                x=3000,
            ),
        )

        reward = self.manager.result(frame)

        self.assertGreater(reward["danger_penalty"], 0.0)

    def test_enemy_tower_danger_is_not_waived_by_low_enemy_hero_hp(self):
        frame = make_frame(
            main=make_hero(MAIN_ID, 1, hp=300, max_hp=1000, x=14500),
            enemy=make_hero(
                ENEMY_ID,
                2,
                hp=100,
                max_hp=1000,
                attack_range=5000,
                x=3000,
            ),
            enemy_tower=make_tower(2, attack_target=MAIN_ID, x=15000),
        )

        reward = self.manager.result(frame)

        self.assertGreater(reward["danger_penalty"], 0.0)

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

    def test_tower_damage_is_discounted_without_lane_pressure(self):
        initial = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, x=14500),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, x=9000),
            enemy_tower=make_tower(2, hp=1000, x=15000),
        )
        no_wave_damage = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, hp=1000, x=14500),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, x=9000),
            enemy_tower=make_tower(2, hp=900, x=15000),
        )

        self.manager.result(initial)
        no_wave_reward = self.manager.result(no_wave_damage)["tower_hp_point"]

        with_wave_manager = GameRewardManager(MAIN_ID)
        with_wave_initial = make_frame(
            main=make_hero(MAIN_ID, 1, hp=1000, x=14500),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, x=9000),
            enemy_tower=make_tower(2, hp=1000, x=15000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, hp=1000, x=15000),
                make_minion(301, 1, x=14200),
            ],
        )
        with_wave_damage = make_frame(
            frame_no=1,
            main=make_hero(MAIN_ID, 1, hp=1000, x=14500),
            enemy=make_hero(ENEMY_ID, 2, hp=1000, x=9000),
            enemy_tower=make_tower(2, hp=900, x=15000),
            npcs=[
                make_tower(1, x=-15000),
                make_tower(2, hp=900, x=15000),
                make_minion(301, 1, x=14200),
            ],
        )
        with_wave_manager.result(with_wave_initial)
        with_wave_reward = with_wave_manager.result(with_wave_damage)["tower_hp_point"]

        self.assertAlmostEqual(
            no_wave_reward,
            with_wave_reward * GameConfig.TOWER_NO_MINION_DISCOUNT,
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
