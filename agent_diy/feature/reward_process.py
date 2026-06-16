#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

目标优先的稠密奖励管理器。

零和子项（主-敌之差的帧间增量）：
  tower_hp_point, hp_point, kill, money, exp
非零和子项（仅主视角）：
  lane_progress（满血后场位置惩罚：泉水强，己方神符后快速衰减，神符到中线极弱）、
  danger_penalty（低血仍处在敌方威胁区）、
  last_hit / kill_monster（dead_action 事件归因）、
  idle_penalty（挂机惩罚：经济/伤害产出停滞且不在回撤/泉水区时累计，权重为负）

判定胜负的塔 = 外塔 sub_type=21（reward 按它跟踪）。二塔(24)/水晶(23) 不参与。
终局胜负奖励独立于 shaping，不做时间衰减。
"""

import math

from agent_diy.conf.conf import FeatureConfig as FC
from agent_diy.conf.conf import GameConfig
from agent_diy.feature.targeting import target_slot_enemy_soldiers, visible_to_camp

TOWER_SUBTYPE = 21
MINION_ACTOR_TYPE = 1
MINION_SUBTYPE = 11
SENTINEL = 99999


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


# ---- 挂机检测参数（从 GameConfig 读取，集中管理）----
IDLE_GRACE_FRAMES = GameConfig.IDLE_GRACE_FRAMES
IDLE_RAMP_FRAMES = GameConfig.IDLE_RAMP_FRAMES
IDLE_RETREAT_RATIO = GameConfig.IDLE_RETREAT_RATIO
IDLE_MAX_VALUE = GameConfig.IDLE_MAX_VALUE

# 升到下一级所需经验。累计后可消除升级时 exp 字段清零造成的负奖励。
EXP_TO_NEXT_LEVEL = {
    1: 160,
    2: 298,
    3: 446,
    4: 524,
    5: 613,
    6: 713,
    7: 825,
    8: 950,
    9: 1088,
    10: 1240,
    11: 1406,
    12: 1585,
    13: 1778,
    14: 1984,
}


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        # 挂机检测：跟踪产出类累计值的上帧快照。
        self._last_money_cnt = 0.0
        self._last_hurt_to_hero = 0.0
        self._last_pos = None
        self._inactive_frames = 0
        self._first_frame = True
        # 距离整形：当前帧 action 的越程攻击惩罚，由 workflow 在 predict 后注入。
        self._distance_penalty = 0.0
        self._last_hit_event = 0.0
        self._monster_event = 0.0
        self._terminal_applied = False
        self._lane_cake_anchor_by_camp = {}

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
        return dict(self.m_reward_value)

    def out_of_range_penalty(self, action, decided_frame_state):
        """越程攻击惩罚（distance shaping）。

        action = [button, move_x, move_z, skill_x, skill_z, target]
        decided_frame_state = 做出该 action 时所基于的 frame_state（即上一帧）。
        返回 <= 0 的标量；只在「攻击类 button + target 指向可解析实体 + 目标在
        主英雄攻击范围外」时为负，其余为 0。权重设 0（OUT_OF_RANGE_PENALTY=0）时恒 0。
        """
        w = GameConfig.OUT_OF_RANGE_PENALTY
        if w <= 0 or action is None or len(action) < 6:
            return 0.0
        button, target = action[0], action[5]
        if button not in GameConfig.ATTACK_BUTTONS:
            return 0.0
        # target 槽：0 None / 1 EnemyHero / 2 Self / 3-6 Soldier / 7 Tower / 8 Monster
        # 对可解析攻击目标做越程判定；Tower 保持不罚，避免推塔策略被该 shaping 干扰。
        if target == 1:
            tpos = self._enemy_hero_pos(decided_frame_state)
        elif 3 <= target <= 6:
            tpos = self._nth_enemy_minion_pos(decided_frame_state, target - 3)
        elif target == 8:
            tpos = self._nearest_monster_pos(decided_frame_state)
        else:
            return 0.0
        if tpos is None:
            return 0.0
        mh = self._main_hero(decided_frame_state)
        if mh is None:
            return 0.0
        mpos = (mh["location"]["x"], mh["location"]["z"])
        atk_range = float(mh.get("attack_range", 0) or 0)
        if atk_range <= 0:
            return 0.0
        dist = math.hypot(tpos[0] - mpos[0], tpos[1] - mpos[1])
        return -w if dist > atk_range else 0.0

    def set_distance_penalty(self, action, decided_frame_state):
        """由 workflow 在 predict/exploit 后调用，注入当前 action 的距离惩罚。"""
        self._distance_penalty = self.out_of_range_penalty(action, decided_frame_state)

    # ---- distance shaping 辅助：定位主英雄 / 敌英雄 / 第 k 近敌方小兵 / 最近野怪 ----
    def _main_hero(self, fs):
        for h in fs.get("hero_states", []):
            if h.get("runtime_id") == self.main_hero_player_id:
                return h if not self._is_sentinel(h.get("location", {})) else None
        return None

    def _enemy_hero_pos(self, fs):
        for h in fs.get("hero_states", []):
            if h.get("runtime_id") != self.main_hero_player_id:
                loc = h.get("location", {})
                if self._is_sentinel(loc) or h.get("hp", 0) <= 0:
                    return None
                return (loc["x"], loc["z"])
        return None

    def _nth_enemy_minion_pos(self, fs, k):
        mh = self._main_hero(fs)
        if mh is None:
            return None
        mpos = (mh["location"]["x"], mh["location"]["z"])
        ordered = target_slot_enemy_soldiers(
            fs.get("npc_states", []),
            mpos,
            mh["camp"],
            FC.N_MINION_PER_CAMP,
            visible_fn=visible_to_camp,
        )
        return ordered[k]["pos"] if k < len(ordered) else None

    def _nearest_monster_pos(self, fs):
        mh = self._main_hero(fs)
        if mh is None:
            return None
        mpos = (mh["location"]["x"], mh["location"]["z"])
        cand = []
        for npc in fs.get("npc_states", []):
            if not self._is_resource_monster(npc):
                continue
            loc = npc.get("location", {})
            if self._is_sentinel(loc):
                continue
            p = (loc["x"], loc["z"])
            cand.append((math.hypot(p[0] - mpos[0], p[1] - mpos[1]), p))
        cand.sort(key=lambda t: t[0])
        return cand[0][1] if cand else None

    @staticmethod
    def _is_resource_monster(npc):
        return (npc.get("actor_type") == MINION_ACTOR_TYPE
                and npc.get("sub_type") != MINION_SUBTYPE
                and npc.get("kill_income", 0) > 0
                and npc.get("hp", 0) > 0)

    @staticmethod
    def _is_sentinel(loc):
        return abs(loc.get("x", 0)) >= SENTINEL or abs(loc.get("z", 0)) >= SENTINEL

    @staticmethod
    def _get_main_hero_pos(frame_data, main_camp):
        """获取主英雄的 (x, z) 位置，失败返回 None。"""
        for h in frame_data.get("hero_states", []):
            if h.get("camp") == main_camp and h.get("hp", 0) > 0:
                loc = h.get("location", {})
                if abs(loc.get("x", 0)) < SENTINEL and abs(loc.get("z", 0)) < SENTINEL:
                    return (loc["x"], loc["z"])
        return None

    # ---- 工具 ----
    @staticmethod
    def _hp_ratio(hero):
        if hero is None:
            return 0.0
        mh = hero.get("max_hp", 0) or 1
        ratio = max(0.0, min(1.0, hero.get("hp", 0) / mh))
        return math.sqrt(ratio)

    @staticmethod
    def _raw_hp_ratio(hero):
        if hero is None:
            return 0.0
        mh = hero.get("max_hp", 0) or 1
        return max(0.0, min(1.0, hero.get("hp", 0) / mh))

    @staticmethod
    def _total_exp(hero):
        if hero is None:
            return 0.0
        level = max(1, int(hero.get("level", 1) or 1))
        total = sum(EXP_TO_NEXT_LEVEL.get(i, 0) for i in range(1, level))
        return total + float(hero.get("exp", 0) or 0)

    def _get_camp_units(self, frame_data, camp):
        hero = None
        for h in frame_data["hero_states"]:
            if h["camp"] == camp:
                hero = h
        tower = None
        for npc in frame_data["npc_states"]:
            if npc.get("camp") == camp and npc.get("sub_type") == TOWER_SUBTYPE:
                tower = npc
        return hero, tower

    def set_cur_calc_frame_vec(self, calc_map, frame_data, camp):
        hero, tower = self._get_camp_units(frame_data, camp)
        enemy_hero = next(
            (item for item in frame_data.get("hero_states", [])
             if item.get("camp") != camp),
            None,
        )
        for reward_name, rs in calc_map.items():
            rs.last_frame_value = rs.cur_frame_value
            if reward_name == "tower_hp_point":
                if tower is not None and tower.get("max_hp", 0) > 0:
                    tower_hp = 1.0 * tower["hp"] / tower["max_hp"]
                    attacking_enemy = (
                        enemy_hero is not None
                        and tower.get("attack_target") == enemy_hero.get("runtime_id")
                    )
                    rs.cur_frame_value = (tower_hp, attacking_enemy)
                else:
                    previous = rs.last_frame_value
                    if not isinstance(previous, tuple):
                        previous = (0.0, False)
                    rs.cur_frame_value = previous
            elif reward_name == "lane_progress":
                main_hero, main_tower = self._get_camp_units(frame_data, camp)
                enemy_camp = 2 if camp == 1 else 1
                _, enemy_tower = self._get_camp_units(frame_data, enemy_camp)
                rs.cur_frame_value = self.calculate_lane_guidance_penalty(
                    frame_data,
                    camp,
                    main_hero,
                    main_tower,
                    enemy_tower,
                )
            elif reward_name == "hp_point":
                rs.cur_frame_value = self._hp_ratio(hero)
            elif reward_name == "danger_penalty":
                main_hero, _ = self._get_camp_units(frame_data, camp)
                enemy_camp = 2 if camp == 1 else 1
                enemy_hero, enemy_tower = self._get_camp_units(frame_data, enemy_camp)
                rs.cur_frame_value = self.calculate_danger_penalty(
                    frame_data,
                    main_hero,
                    enemy_hero,
                    enemy_tower,
                    camp,
                )
            elif reward_name == "kill":
                rs.cur_frame_value = float(hero.get("kill_cnt", 0)) if hero else 0.0
            elif reward_name == "money":
                rs.cur_frame_value = float(hero.get("money_cnt", 0)) / 1000.0 if hero else 0.0
            elif reward_name == "exp":
                rs.cur_frame_value = self._total_exp(hero) / 1000.0
            # 事件奖励与 idle_penalty 在 get_reward 中直接计算。

    def calculate_lane_guidance_penalty(
        self,
        frame_data,
        camp,
        main_hero,
        main_tower,
        enemy_tower,
    ):
        """Return full-HP backfield penalty strength in [0, 1]."""
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        if self._is_sentinel(main_hero.get("location", {})):
            return 0.0
        if self._raw_hp_ratio(main_hero) < GameConfig.LANE_GUIDANCE_HP_THRESHOLD:
            return 0.0

        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        own_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_t = self._lane_projection_t(hero_pos, own_pos, enemy_pos)
        if hero_t is None:
            return 0.0
        if hero_t >= 0.5:
            return 0.0

        cake_t = self._own_cake_projection_t(frame_data, camp, own_pos, enemy_pos)
        if cake_t is None:
            cake_t = GameConfig.LANE_GUIDANCE_FALLBACK_CAKE_T
        cake_t = min(cake_t, 0.49)

        epsilon = GameConfig.LANE_GUIDANCE_EPSILON
        if hero_t <= cake_t:
            fountain_t = min(GameConfig.LANE_GUIDANCE_FOUNTAIN_T, cake_t - 1e-3)
            if hero_t <= fountain_t:
                return 1.0
            progress = (hero_t - fountain_t) / (cake_t - fountain_t)
            progress = max(0.0, min(1.0, progress))
            return epsilon + (1.0 - epsilon) * math.pow(
                1.0 - progress,
                GameConfig.LANE_GUIDANCE_BACK_EXPONENT,
            )

        center_span = 0.5 - cake_t
        if center_span <= 0:
            return 0.0
        progress_to_center = (hero_t - cake_t) / center_span
        progress_to_center = max(0.0, min(1.0, progress_to_center))
        return epsilon * (1.0 - progress_to_center)

    def _own_cake_projection_t(self, frame_data, camp, own_pos, enemy_pos):
        best = None
        for cake in frame_data.get("cakes", []) or []:
            loc = ((cake.get("collider", {}) or {}).get("location", {}) or {})
            if not loc or self._is_sentinel(loc):
                continue
            pos = (loc["x"], loc["z"])
            t = self._lane_projection_t(pos, own_pos, enemy_pos)
            if t is None:
                continue
            if t < 0.0 and (best is None or t > best):
                best = t
        if best is not None:
            self._lane_cake_anchor_by_camp[camp] = best
            return best
        return self._lane_cake_anchor_by_camp.get(camp)

    @staticmethod
    def _lane_projection_t(pos, own_pos, enemy_pos):
        dx = enemy_pos[0] - own_pos[0]
        dz = enemy_pos[1] - own_pos[1]
        denom = dx * dx + dz * dz
        if denom <= 0:
            return None
        return ((pos[0] - own_pos[0]) * dx + (pos[1] - own_pos[1]) * dz) / denom

    def calculate_danger_penalty(
        self,
        frame_data,
        main_hero,
        enemy_hero,
        enemy_tower,
        main_camp,
    ):
        if main_hero is None or self._is_sentinel(main_hero.get("location", {})):
            return 0.0
        hp_ratio = self._raw_hp_ratio(main_hero)
        threshold = GameConfig.DANGER_HP_THRESHOLD
        if hp_ratio >= threshold:
            return 0.0

        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        threat = 0.0
        if (
            enemy_hero is not None
            and enemy_hero.get("hp", 0) > 0
            and visible_to_camp(enemy_hero, main_camp)
        ):
            loc = enemy_hero.get("location", {})
            if not self._is_sentinel(loc):
                enemy_pos = (loc["x"], loc["z"])
                enemy_range = float(enemy_hero.get("attack_range", 0) or 0)
                threat_range = max(enemy_range * GameConfig.DANGER_RANGE_MULT, 3500.0)
                dist = math.dist(hero_pos, enemy_pos)
                if dist <= threat_range:
                    hp_threat = self._raw_hp_ratio(enemy_hero)
                    threat = max(threat, (1.0 - (dist / threat_range) * 0.75) * hp_threat)

        if enemy_tower is not None and enemy_tower.get("hp", 0) > 0:
            tower_loc = enemy_tower.get("location", {})
            if not self._is_sentinel(tower_loc):
                tower_pos = (tower_loc["x"], tower_loc["z"])
                tower_range = float(enemy_tower.get("attack_range", 0) or 0)
                dist = math.dist(hero_pos, tower_pos)
                tower_targets_me = enemy_tower.get("attack_target") == main_hero.get("runtime_id")
                unsafe_tower_range = tower_range > 0 and dist <= tower_range
                if tower_targets_me or (
                    unsafe_tower_range
                    and not self._has_minion_pressure_on_enemy_tower(frame_data, main_camp)
                ):
                    threat = max(threat, 1.0)

        if threat <= 0:
            return 0.0
        low_hp_severity = (threshold - hp_ratio) / threshold
        return low_hp_severity * threat * GameConfig.DANGER_FRAME_SCALE

    def _has_minion_pressure_on_enemy_tower(self, frame_data, main_camp):
        enemy_camp = 2 if main_camp == 1 else 1
        _, enemy_tower = self._get_camp_units(frame_data, enemy_camp)
        if enemy_tower is None or enemy_tower.get("hp", 0) <= 0:
            return False
        tower_loc = enemy_tower.get("location", {})
        if self._is_sentinel(tower_loc):
            return False
        tower_pos = (tower_loc["x"], tower_loc["z"])
        tower_target = enemy_tower.get("attack_target")
        radius = GameConfig.TOWER_PUSH_MINION_RADIUS
        for npc in frame_data.get("npc_states", []):
            if not (
                npc.get("actor_type") == MINION_ACTOR_TYPE
                and npc.get("sub_type") == MINION_SUBTYPE
                and npc.get("camp") == main_camp
                and npc.get("hp", 0) > 0
            ):
                continue
            if tower_target and tower_target == npc.get("runtime_id"):
                return True
            loc = npc.get("location", {})
            if self._is_sentinel(loc):
                continue
            if math.dist((loc["x"], loc["z"]), tower_pos) <= radius:
                return True
        return False

    def _main_hero_in_enemy_tower_range(self, frame_data, main_camp):
        main_hero, _ = self._get_camp_units(frame_data, main_camp)
        enemy_camp = 2 if main_camp == 1 else 1
        _, enemy_tower = self._get_camp_units(frame_data, enemy_camp)
        if main_hero is None or enemy_tower is None:
            return False
        if self._is_sentinel(main_hero.get("location", {})) or self._is_sentinel(
            enemy_tower.get("location", {})
        ):
            return False
        tower_range = float(enemy_tower.get("attack_range", 0) or 0)
        if tower_range <= 0:
            return False
        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        return math.dist(hero_pos, tower_pos) <= tower_range

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)
        self._last_hit_event, self._monster_event = self._objective_events(
            frame_data,
            main_camp,
            enemy_camp,
        )
        # 首帧同步 last 到 cur，消除 0→真实值 造成的假增量 spike。
        if self._first_frame:
            self._first_frame = False
            for calc_map in (self.m_main_calc_frame_map, self.m_enemy_calc_frame_map):
                for rs in calc_map.values():
                    rs.last_frame_value = rs.cur_frame_value
            self._last_money_cnt = float(
                next((h.get("money_cnt", 0) for h in frame_data.get("hero_states", [])
                      if h.get("camp") == main_camp and h.get("hp", 0) > 0), 0)
            )
            self._last_hurt_to_hero = float(
                next((h.get("total_hurt_to_hero", 0) for h in frame_data.get("hero_states", [])
                      if h.get("camp") == main_camp and h.get("hp", 0) > 0), 0)
            )
            self._last_pos = self._get_main_hero_pos(frame_data, main_camp)
        self._update_inactive(frame_data, main_camp)

    def _objective_events(self, frame_data, main_camp, enemy_camp):
        """Return main-perspective last-hit and neutral-monster event rewards."""
        heroes = {
            hero.get("camp"): hero.get("runtime_id")
            for hero in frame_data.get("hero_states", [])
        }
        main_id = heroes.get(main_camp)
        enemy_id = heroes.get(enemy_camp)
        last_hit = 0.0
        monster = 0.0
        dead_actions = (
            (frame_data.get("frame_action") or {}).get("dead_action") or []
        )
        for action in dead_actions:
            death = action.get("death") or {}
            killer = action.get("killer") or {}
            killer_id = killer.get("runtime_id")
            subtype = death.get("sub_type")
            death_camp = death.get("camp")

            is_soldier = subtype in (MINION_SUBTYPE, "ACTOR_SUB_SOLDIER")
            if is_soldier:
                if death_camp == enemy_camp and killer_id == main_id:
                    last_hit += 1.0
                elif death_camp == main_camp and killer_id == enemy_id:
                    last_hit -= 1.0
                continue

            is_neutral = death_camp in (0, "PLAYERCAMP_MID")
            if is_neutral and killer_id == main_id:
                monster += 1.0
            elif is_neutral and killer_id == enemy_id:
                monster -= 1.0
        return last_hit, monster

    def _in_retreat_zone(self, frame_data, hero, main_camp):
        """英雄是否在己方塔后方的安全回撤/泉水区（回血/回城）。"""
        hloc = hero.get("location", {})
        if self._is_sentinel(hloc):
            return True
        _, own_tower = self._get_camp_units(frame_data, main_camp)
        enemy_camp = 2 if main_camp == 1 else 1
        _, enemy_tower = self._get_camp_units(frame_data, enemy_camp)
        if own_tower is None or enemy_tower is None:
            return False
        otl = own_tower.get("location", {})
        etl = enemy_tower.get("location", {})
        if self._is_sentinel(otl) or self._is_sentinel(etl):
            return False
        hpos = (hloc["x"], hloc["z"])
        opos = (otl["x"], otl["z"])
        epos = (etl["x"], etl["z"])
        dist_hero_to_enemy = math.hypot(hpos[0] - epos[0], hpos[1] - epos[1])
        dist_own_to_enemy = math.hypot(opos[0] - epos[0], opos[1] - epos[1])
        if dist_own_to_enemy <= 0:
            return False
        return dist_hero_to_enemy > dist_own_to_enemy * IDLE_RETREAT_RATIO

    def _update_inactive(self, frame_data, main_camp):
        """挂机计数更新（位置 + 产出双重判据 + 回撤/泉水冻结）。"""
        hero = None
        cur_pos = None
        for h in frame_data.get("hero_states", []):
            if h.get("camp") == main_camp and h.get("hp", 0) > 0:
                loc = h.get("location", {})
                if not self._is_sentinel(loc):
                    hero = h
                    cur_pos = (loc["x"], loc["z"])
                    break

        if hero is None:
            self._inactive_frames += 1
            return

        pos_moved = False
        if self._last_pos is not None and cur_pos is not None:
            dist = math.hypot(cur_pos[0] - self._last_pos[0], cur_pos[1] - self._last_pos[1])
            pos_moved = (dist > GameConfig.IDLE_POS_DELTA_THRESHOLD)
        self._last_pos = cur_pos

        if pos_moved:
            self._inactive_frames = 0
            return

        cur_money = float(hero.get("money_cnt", 0))
        cur_hurt = float(hero.get("total_hurt_to_hero", 0))
        money_delta = cur_money - self._last_money_cnt
        hurt_delta = cur_hurt - self._last_hurt_to_hero
        self._last_money_cnt = cur_money
        self._last_hurt_to_hero = cur_hurt

        zero_output = (money_delta <= 0 and hurt_delta <= 0)
        if not zero_output:
            self._inactive_frames = 0
            return
        if self._in_retreat_zone(frame_data, hero, main_camp):
            return
        self._inactive_frames += 1

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum = 0.0
        for reward_name, rs in self.m_cur_calc_frame_map.items():
            main = self.m_main_calc_frame_map[reward_name]
            enemy = self.m_enemy_calc_frame_map[reward_name]
            if reward_name == "tower_hp_point":
                main_cur_hp, main_attacking_enemy = main.cur_frame_value
                main_last_hp, _ = main.last_frame_value
                enemy_cur_hp, enemy_attacking_main = enemy.cur_frame_value
                enemy_last_hp, _ = enemy.last_frame_value
                current_advantage = main_cur_hp - enemy_cur_hp
                last_advantage = main_last_hp - enemy_last_hp
                rs.value = current_advantage - last_advantage
                if rs.value > 0:
                    if enemy_attacking_main:
                        rs.value *= GameConfig.TOWER_DIVE_DISCOUNT
                    elif (
                        self._main_hero_in_enemy_tower_range(frame_data, self.main_hero_camp)
                        and not self._has_minion_pressure_on_enemy_tower(
                            frame_data,
                            self.main_hero_camp,
                        )
                    ):
                        rs.value *= GameConfig.TOWER_NO_MINION_DISCOUNT
                elif rs.value < 0 and main_attacking_enemy:
                    rs.value *= GameConfig.TOWER_DIVE_DISCOUNT
            elif reward_name == "lane_progress":
                rs.value = main.cur_frame_value
            elif reward_name == "danger_penalty":
                rs.value = main.cur_frame_value
            elif reward_name == "last_hit":
                rs.value = self._last_hit_event
            elif reward_name == "kill_monster":
                rs.value = self._monster_event
            elif reward_name == "idle_penalty":
                if self._inactive_frames > IDLE_GRACE_FRAMES:
                    ramp = min(
                        (self._inactive_frames - IDLE_GRACE_FRAMES) / IDLE_RAMP_FRAMES,
                        IDLE_MAX_VALUE,
                    )
                    rs.value = ramp * GameConfig.IDLE_FRAME_SCALE
                else:
                    rs.value = 0.0
            else:
                # 通用零和子项：主-敌之差的帧间增量。
                rs.cur_frame_value = main.cur_frame_value - enemy.cur_frame_value
                rs.last_frame_value = main.last_frame_value - enemy.last_frame_value
                rs.value = rs.cur_frame_value - rs.last_frame_value
            reward_sum += rs.value * rs.weight
            reward_dict[reward_name] = rs.value
        reward_dict["distance_penalty"] = self._distance_penalty
        reward_sum += self._distance_penalty
        self._distance_penalty = 0.0
        reward_dict["reward_sum"] = reward_sum

    def apply_terminal_outcome(self, reward_dict, frame_data, win=None):
        """Add a one-shot terminal reward and return its weighted contribution."""
        if self._terminal_applied:
            return 0.0
        self._terminal_applied = True

        outcome = self._terminal_outcome(frame_data, win)
        bonus = outcome * GameConfig.TERMINAL_WIN_REWARD
        reward_dict["terminal"] = outcome
        reward_dict["reward_sum"] = reward_dict.get("reward_sum", 0.0) + bonus
        return bonus

    def _terminal_outcome(self, frame_data, win):
        if win == 1:
            return 1.0
        if win == 0:
            return -1.0

        main_camp = self.main_hero_camp
        if main_camp not in (1, 2):
            main_hero = next(
                (hero for hero in frame_data.get("hero_states", [])
                 if hero.get("runtime_id") == self.main_hero_player_id),
                None,
            )
            main_camp = main_hero.get("camp") if main_hero else -1

        own_alive = False
        enemy_alive = False
        for npc in frame_data.get("npc_states", []):
            if npc.get("sub_type") != TOWER_SUBTYPE:
                continue
            alive = npc.get("hp", 0) > 0
            if npc.get("camp") == main_camp:
                own_alive = own_alive or alive
            else:
                enemy_alive = enemy_alive or alive
        if own_alive and not enemy_alive:
            return 1.0
        if enemy_alive and not own_alive:
            return -1.0
        return 0.0
