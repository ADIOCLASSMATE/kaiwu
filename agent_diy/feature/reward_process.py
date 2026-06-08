#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

11 子项稠密奖励管理器。

零和子项（主-敌之差的帧间增量）：
  tower_hp_point, enemy_tower_hp, hp_point, ep_rate, kill, death, money, exp
非零和子项（仅主视角）：
  forward（向敌方塔站位，HP 越高权重越大；处于敌塔范围内不给，防贴脸白嫖）、
  last_hit（补刀收益差分）、
  idle_penalty（挂机惩罚：经济/伤害产出停滞且不在回撤/泉水区时累计，权重为负）

判定胜负的塔 = 外塔 sub_type=21（reward 按它跟踪）。二塔(24)/水晶(23) 不参与。

注意（设计权衡，非 bug）：
  补一个兵会同时抬升 money_cnt（→ last_hit）与 money（→ money 子项），即补刀
  被 last_hit 与 money 各计一次。这是有意放大补刀吸引力；若发现过度刷线不打架，
  优先调小 last_hit 权重。
"""

import math
from agent_diy.conf.conf import GameConfig
from agent_diy.conf.conf import FeatureConfig as FC
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


# 非零和子项：直接用主视角值，不做主-敌相减。
NON_ZERO_SUM = {"forward", "last_hit", "idle_penalty"}

# ---- 挂机检测参数（从 GameConfig 读取，集中管理）----
IDLE_GRACE_FRAMES = GameConfig.IDLE_GRACE_FRAMES
IDLE_RAMP_FRAMES = GameConfig.IDLE_RAMP_FRAMES
IDLE_RETREAT_RATIO = GameConfig.IDLE_RETREAT_RATIO
IDLE_MAX_VALUE = GameConfig.IDLE_MAX_VALUE


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        # 挂机检测：跟踪产出类累计值的上帧快照
        self._last_money_cnt = 0.0
        self._last_hurt_to_hero = 0.0
        self._inactive_frames = 0

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
        return self.m_reward_value

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

    # ---- 工具 ----
    @staticmethod
    def _hp_ratio(hero):
        if hero is None:
            return 0.0
        mh = hero.get("max_hp", 0) or 1
        return max(0.0, min(1.0, hero.get("hp", 0) / mh))

    @staticmethod
    def _ep_ratio(hero):
        if hero is None:
            return 0.0
        me = hero.get("max_ep", 0) or 1
        return max(0.0, min(1.0, hero.get("ep", 0) / me))

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
        for reward_name, rs in calc_map.items():
            rs.last_frame_value = rs.cur_frame_value
            if reward_name in ("tower_hp_point", "enemy_tower_hp"):
                if tower is not None and tower.get("max_hp", 0) > 0:
                    rs.cur_frame_value = 1.0 * tower["hp"] / tower["max_hp"]
                else:
                    rs.cur_frame_value = rs.last_frame_value
            elif reward_name == "hp_point":
                rs.cur_frame_value = self._hp_ratio(hero)
            elif reward_name == "ep_rate":
                rs.cur_frame_value = self._ep_ratio(hero)
            elif reward_name == "kill":
                rs.cur_frame_value = float(hero.get("kill_cnt", 0)) if hero else 0.0
            elif reward_name == "death":
                rs.cur_frame_value = float(hero.get("dead_cnt", 0)) if hero else 0.0
            elif reward_name == "money":
                rs.cur_frame_value = float(hero.get("money", 0)) / 1000.0 if hero else 0.0
            elif reward_name == "exp":
                rs.cur_frame_value = float(hero.get("exp", 0)) / 1000.0 if hero else 0.0
            elif reward_name == "forward":
                main_hero, main_tower = self._get_camp_units(frame_data, camp)
                enemy_camp = 2 if camp == 1 else 1
                _, enemy_tower = self._get_camp_units(frame_data, enemy_camp)
                rs.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "last_hit":
                # 近似：英雄 money_cnt（补刀累计）作累计指标，get_reward 里差分得「最近收益」。
                rs.cur_frame_value = float(hero.get("money_cnt", 0)) / 100.0 if hero else 0.0
            # idle_penalty 不在此设置 cur_frame_value（其值在 get_reward 里据 inactive 计数算）

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        """英雄沿兵线的站位比例，HP 越高有效权重越大。

        forward_raw ∈ [-0.2, 1.0]：0=在己方塔位，1=在敌方塔位。
        乘以 hp_ratio：满血时鼓励前压，残血时自然减弱（撤退合理）。
        反 hack：若处于敌方外塔攻击范围内，则不给前压奖励——否则满血贴脸敌塔
        反复横跳即可零风险白嫖该奖励。
        """
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        if abs(main_hero["location"]["x"]) >= SENTINEL:
            return 0.0

        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        own_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])

        # 反 hack：在敌方外塔攻击范围内不发前压奖励
        if GameConfig.FORWARD_NO_REWARD_IN_ENEMY_TOWER:
            etr = float(enemy_tower.get("attack_range", 0) or 0)
            if etr > 0 and math.dist(hero_pos, enemy_pos) <= etr:
                return 0.0

        dist_hero_to_enemy = math.dist(hero_pos, enemy_pos)
        dist_own_to_enemy = math.dist(own_pos, enemy_pos)
        if dist_own_to_enemy <= 0:
            return 0.0

        # 0 = 贴着己方塔，1 = 贴着敌方塔
        forward_raw = 1.0 - dist_hero_to_enemy / dist_own_to_enemy
        # 轻量 clip：防止缩在泉水时产生极端负值
        if forward_raw < -0.2:
            forward_raw = -0.2
        if forward_raw > 1.0:
            forward_raw = 1.0

        hp = main_hero.get("hp", 0)
        max_hp = main_hero.get("max_hp", 1) or 1
        hp_ratio = max(0.0, min(1.0, hp / max_hp))

        return forward_raw * hp_ratio

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
        # 挂机检测：基于经济/伤害产出（非位置）+ 回撤区豁免
        self._update_inactive(frame_data, main_camp)

    def _in_retreat_zone(self, frame_data, hero, main_camp):
        """英雄是否在己方塔后方的安全回撤/泉水区（回血/回城）。

        几何判据：到敌方外塔的距离 > 己方外塔到敌方外塔距离 × IDLE_RETREAT_RATIO。
        即英雄比自家塔还靠后（朝己方泉水方向）。这天然涵盖「在泉水回血」与
        「在安全后方回城」两种不该被罚的场景，且不依赖未公开的 behav_mode 编码。
        """
        hloc = hero.get("location", {})
        if self._is_sentinel(hloc):
            return True   # 取不到位置时保守不罚
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
        """挂机计数更新（纯产出停滞 + 回撤/泉水冻结）：
          - 真有产出（任一增量>0）→ 清零；
          - 产出停滞 且 在回撤区（泉水回血/后方回城）→ 冻结（不增不减）；
          - 产出停滞 且 不在回撤区 → 累加。
        死亡（hero=None）也视为累加，鼓励尽快复活投入战斗。
        冻结而非清零是关键：否则 agent 可踏进回撤区 1 帧归零、再出来发呆，
        反复横跳永远凑不满宽限期，惩罚被绕过（与兵线原地抽搐同类漏洞）。
        """
        hero = None
        for h in frame_data.get("hero_states", []):
            if h.get("camp") == main_camp and h.get("hp", 0) > 0:
                loc = h.get("location", {})
                if not self._is_sentinel(loc):
                    hero = h
                    break

        if hero is None:
            self._inactive_frames += 1
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
            return    # 冻结，不增不减
        self._inactive_frames += 1

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum = 0.0
        for reward_name, rs in self.m_cur_calc_frame_map.items():
            main = self.m_main_calc_frame_map[reward_name]
            enemy = self.m_enemy_calc_frame_map[reward_name]
            if reward_name == "tower_hp_point":
                # 己方塔血量比例（主-敌零和），帧间增量
                rs.cur_frame_value = main.cur_frame_value - enemy.cur_frame_value
                rs.last_frame_value = main.last_frame_value - enemy.last_frame_value
                rs.value = rs.cur_frame_value - rs.last_frame_value
            elif reward_name == "enemy_tower_hp":
                # 敌方塔血量下降 = 正向激励：-(敌方塔比例增量)
                rs.value = -(enemy.cur_frame_value - enemy.last_frame_value)
            elif reward_name in NON_ZERO_SUM:
                if reward_name == "forward":
                    rs.value = main.cur_frame_value
                elif reward_name == "idle_penalty":
                    # 产出停滞且不在回撤区的惩罚：宽限期后线性爬升，IDLE_MAX_VALUE 封顶。
                    if self._inactive_frames > IDLE_GRACE_FRAMES:
                        rs.value = min(
                            (self._inactive_frames - IDLE_GRACE_FRAMES) / IDLE_RAMP_FRAMES,
                            IDLE_MAX_VALUE)
                    else:
                        rs.value = 0.0
                else:  # last_hit: 主视角差分
                    rs.value = max(0.0, main.cur_frame_value - main.last_frame_value)
            else:
                # 通用零和子项：主-敌之差的帧间增量
                rs.cur_frame_value = main.cur_frame_value - enemy.cur_frame_value
                rs.last_frame_value = main.last_frame_value - enemy.last_frame_value
                rs.value = rs.cur_frame_value - rs.last_frame_value
            reward_sum += rs.value * rs.weight
            reward_dict[reward_name] = rs.value
        reward_dict["reward_sum"] = reward_sum
