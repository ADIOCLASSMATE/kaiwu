#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

简化版稠密奖励管理器 —— 面向 common_ai（只会走路）训练。

设计目标：让模型学会基础玩法 —— 补刀/刷钱、前压推塔、存活、赢游戏。
与 competitive 版（reward_process_competitive.py）的关键区别：

  1. 去掉零和设计：common_ai 不做任何操作，enemy 侧值几乎恒为常数，
     零和帧差退化为纯主英雄帧差。直接用绝对增量。
  2. 减少子项：去掉 ep_rate（蓝量管理）、last_hit（微操补刀）、
     idle_penalty（面对 passive 对手不需要复杂挂机检测）、
     out_of_range_penalty（distance shaping 是 fine-tuning 工具）。
  3. death 权重 >> kill 权重：面对 walking AI 杀人无风险，
     重点是教会模型"冲塔会死、死了推不了塔"。
  4. 推塔是核心目标：enemy_tower_hp 权重最高(10)，win 终局奖励(20)。
  5. 时间衰减大幅放宽：TIME_SCALE_ARG=20000，末期仍有 ~75% reward
     （原 5000 下末期仅 ~13%），给学习阶段更多探索时间。

子项一览（8 项）：

  绝对值帧差：enemy_tower_hp / own_tower_hp / money_gain / exp_gain / kill / death
  绝对值站位：forward（简化为纯位置比例 [0,1]）
  终局事件：win（敌方外塔被摧毁的那一帧 = 1.0）
"""

import math
from agent_diy.conf.conf import GameConfig

TOWER_SUBTYPE = 21
SENTINEL = 99999


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight


def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self._first_frame = True

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(self.m_reward_value)
        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.75, 1.0 * frame_no / self.time_scale_arg)
        return self.m_reward_value

    # ---- 工具 ----
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

    @staticmethod
    def _hp_ratio(unit):
        if unit is None:
            return 0.0
        mh = unit.get("max_hp", 0) or 1
        return max(0.0, min(1.0, unit.get("hp", 0) / mh))

    @staticmethod
    def _is_sentinel(loc):
        return abs(loc.get("x", 0)) >= SENTINEL or abs(loc.get("z", 0)) >= SENTINEL

    # ---- 每帧更新各子项的 cur_frame_value ----
    def set_cur_frame_vec(self, frame_data):
        """仅追踪主英雄侧绝对值，不做主-敌零和相减。"""
        main_hero, main_tower = self._get_camp_units(frame_data, self.main_hero_camp)
        enemy_camp = 2 if self.main_hero_camp == 1 else 1
        _, enemy_tower = self._get_camp_units(frame_data, enemy_camp)

        for reward_name, rs in self.m_calc_frame_map.items():
            rs.last_frame_value = rs.cur_frame_value

            if reward_name == "enemy_tower_hp":
                rs.cur_frame_value = self._hp_ratio(enemy_tower)
            elif reward_name == "own_tower_hp":
                rs.cur_frame_value = self._hp_ratio(main_tower)
            elif reward_name == "money_gain":
                rs.cur_frame_value = float(main_hero.get("money", 0)) / 1000.0 if main_hero else 0.0
            elif reward_name == "exp_gain":
                rs.cur_frame_value = float(main_hero.get("exp", 0)) / 1000.0 if main_hero else 0.0
            elif reward_name == "kill":
                rs.cur_frame_value = float(main_hero.get("kill_cnt", 0)) if main_hero else 0.0
            elif reward_name == "death":
                rs.cur_frame_value = float(main_hero.get("dead_cnt", 0)) if main_hero else 0.0
            elif reward_name == "forward":
                rs.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "win":
                # win 在 get_reward 中通过检测敌塔 "刚被摧毁" 来设置
                pass

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        """英雄沿兵线的站位比例 [0, 1]。

        0 = 贴着己方外塔，1 = 贴着敌方外塔。
        简化为纯位置比例，不做 hp 乘子（满血才给前压太保守）。
        """
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        if self._is_sentinel(main_hero.get("location", {})):
            return 0.0

        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        own_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])

        dist_hero_to_enemy = math.dist(hero_pos, enemy_pos)
        dist_own_to_enemy = math.dist(own_pos, enemy_pos)
        if dist_own_to_enemy <= 0:
            return 0.0

        forward_raw = 1.0 - dist_hero_to_enemy / dist_own_to_enemy
        return max(0.0, min(1.0, forward_raw))

    def frame_data_process(self, frame_data):
        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                self.main_hero_camp = hero["camp"]

        self.set_cur_frame_vec(frame_data)

        # 首帧同步 last 到 cur，消除 0→真实值 造成的假增量 spike
        if self._first_frame:
            self._first_frame = False
            for rs in self.m_calc_frame_map.values():
                rs.last_frame_value = rs.cur_frame_value

    def get_reward(self, reward_dict):
        reward_dict.clear()
        reward_sum = 0.0

        for reward_name, rs in self.m_calc_frame_map.items():
            if reward_name == "forward":
                # forward 是绝对值（站位比例），不用帧差
                rs.value = rs.cur_frame_value
            elif reward_name == "win":
                # win = 敌方塔「刚被摧毁」的那一帧为 1.0
                # 判据：enemy_tower_hp 上一帧 > 0 且当前帧 == 0
                et = self.m_calc_frame_map["enemy_tower_hp"]
                rs.value = 1.0 if (et.last_frame_value > 0 and et.cur_frame_value <= 0) else 0.0
            else:
                # 绝对值帧差：只追踪主英雄侧的增量
                rs.value = rs.cur_frame_value - rs.last_frame_value

            reward_sum += rs.value * rs.weight
            reward_dict[reward_name] = rs.value

        reward_dict["reward_sum"] = reward_sum
