#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

10 子项稠密奖励管理器。

零和子项（主-敌之差的帧间增量）：
  tower_hp_point, enemy_tower_hp, hp_point, ep_rate, kill, death, money, exp
非零和子项（仅主视角）：
  forward（向敌方塔前进，仅满血时计算）、last_hit（击杀小兵的最后一击近似）

判定胜负的塔 = 外塔 sub_type=21（reward 按它跟踪）。二塔(24)/水晶(23) 不参与。
"""

import math
from agent_diy.conf.conf import GameConfig

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
NON_ZERO_SUM = {"forward", "last_hit"}


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
        return self.m_reward_value

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
                # 近似：本帧己方小兵击杀数（hero money_cnt 增量在 get_reward 里按差分处理）。
                # 直接用英雄 money_cnt 作累计指标，差分得「最近收益」。
                rs.cur_frame_value = float(hero.get("money_cnt", 0)) / 100.0 if hero else 0.0

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        if abs(main_hero["location"]["x"]) >= SENTINEL:
            return 0.0
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        forward_value = 0.0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        mh = main_hero.get("max_hp", 0) or 1
        if main_hero["hp"] / mh > 0.99 and dist_main2emy > 0 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

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
