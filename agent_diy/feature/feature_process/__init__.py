#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

FeatureProcess: 把环境 observation 转成 FeatureConfig.FEATURE_DIM 维特征向量。

输出布局（顺序即 conf.FeatureConfig.TOKEN_SEGMENTS + 全局段）：
  main_hero(HERO_DIM) | enemy_hero(HERO_DIM)
  | own_structures x3 (STRUCT_DIM) | enemy_structures x3 (STRUCT_DIM)
  | own_minions x4 (MINION_DIM)    | enemy_minions x4 (MINION_DIM)
  | global(GLOBAL_DIM)

关键约定：
  - 每个 token 第 0 维是 present（1=有效，0=不存在/不可见/死亡），供模型造 mask。
  - 视角统一到「主视角」：camp==2 时把坐标 x、z 取反，镜像到 camp1 视角，
    使 camp1 / camp2 的输出对称。
  - 哨兵处理：|x|>=SENTINEL 或 |z|>=SENTINEL → present=0、整 token 置零、
    不参与任何距离/位置计算。NPC 还要判 camp_visible[main_camp-1]。
  - 数值经软饱和 value/(value+K) 或固定尺度归一，避免依赖精确上限。
  - 相对位移用 ENGAGE_SCALE 后 clip 到 [-1,1] 再线性映射到 [0,1]；
    绝对位置用 MAP_SCALE；整体距离用 DIST_SCALE。
"""

from agent_diy.feature.feature_process.builder import FeatureBuilder


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.builder = FeatureBuilder(camp)

    def reset(self, camp):
        self.camp = camp
        self.builder = FeatureBuilder(camp)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        return self.builder.build(frame_state)
