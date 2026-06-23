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
  | own_tower(STRUCT_DIM)          | enemy_tower(STRUCT_DIM)
  | own_minions x4 (MINION_DIM)    | enemy_minions x4 (MINION_DIM)
  | monster(MONSTER_DIM)           | hero bullets x4 (BULLET_DIM)
  | cakes x2 (CAKE_DIM)
  | global(GLOBAL_DIM)

关键约定：
  - 每个 token 第 0 维是 exists（1=槽位有效，0=padding），供模型造 mask。
  - 视角统一到「主视角」：camp==2 时把坐标 x、z 取反，镜像到 camp1 视角，
    使 camp1 / camp2 的输出对称。
  - 哨兵处理：|x|>=SENTINEL 或 |z|>=SENTINEL 的当前可见 NPC 不入槽；
    英雄/塔保留 last-known 位置特征。NPC 还要判 camp_visible[main_camp-1]。
  - 数值经 log01、比率或固定尺度归一；所有特征约束到 [0, 1]，不依赖
    模型侧 token-level LayerNorm 统一尺度。
  - 相对位移用 ENGAGE_SCALE 后 clip 到 [-1,1] 再线性映射到 [0,1]；
    绝对位置用 MAP_SCALE；整体距离用 DIST_SCALE。
  - Soldier1-4 target 槽：敌方小兵先选最近 4 个，再按 runtime_id 升序入槽。
  - bullet 只编码 hero-sourced projectile；NPC/minion/tower projectile 不入槽。
  - monster 未观测到 attack_target 语义，保持资源/位置/血量特征，不加目标关系。
"""

import numpy as np
from agent_ppo.feature.feature_process.builder import FeatureBuilder
from agent_ppo.conf.conf import FeatureConfig


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.builder = FeatureBuilder(camp)
        self._reset_stats()

    def reset(self, camp):
        self.camp = camp
        self.builder = FeatureBuilder(camp)
        self._reset_stats()

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        feat = self.builder.build(
            frame_state,
            retreat_need_active=bool(observation.get("retreat_need_active", False)),
        )
        self._accumulate(feat)
        return feat

    # ---- feature 健康度统计（全特征覆盖）----
    def _reset_stats(self):
        self._n = 0
        self._nan = 0
        self._inf = 0
        self._neg = 0
        # 按 label 报告各项指标，每个 TOKEN_SEGMENTS 的 type_key 一组 + global 一组
        self._tk = {}
        for key, dim, cnt in FeatureConfig.TOKEN_SEGMENTS:
            self._tk[key] = {
                "cnt": cnt, "dim": dim,
                "exists_sum": 0.0,
                "act_sum": 0.0, "act_sum_sq": 0.0,
                "dim_sum": np.zeros(dim * cnt, dtype=np.float64),
                "dim_sum_sq": np.zeros(dim * cnt, dtype=np.float64),
            }
        gd = FeatureConfig.GLOBAL_DIM
        self._g_sum = 0.0
        self._g_sum_sq = 0.0
        self._g_dim_sum = np.zeros(gd, dtype=np.float64)
        self._g_dim_sum_sq = np.zeros(gd, dtype=np.float64)

    def _accumulate(self, feat):
        arr = np.asarray(feat, dtype=np.float64)
        self._n += 1
        self._nan += int(np.isnan(arr).sum())
        self._inf += int(np.isinf(arr).sum())
        self._neg += int((arr < 0).sum())

        off = 0
        for key, dim, cnt in FeatureConfig.TOKEN_SEGMENTS:
            seg_len = dim * cnt
            seg = arr[off:off + seg_len]
            st = self._tk[key]
            for t in range(cnt):
                st["exists_sum"] += seg[t * dim]          # 第 0 维 = exists
            m = seg.mean()
            st["act_sum"] += m
            st["act_sum_sq"] += m * m
            st["dim_sum"] += seg
            st["dim_sum_sq"] += seg * seg
            off += seg_len

        g_seg = arr[off:off + FeatureConfig.GLOBAL_DIM]
        gm = g_seg.mean()
        self._g_sum += gm
        self._g_sum_sq += gm * gm
        self._g_dim_sum += g_seg
        self._g_dim_sum_sq += g_seg * g_seg

    def get_stats(self):
        """返回整局聚合的特征健康指标，并重置累加器。"""
        n = self._n
        if n == 0:
            self._reset_stats()
            return {}

        out = {
            "feat_nan": self._nan,
            "feat_inf": self._inf,
            "feat_neg": self._neg,
            "feat_frames": n,
        }

        # 按 TOKEN_SEGMENTS 中每个 type_key 逐一报告
        for key, dim, cnt in FeatureConfig.TOKEN_SEGMENTS:
            st = self._tk[key]
            tag = key  # "main_hero", "enemy_hero", ...

            # exists 率（所有 token 槽平均）
            out[f"feat_{tag}_exists"] = round(st["exists_sum"] / (n * cnt), 3)

            # 激活均值 / 标准差（反映该 token 组整体活跃度）
            mean_a = st["act_sum"] / n
            var_a = st["act_sum_sq"] / n - mean_a * mean_a
            std_a = max(0.0, var_a) ** 0.5
            out[f"feat_{tag}_mean"] = round(mean_a, 4)
            out[f"feat_{tag}_std"] = round(std_a, 4)

            # 死特征维度数：整局标准差 < 1e-6 → 该维完全不变化
            dim_mean = st["dim_sum"] / n
            dim_var = st["dim_sum_sq"] / n - dim_mean * dim_mean
            dim_std = np.sqrt(np.maximum(dim_var, 0))
            dead = int((dim_std < 1e-6).sum())
            out[f"feat_{tag}_dead"] = dead

        # global 段（无 exists）
        g_mean = self._g_sum / n
        g_var = self._g_sum_sq / n - g_mean * g_mean
        g_std = max(0.0, g_var) ** 0.5
        out["feat_global_mean"] = round(g_mean, 4)
        out["feat_global_std"] = round(g_std, 4)
        g_dim_mean = self._g_dim_sum / n
        g_dim_var = self._g_dim_sum_sq / n - g_dim_mean * g_dim_mean
        g_dim_std = np.sqrt(np.maximum(g_dim_var, 0))
        out["feat_global_dead"] = int((g_dim_std < 1e-6).sum())

        self._reset_stats()
        return out
