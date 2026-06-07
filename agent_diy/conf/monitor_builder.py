#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    builder = monitor.title("智能体决策1V1")

    # ---- 算法/回报指标 ----
    builder = (
        builder.add_group(group_name="回报指标", group_name_en="reward")
        .add_panel(name="累积回报", name_en="reward", type="line", unit="")
        .add_metric(metrics_name="reward", expr="round(avg(reward{}), 0.01)")
        .end_panel()
        .add_panel(name="对局时长帧", name_en="episode_len", type="line", unit="")
        .add_metric(metrics_name="episode_len", expr="round(avg(episode_len{}), 1)")
        .end_panel()
        .end_group()
    )

    # ---- reward 子项分解：看清回报由什么驱动 ----
    reward_items = [
        ("rwd_tower_hp_point", "己方塔血量"),
        ("rwd_enemy_tower_hp", "敌方塔血量"),
        ("rwd_hp_point", "自身血量"),
        ("rwd_ep_rate", "能量"),
        ("rwd_kill", "击杀"),
        ("rwd_death", "死亡"),
        ("rwd_money", "经济"),
        ("rwd_exp", "经验"),
        ("rwd_forward", "推进"),
        ("rwd_last_hit", "补刀"),
    ]
    builder = builder.add_group(group_name="回报子项", group_name_en="reward_items")
    for en, cn in reward_items:
        builder = (
            builder.add_panel(name=cn, name_en=en, type="line", unit="")
            .add_metric(metrics_name=en, expr="round(avg(%s{}), 0.001)" % en)
            .end_panel()
        )
    builder = builder.end_group()

    # ---- distance shaping 监控 ----
    shaping_items = [
        ("out_of_range_cnt", "越程攻击次数", "1"),
        ("out_of_range_rate", "越程攻击占比", "0.0001"),
        ("out_of_range_sum", "越程惩罚累计", "0.001"),
        ("attack_action_cnt", "攻击动作次数", "1"),
    ]
    builder = builder.add_group(group_name="距离整形", group_name_en="distance_shaping")
    for en, cn, prec in shaping_items:
        builder = (
            builder.add_panel(name=cn, name_en=en, type="line", unit="")
            .add_metric(metrics_name=en, expr="round(avg(%s{}), %s)" % (en, prec))
            .end_panel()
        )
    builder = builder.end_group()

    # ---- 对局结果指标 ----
    outcome_items = [
        ("win", "胜率", "0.001"),
        ("final_level", "终局等级", "0.1"),
        ("final_money", "终局经济", "1"),
        ("kill_cnt", "击杀数", "0.1"),
        ("dead_cnt", "死亡数", "0.1"),
        ("final_hp_ratio", "终局血量比", "0.001"),
    ]
    builder = builder.add_group(group_name="对局结果", group_name_en="outcome")
    for en, cn, prec in outcome_items:
        builder = (
            builder.add_panel(name=cn, name_en=en, type="line", unit="")
            .add_metric(metrics_name=en, expr="round(avg(%s{}), %s)" % (en, prec))
            .end_panel()
        )
    builder = builder.end_group()

    return builder.build()
