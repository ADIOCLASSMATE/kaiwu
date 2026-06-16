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

    # ============================================================
    # 特征健康度（全特征覆盖：有效性 + 占位率 + 激活度 + 死特征）
    # ============================================================

    # ---- 特征有效性：NaN / Inf / 负值（任何 >0 即 bug）----
    validity_items = [
        ("feat_nan", "NaN数", "1"),
        ("feat_inf", "Inf数", "1"),
        ("feat_neg", "负值数", "1"),
        ("feat_frames", "统计帧数", "1"),
    ]
    builder = builder.add_group(group_name="特征有效性", group_name_en="feature_validity")
    for en, cn, prec in validity_items:
        builder = (
            builder.add_panel(name=cn, name_en=en, type="line", unit="")
            .add_metric(metrics_name=en, expr="round(avg(%s{}), %s)" % (en, prec))
            .end_panel()
        )
    builder = builder.end_group()

    # ---- Token 占位率（exists 标志均值）----
    tok_labels = [
        ("main_hero", "主英雄"),
        ("enemy_hero", "敌英雄"),
        ("own_tower", "己方塔"),
        ("enemy_tower", "敌方塔"),
        ("own_minions", "己方小兵"),
        ("enemy_minions", "敌方小兵"),
        ("monsters", "野怪"),
        ("bullets", "弹道"),
        ("cakes", "神符"),
    ]
    builder = builder.add_group(group_name="Token占位率", group_name_en="token_exists")
    for en, cn in tok_labels:
        metric = "feat_%s_exists" % en
        builder = (
            builder.add_panel(name=cn, name_en=metric, type="line", unit="")
            .add_metric(metrics_name=metric, expr="round(avg(%s{}), 0.001)" % metric)
            .end_panel()
        )
    builder = builder.end_group()

    # ---- Token 激活度：均值 ----
    builder = builder.add_group(group_name="Token激活均值", group_name_en="token_act_mean")
    for en, cn in tok_labels:
        metric = "feat_%s_mean" % en
        builder = (
            builder.add_panel(name=cn, name_en=metric, type="line", unit="")
            .add_metric(metrics_name=metric, expr="round(avg(%s{}), 0.0001)" % metric)
            .end_panel()
        )
    builder = builder.end_group()

    # ---- Token 激活度：标准差 ----
    builder = builder.add_group(group_name="Token激活波动", group_name_en="token_act_std")
    for en, cn in tok_labels:
        metric = "feat_%s_std" % en
        builder = (
            builder.add_panel(name=cn, name_en=metric, type="line", unit="")
            .add_metric(metrics_name=metric, expr="round(avg(%s{}), 0.0001)" % metric)
            .end_panel()
        )
    builder = builder.end_group()

    # ---- Global 激活度：均值 / 标准差 ----
    builder = builder.add_group(group_name="Global激活度", group_name_en="global_act")
    for metric, cn, prec in [
        ("feat_global_mean", "全局段均值", "0.0001"),
        ("feat_global_std", "全局段波动", "0.0001"),
    ]:
        builder = (
            builder.add_panel(name=cn, name_en=metric, type="line", unit="")
            .add_metric(metrics_name=metric, expr="round(avg(%s{}), %s)" % (metric, prec))
            .end_panel()
        )
    builder = builder.end_group()

    # ---- 死特征维度数（整局 std ≈ 0 的维度数）----
    builder = builder.add_group(group_name="死特征维度", group_name_en="dead_dims")
    for en, cn in tok_labels:
        metric = "feat_%s_dead" % en
        builder = (
            builder.add_panel(name=cn, name_en=metric, type="line", unit="")
            .add_metric(metrics_name=metric, expr="round(avg(%s{}), 1)" % metric)
            .end_panel()
        )
    # global 段
    for en, cn in [("global", "全局段")]:
        metric = "feat_%s_dead" % en
        builder = (
            builder.add_panel(name=cn, name_en=metric, type="line", unit="")
            .add_metric(metrics_name=metric, expr="round(avg(%s{}), 1)" % metric)
            .end_panel()
        )
    builder = builder.end_group()

    # ---- 算法/回报指标 ----
    builder = (
        builder.add_group(group_name="回报指标", group_name_en="reward")
        .add_panel(name="对局序号", name_en="episode_cnt", type="line", unit="")
        .add_metric(metrics_name="episode_cnt", expr="round(avg(episode_cnt{}), 1)")
        .end_panel()
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
        ("rwd_tower_hp_point", "塔血优势"),
        ("rwd_lane_progress", "满血后场惩罚"),
        ("rwd_hp_point", "英雄血量优势"),
        ("rwd_danger_penalty", "低血危险惩罚"),
        ("rwd_kill", "击杀"),
        ("rwd_money", "累计经济优势"),
        ("rwd_exp", "累计经验优势"),
        ("rwd_last_hit", "英雄补刀"),
        ("rwd_kill_monster", "野怪控制"),
        ("rwd_idle_penalty", "挂机惩罚"),
        ("rwd_distance_penalty", "越程惩罚"),
        ("rwd_terminal", "终局结果"),
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

    # ---- 挂机检测健康度 ----
    idle_items = [
        ("idle_triggered", "触发挂机帧数", "1"),
        ("idle_triggered_rate", "挂机帧占比", "0.0001"),
    ]
    builder = builder.add_group(group_name="挂机检测", group_name_en="idle_health")
    for en, cn, prec in idle_items:
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

    # ---- 训练指标：loss / 梯度 / 学习率 ----
    train_items = [
        ("total_loss", "总损失", "0.01"),
        ("value_loss", "价值损失", "0.01"),
        ("policy_loss", "策略损失", "0.01"),
        ("entropy_loss", "熵损失", "0.01"),
        ("grad_norm", "梯度范数", "0.0001"),
        ("learning_rate", "学习率", "0.00000001"),
        ("is_train_rate", "有效样本占比", "0.0001"),
    ]
    builder = builder.add_group(group_name="训练指标", group_name_en="training")
    for en, cn, prec in train_items:
        builder = (
            builder.add_panel(name=cn, name_en=en, type="line", unit="")
            .add_metric(metrics_name=en, expr="round(avg(%s{}), %s)" % (en, prec))
            .end_panel()
        )
    builder = builder.end_group()

    # ---- 策略熵分解：检测各动作头是否过早坍缩 ----
    head_names = ["head_0", "head_1", "head_2", "head_3", "head_4", "head_5"]
    head_labels = ["主标签-12", "方向1-16", "方向2-16", "方向3-16", "方向4-16", "目标-9"]
    builder = builder.add_group(group_name="策略熵分解", group_name_en="entropy_per_head")
    for en_suffix, cn_suffix in zip(head_names, head_labels):
        en = "entropy_" + en_suffix
        builder = (
            builder.add_panel(name=cn_suffix, name_en=en, type="line", unit="")
            .add_metric(metrics_name=en, expr="round(avg(%s{}), 0.0001)" % en)
            .end_panel()
        )
    builder = builder.end_group()

    # ---- advantage 统计量 ----
    adv_items = [
        ("adv_mean", "Advantage均值", "0.0001"),
        ("adv_std", "Advantage标准差", "0.0001"),
    ]
    builder = builder.add_group(group_name="Advantage统计", group_name_en="advantage")
    for en, cn, prec in adv_items:
        builder = (
            builder.add_panel(name=cn, name_en=en, type="line", unit="")
            .add_metric(metrics_name=en, expr="round(avg(%s{}), %s)" % (en, prec))
            .end_panel()
        )
    builder = builder.end_group()

    return builder.build()
