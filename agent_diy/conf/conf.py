#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

agent_diy 增强版配置。

设计要点（与 learner / 环境的契约）：
  - 动作空间 LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9] 固定不变。
  - 样本里存的 legal_action 为压缩后的 85 维 (= 12+16+16+16+16+9)，
    DATA_SPLIT_SHAPE[0] = FEATURE_DIM + 85。
  - LSTM_UNIT_SIZE = 256（已启用 LSTM，区别于 ppo 基线的 512 透传）。
  - SERI_VEC_SPLIT_SHAPE / DATA_SPLIT_SHAPE / data_shapes / SAMPLE_DIM
    全部从 FeatureConfig.FEATURE_DIM 派生，改特征只需改 FeatureConfig。

特征向量布局（顺序即 model._encode 切分顺序）：
  [ main_hero_token | enemy_hero_token
    | own_struct(外塔,二塔,水晶) x3 | enemy_struct(外塔,二塔,水晶) x3
    | own_minion x4 | enemy_minion x4
    | global(GLOBAL_DIM) ]
  其中 main_hero 固定为 token index 0（契约 A）。
  每个 token 第 0 维为 present（>0.5 表示有效），供模型构造 key_padding_mask。
"""


class GameConfig:
    # 各奖励子项权重，在 reward_manager 中使用（10 子项稠密奖励）。
    # Weight of each reward item, used in reward_manager (10 dense sub-rewards).
    REWARD_WEIGHT_DICT = {
        "tower_hp_point": 5.0,      # 推塔（判定胜负的外塔 sub_type=21），零和
        "enemy_tower_hp": 4.0,      # 敌方外塔血量下降的正向激励，零和
        "hp_point": 2.0,            # 自身血量比例，零和
        "ep_rate": 0.5,            # 法力/能量比例，零和
        "kill": 1.0,                # 击杀数差，零和
        "death": -1.0,              # 死亡数差（权重为负，越少越好），零和
        "money": 0.6,              # 经济差，零和
        "exp": 0.6,                # 经验差，零和
        "forward": 0.05,           # 向敌方塔前进（仅满血时计算），非零和
        "last_hit": 0.5,           # 对小兵的最后一击（击杀小兵收益），非零和
    }
    # 时间衰减因子，在 reward_manager 中使用。0 表示关闭衰减。
    TIME_SCALE_ARG = 0
    # 模型保存间隔（秒），在 workflow 中使用。
    MODEL_SAVE_INTERVAL = 1800


class FeatureConfig:
    """特征工程的唯一真源（single source of truth）。

    改这里的 *_DIM / TOKEN_SEGMENTS / GLOBAL_DIM 后，Config 中所有维度
    会自动更新；但必须保证 FeatureProcess 实际输出长度 == FEATURE_DIM。
    """

    # ---- 英雄池 config_id（用于 one-hot） ----
    HERO_CONFIG_IDS = [112, 133, 199]      # 鲁班 / 狄仁杰 / 公孙离
    HERO_ID_ONEHOT_DIM = len(HERO_CONFIG_IDS) + 1   # +unknown

    # ---- 技能槽（实测 slot_type ∈ [0,1,2,3,5,6,7]，4 仅部分英雄保留兜底） ----
    # 全量索引 0,1,2,3,4,5,6,7：
    #   0 普攻 / 1-3 本命三技能 / 4 技能4(仅特定英雄,当前英雄池恒空,留位防扩展)
    #   5 回城(90003) / 6 召唤师技能(80xxx,同英雄不同局会变) / 7 装备技能(90005)
    # 每槽基础 3 维：usable, cd_remaining_ratio(cooldown/cooldown_max), level_ratio。
    # 仅召唤师槽(slot 6)额外编码技能身份 one-hot：英雄 config_id 推不出召唤师技能,
    # 是真正的信息缺口；本命/回城/装备技能由英雄唯一确定,configId 冗余不编码。
    SKILL_SLOT_TYPES = [0, 1, 2, 3, 4, 5, 6, 7]
    SKILL_FEAT_PER_SLOT = 3

    # 召唤师技能 one-hot：10 种已知 + 1 unknown 兜底。顺序须与 builder 一致。
    SUMMONER_SKILL_IDS = [80102, 80109, 80104, 80108, 80110,
                          80105, 80103, 80107, 80121, 80115]
    SUMMONER_SLOT_TYPE = 6
    SUMMONER_ONEHOT_DIM = len(SUMMONER_SKILL_IDS) + 1   # 11

    SKILL_DIM = (len(SKILL_SLOT_TYPES) * SKILL_FEAT_PER_SLOT   # 8*3 = 24
                 + SUMMONER_ONEHOT_DIM)                       # +11 = 35

    # ---- 各实体 token 维度 ----
    # HERO_DIM 明细见 hero_process 注释。
    HERO_DIM = (
        4          # present, hp_ratio, ep_ratio, camp_is_main
        + HERO_ID_ONEHOT_DIM   # 4: config one-hot (+unknown)
        + 2 + 2 + 1            # rel_pos(交战尺度), abs_pos(地图尺度), dist_to_main(大尺度)
        + 2                    # forward 朝向 (归一化 x,z)
        + 3                    # level_ratio, money_soft, exp_soft
        + 4                    # phy_atk/phy_def/mgc_atk/mgc_def 软饱和
        + 2                    # mov_spd, atk_spd 软饱和
        + 3                    # crit_rate, crit_effe, phy_vamp（万分比 /1e4）
        + 2                    # hp_recover, ep_recover 软饱和
        + SKILL_DIM            # 35 (8槽*3 + 召唤师 one-hot 11)
        + 2                    # in_enemy_tower_range, enemy_in_my_atk_range
    )

    # STRUCT_DIM: present, hp_ratio, camp_is_main, type_onehot(塔/二塔/水晶=3),
    #             rel_pos(2), abs_pos(2), dist_to_main(1), attack_range_soft(1), main_in_range(1)
    STRUCT_TYPE_ONEHOT_DIM = 3
    STRUCT_DIM = 1 + 1 + 1 + STRUCT_TYPE_ONEHOT_DIM + 2 + 2 + 1 + 1 + 1   # 13

    # MINION_DIM: present, hp_ratio, camp_is_main, rel_pos(2), dist_to_main(1), in_my_atk_range(1)
    MINION_DIM = 1 + 1 + 1 + 2 + 1 + 1   # 7

    # ---- token 数量 ----
    N_STRUCT_PER_CAMP = 3   # 固定顺序 [外塔21, 二塔24, 水晶23]
    N_MINION_PER_CAMP = 4   # 最近 4 个小兵（与 legal_action 的 Soldier=4 一致）

    # ---- TOKEN_SEGMENTS：(type_key, dim, count)，顺序即特征拼接顺序 ----
    # model._encode 按此自动切分各段、按 type_key 建投影。main_hero 必须是 index 0。
    TOKEN_SEGMENTS = [
        ("main_hero",       HERO_DIM,   1),
        ("enemy_hero",      HERO_DIM,   1),
        ("own_structures",  STRUCT_DIM, N_STRUCT_PER_CAMP),
        ("enemy_structures", STRUCT_DIM, N_STRUCT_PER_CAMP),
        ("own_minions",     MINION_DIM, N_MINION_PER_CAMP),
        ("enemy_minions",   MINION_DIM, N_MINION_PER_CAMP),
    ]

    # ---- 全局特征（非 token，拼在 token 之后） ----
    # frame_no, own_struct_alive, enemy_struct_alive, main_in_enemy_tower_range,
    # enemy_hero_in_my_atk_range, hp_adv, level_adv, money_adv, enemy_hero_visible
    GLOBAL_DIM = 9

    # ---- 总 token 数 / token 特征长度 / 总特征维度 ----
    NUM_TOKENS = sum(count for _, _, count in TOKEN_SEGMENTS)
    TOKEN_FEATURE_DIM = sum(dim * count for _, dim, count in TOKEN_SEGMENTS)
    FEATURE_DIM = TOKEN_FEATURE_DIM + GLOBAL_DIM

    # ---- 坐标重定标常量（详见 feature 构造器注释） ----
    MAP_SCALE = 46000.0        # 绝对位置尺度（地图对角 ~130000，单边 ~±46000）
    ENGAGE_SCALE = 13000.0     # 相对位移（交战尺度，近处高分辨率，clip 到 [-1,1]）
    DIST_SCALE = 130000.0      # 整体距离尺度（地图对角）
    SENTINEL = 99999           # |x| 或 |z| >= 该值视为不可见/死亡哨兵


# 模型与算法相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 256        # 已启用的 LSTM(256)

    # 特征 / 合法动作维度（全部派生自 FeatureConfig）
    FEATURE_DIM = FeatureConfig.FEATURE_DIM
    LEGAL_ACTION_DIM = 85       # 压缩后：12+16+16+16+16+9

    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    LABEL_SUM = sum(LABEL_SIZE_LIST)   # 85

    # 样本里一帧的字段顺序（与 _format_data 写入顺序一致）：
    #   feature+legal(FEATURE_DIM+85), reward_sum(1), advantage(1),
    #   action(6 个标量), old_prob(12+16+16+16+16+9=85), sub_action/weight(6 个标量),
    #   is_train(1), lstm_cell(256), lstm_hidden(256)
    # 注意：action 段是 6 个「动作索引标量」(各 1 维)，old_prob 段才是 6 个标签的概率分布
    # (12,16,16,16,16,9)。这与 learner 中 compute_loss 的字段切分严格对应。
    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,   # 0: feature + 压缩 legal_action
        1,                                # 1: reward_sum
        1,                                # 2: advantage
        1, 1, 1, 1, 1, 1,                 # 3-8: 6 个 action 索引（标量）
        12, 16, 16, 16, 16, 9,            # 9-14: 6 个 old_prob 分布
        1, 1, 1, 1, 1, 1,                 # 15-20: 6 个 weight(sub_action)
        1,                                # 21: is_train
        LSTM_UNIT_SIZE,                   # 22: lstm_cell
        LSTM_UNIT_SIZE,                   # 23: lstm_hidden
    ]

    # 模型按 SERI_VEC_SPLIT_SHAPE 切出 feature 与 legal_action。
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6

    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    # PPO clip / dual-clip / value-clip / adv-norm 相关
    CLIP_PARAM = 0.2
    DUAL_CLIP_PARAM = 3.0       # dual-clip 的下界系数
    VALUE_CLIP_PARAM = 0.2      # value 裁剪范围
    USE_ADV_NORM = True         # advantage 标准化
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

    # data_shapes：每个字段在一个 LSTM 序列（LSTM_TIME_STEPS 帧）里的展开长度。
    # 前 N-2 段 = 单帧维度 * LSTM_TIME_STEPS；最后 2 段为单份 LSTM 状态。
    # 用普通 for 循环而非推导式，避免类作用域名字在推导式内不可见的问题。
    data_shapes = []
    for _i in range(len(DATA_SPLIT_SHAPE) - 2):
        data_shapes.append([DATA_SPLIT_SHAPE[_i] * LSTM_TIME_STEPS])
    data_shapes.append([LSTM_UNIT_SIZE])
    data_shapes.append([LSTM_UNIT_SIZE])
    del _i

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # learner 上 reverb 样本输入维度。
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])


# 维度配置，构建模型时使用。
class DimConfig:
    DIM_OF_FEATURE = [FeatureConfig.FEATURE_DIM]