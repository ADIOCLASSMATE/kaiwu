#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

agent_diy 增强版配置（v2）。

本次相对上一版的结构性改动：
  1. 结构实体只保留「外塔(21)」：1v1 墨家机关道里二塔(24)/水晶(23) 不是目标、
     对决策也无用，删掉后 own/enemy 各只剩 1 个 tower token，并删掉 struct 的
     type one-hot（只剩一种结构，one-hot 退化成常量）。
  2. present 解耦：token 第 0 维改为 exists（=padding，槽位是否被占用，纯结构信息，
     唯一驱动 mask）；visible / alive / time_since_seen 作为普通特征留在 token 内，
     不可见实体（如雾中的敌英雄）不再整 token 清零，而是保留「最后已知位置」+
     「消失多久」，由模型自己决定要不要采信。
  3. camp 不再用 token 内 camp_is_main 原始位编码，也不再用 own_/enemy_ 拆两套投影；
     改为「type 共享投影(3 个) + (type×camp) AdaLN 条件」在模型侧注入。
     own/enemy 区分仍体现在 TOKEN_SEGMENTS 的 type_key（用于映射 AdaLN 条件、
     pointer target key），但不再产生额外的 token 内特征位。
  4. target 头改为 pointer（见 model）：9 个 target 槽与 token 一一对应（见
     TARGET_SLOT_DESC），槽 0 为可学习 null key。

特征向量布局（顺序即 model._encode 切分顺序 / FeatureProcess 输出顺序）：
  [ main_hero | enemy_hero | own_tower | enemy_tower
    | own_minion x4 | enemy_minion x4 | monster | global(GLOBAL_DIM) ]
  其中 main_hero 固定为 token index 0。
  每个 token 第 0 维为 exists（>0.5 表示槽位被占用），供模型构造 key_padding_mask。
"""


class GameConfig:
    # 各奖励子项权重，在 reward_manager 中使用（11 子项稠密奖励）。
    REWARD_WEIGHT_DICT = {
        "tower_hp_point": 5.0,      # 推塔（判定胜负的外塔 sub_type=21），零和
        "enemy_tower_hp": 4.0,      # 敌方外塔血量下降的正向激励，零和
        "hp_point": 2.0,            # 自身血量比例，零和
        "ep_rate": 0.5,             # 法力/能量比例，零和
        "kill": 1.0,                # 击杀数差，零和
        "death": -1.0,              # 死亡数差（权重为负，越少越好），零和
        "money": 0.6,               # 经济差，零和
        "exp": 0.6,                 # 经验差，零和
        "forward": 0.05,            # 向敌方塔站位（HP 越高权重越大），非零和
        "last_hit": 0.4,            # 补刀收益差分（降权防刷线不推塔），非零和
        # 挂机惩罚：长时间零产出 *且* 不在回撤/泉水区时才罚。权重为负，非零和。
        "idle_penalty": -0.15,
    }
    # 时间衰减：reward *= 0.6^(frame_no/TIME_SCALE_ARG)，末期约 ×0.13。
    # 制造终局压力——越早获胜收益越大，拖到 timeout 所有 reward 几乎归零。
    TIME_SCALE_ARG = 5000
    MODEL_SAVE_INTERVAL = 1800

    # ---- 越程攻击惩罚（distance shaping，与 action 相关，独立于上面的帧差子项）----
    OUT_OF_RANGE_PENALTY = 0.01      # 量级 ~ forward(0.05) 的 1/5；设 0 关闭
    ATTACK_BUTTONS = (3, 4, 5, 6, 8, 10, 11)

    # ---- 挂机检测参数（纯产出停滞判据）----
    # 判据：经济(money_cnt)与对英雄伤害(total_hurt_to_hero)帧间增量同时停滞 → 累计 inactive。
    # 叠加「非回城/非泉水」豁免（冻结计数而非清零）：在己方塔后方的安全回撤/泉水区不罚。
    IDLE_GRACE_FRAMES = 150          # 宽限期（帧，~5s）：赶路/等 CD/补刀间隙不罚
    IDLE_RAMP_FRAMES = 600           # 爬升期（帧，~20s）：从 0 线性到满额惩罚
    IDLE_MAX_VALUE = 1.0             # idle_penalty value 封顶：防累积成巨额负悬崖
    # 回撤/泉水豁免：英雄到敌方外塔的距离 > (己方外塔到敌方外塔距离 × 此比例) 时，
    # 视为在己方塔后方安全区（回血/回城/泉水），冻结挂机计数。1.0=己方塔位置，
    # >1.0 表示更靠后。设为 1.05 给己方塔身前一点余量仍算"在场"。
    IDLE_RETREAT_RATIO = 1.05
    # forward 反 hack：处于敌方外塔攻击范围内时不发前压奖励。
    FORWARD_NO_REWARD_IN_ENEMY_TOWER = True


class FeatureConfig:
    """特征工程的唯一真源（single source of truth）。

    改 *_DIM / TOKEN_SEGMENTS / GLOBAL_DIM 后，Config 中所有维度会自动更新；
    但必须保证 FeatureProcess 实际输出长度 == FEATURE_DIM。
    """

    # ---- 英雄池 config_id（用于 one-hot） ----
    HERO_CONFIG_IDS = [112, 133, 199]      # 鲁班 / 狄仁杰 / 公孙离
    HERO_ID_ONEHOT_DIM = len(HERO_CONFIG_IDS) + 1   # +unknown

    # ---- 技能槽 ----
    # 实测 slot_type ∈ {0,1,2,3,5,6,7}：
    #   0-3 = 本命技能（被动/普攻 + 三主动，英雄 config_id 可推，不再单独编码 configId）
    #   4   = 第4技能（仅特定英雄有效；当前英雄池无 4 技能，留恒零槽以便扩英雄池不改维度）
    #   5   = 回城 (configId=90003，英雄无关，无需编码 configId)
    #   6   = 召唤师技能 (同一英雄不同局可带不同技能，英雄 config_id 推不出 → 必须编码)
    #   7   = 装备技能 (configId=90005，英雄无关，无需编码 configId)
    # 每槽基础 3 维：usable, cd_remaining_ratio(cooldown/cooldown_max), level_ratio。
    # 仅召唤师槽(6) 额外拼一个 (len(SUMMONER_SKILL_IDS)+1) 维 one-hot（+unknown）。
    SKILL_SLOT_TYPES = [0, 1, 2, 3, 4, 5, 6, 7]
    SKILL_FEAT_PER_SLOT = 3
    SUMMONER_SLOT_TYPE = 6

    # 召唤师技能池（single source of truth；agent.py 选技能 / builder 编码共用）。
    SUMMONER_SKILL_IDS = [80102, 80109, 80104, 80108, 80110,
                          80105, 80103, 80107, 80121, 80115]
    SUMMONER_ONEHOT_DIM = len(SUMMONER_SKILL_IDS) + 1   # +unknown，= 11

    SKILL_DIM = len(SKILL_SLOT_TYPES) * SKILL_FEAT_PER_SLOT + SUMMONER_ONEHOT_DIM   # 8*3 + 11 = 35

    # ---- token 通用状态块（解耦后的 present）----
    # 所有 token 第 0 维都是 exists（padding 位）。其余状态位按实体类型不同：
    #   hero/tower: exists, visible, alive, time_since_seen, ...
    #   minion:     exists, ...（小兵只在「当前可见且存活」时入槽，故省略冗余状态位）
    HERO_STATUS_DIM = 4    # exists, visible, alive, time_since_seen
    STRUCT_STATUS_DIM = 4  # exists, visible, alive, time_since_seen
    MINION_STATUS_DIM = 1  # exists

    # ---- 各实体 token 维度 ----
    HERO_DIM = (
        HERO_STATUS_DIM        # exists, visible, alive, time_since_seen
        + 2                    # hp_ratio, ep_ratio
        + HERO_ID_ONEHOT_DIM   # 4: config one-hot (+unknown)
        + 2 + 2 + 1            # rel_pos(交战尺度), abs_pos(地图尺度), dist_to_main
        + 2                    # forward 朝向 (归一化 x,z)
        + 3                    # level_ratio, money_soft, exp_soft
        + 4                    # phy_atk/phy_def/mgc_atk/mgc_def 软饱和
        + 2                    # mov_spd, atk_spd 软饱和
        + 3                    # crit_rate, crit_effe, phy_vamp（万分比 /1e4）
        + 2                    # hp_recover, ep_recover 软饱和
        + SKILL_DIM            # 35 (8 槽 × 3 + 召唤师 one-hot 11)
        + 2                    # in_enemy_tower_range, enemy_in_my_atk_range
    )                          # = 68

    # STRUCT_DIM: 状态块(4) + hp_ratio(1) + rel_pos(2) + abs_pos(2) + dist(1)
    #             + attack_range_soft(1) + main_in_range(1)
    STRUCT_DIM = STRUCT_STATUS_DIM + 1 + 2 + 2 + 1 + 1 + 1   # = 12

    # NPC 资源价值/击杀成本的软饱和尺度。收益对小兵/野怪共享，便于模型比较资源目标。
    NPC_HP_SOFT_SCALE = 4000.0
    KILL_INCOME_SOFT_SCALE = 100.0

    # MINION_DIM: exists(1) + hp_ratio(1) + hp_soft(1) + rel_pos(2) + dist(1)
    #             + in_my_atk_range(1) + kill_income_soft(1)
    MINION_DIM = MINION_STATUS_DIM + 1 + 1 + 2 + 1 + 1 + 1   # = 8

    # MONSTER_DIM: exists(1) + hp_ratio(1) + hp_soft(1) + rel_pos(2) + dist(1)
    #              + in_my_atk_range(1) + kill_income_soft(1)
    MONSTER_DIM = 1 + 1 + 1 + 2 + 1 + 1 + 1   # = 8

    # ---- token 数量 ----
    N_STRUCT_PER_CAMP = 1   # 仅外塔(21)
    N_MINION_PER_CAMP = 4   # 最近 4 个小兵（与 target 槽 Soldier1-4 一致）
    N_MONSTER = 1           # target 槽 Monster = 最近 1 个有收益野怪

    # ---- TOKEN_SEGMENTS：(type_key, dim, count)，顺序即特征拼接顺序 ----
    # type_key 同时编码了 (实体类型, 阵营)，供模型映射「共享投影(按类型) + AdaLN 条件
    # (按 type×camp)」。main_hero 必须是 index 0。
    TOKEN_SEGMENTS = [
        ("main_hero",     HERO_DIM,   1),
        ("enemy_hero",    HERO_DIM,   1),
        ("own_tower",     STRUCT_DIM, N_STRUCT_PER_CAMP),
        ("enemy_tower",   STRUCT_DIM, N_STRUCT_PER_CAMP),
        ("own_minions",   MINION_DIM, N_MINION_PER_CAMP),
        ("enemy_minions", MINION_DIM, N_MINION_PER_CAMP),
        ("monsters",      MONSTER_DIM, N_MONSTER),
    ]

    # ---- 模型侧用到的语义映射（放这里做 single source of truth）----
    # 每个 type_key 的「类型」(决定共享投影) 与「阵营」(连同类型决定 AdaLN 条件)。
    TYPE_OF = {
        "main_hero": "hero", "enemy_hero": "hero",
        "own_tower": "structure", "enemy_tower": "structure",
        "own_minions": "minion", "enemy_minions": "minion",
        "monsters": "monster",
    }
    CAMP_OF = {
        "main_hero": "ego", "enemy_hero": "enemy",
        "own_tower": "own", "enemy_tower": "enemy",
        "own_minions": "own", "enemy_minions": "enemy",
        "monsters": "neutral",
    }
    # AdaLN 条件 = type_key（每个 type_key 唯一对应一个 (type,camp) 组合）。
    COND_KEYS = ["main_hero", "enemy_hero", "own_tower",
                 "enemy_tower", "own_minions", "enemy_minions",
                 "monsters"]

    # ---- target 头 (label[5]) 的 9 个槽：含义 + key 来源 ----
    # key 来源为 token type_key 的，pointer 直接取该实体的 transformer 输出；
    # 为 None 的（None）使用可学习 null key。Soldier1-4 = 最近 4 个敌方小兵，
    # 与 enemy_minions 的入槽顺序（按到主英雄距离升序）一一对应。
    TARGET_SLOT_DESC = [
        ("None",       None),            # 0
        ("EnemyHero",  "enemy_hero"),    # 1
        ("Self",       "main_hero"),     # 2
        ("Soldier1",   "enemy_minions"), # 3  -> 最近第1个敌方小兵
        ("Soldier2",   "enemy_minions"), # 4
        ("Soldier3",   "enemy_minions"), # 5
        ("Soldier4",   "enemy_minions"), # 6
        ("Tower",      "enemy_tower"),   # 7
        ("Monster",    "monsters"),      # 8  -> 最近 1 个有收益野怪
    ]
    NUM_TARGET_SLOTS = len(TARGET_SLOT_DESC)   # 9

    # ---- 全局特征（非 token，拼在 token 之后） ----
    # frame_no, own_tower_alive, enemy_tower_alive, main_in_enemy_tower_range,
    # enemy_hero_in_my_atk_range, hp_adv, level_adv, money_adv, enemy_hero_visible
    GLOBAL_DIM = 9

    # ---- 总 token 数 / token 特征长度 / 总特征维度 ----
    NUM_TOKENS = sum(count for _, _, count in TOKEN_SEGMENTS)            # 13
    TOKEN_FEATURE_DIM = sum(dim * count for _, dim, count in TOKEN_SEGMENTS)  # 232
    FEATURE_DIM = TOKEN_FEATURE_DIM + GLOBAL_DIM                          # 241

    # ---- 坐标重定标常量 ----
    MAP_SCALE = 46000.0
    ENGAGE_SCALE = 13000.0
    DIST_SCALE = 130000.0
    SENTINEL = 99999
    # time_since_seen 归一化尺度（帧）：消失多久 → clip(dframe / TSS_SCALE, 0, 1)。
    TSS_SCALE = 300.0


# 模型与算法相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 256

    # ---- entity-transformer 编码器超参（model 读取）----
    EMBED_DIM = 128          # token 嵌入维度 d_model
    N_HEADS = 4              # 注意力头数
    N_LAYERS = 2            # encoder 层数
    FFN_MULT = 2            # FFN 隐层 = FFN_MULT * EMBED_DIM
    N_REGISTER = 2          # register token 数量（学习式池化）
    GLOBAL_PROJ_DIM = 64     # 全局特征投影维度

    # 输出头配置：label/value 头的中间隐层维度列表（不含输入和最后一层）
    # 空列表 = 输入直连输出；[256] = 一层 256 隐层（旧行为）
    LABEL_HEAD_HIDDEN_DIMS = []     # [] = 直连，[256] = 恢复旧行为
    VALUE_HEAD_HIDDEN_DIMS = []

    # 特征 / 合法动作维度（派生自 FeatureConfig）
    FEATURE_DIM = FeatureConfig.FEATURE_DIM
    LEGAL_ACTION_DIM = 85       # 压缩后：12+16+16+16+16+9

    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    LABEL_SUM = sum(LABEL_SIZE_LIST)   # 85

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

    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6

    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    CLIP_PARAM = 0.2
    DUAL_CLIP_PARAM = 3.0
    VALUE_CLIP_PARAM = 0.2
    USE_ADV_NORM = True
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

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

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])


# 维度配置，构建模型时使用。
class DimConfig:
    DIM_OF_FEATURE = [FeatureConfig.FEATURE_DIM]
