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
  5. Soldier1-4 槽位按环境 target identity 对齐：先选最近 4 个可见存活敌方
     soldier，再按 runtime_id 升序入 enemy_minions token。旧 checkpoint 因维度
     和 token 语义变化不兼容，需重新训练。

特征向量布局（顺序即 model._encode 切分顺序 / FeatureProcess 输出顺序）：
  [ main_hero | enemy_hero | own_tower | enemy_tower
    | own_minion x4 | enemy_minion x4 | monster | bullet x4
    | global(GLOBAL_DIM) ]
  其中 main_hero 固定为 token index 0。
  每个 token 第 0 维为 exists（>0.5 表示槽位被占用），供模型构造 key_padding_mask。
"""


class GameConfig:
    # 各奖励子项权重，在 reward_manager 中使用（8 子项简化稠密奖励）。
    # 面向 common_ai（只会走路）训练，目标是学会基础玩法：补刀刷钱、前压推塔、存活。
    # 所有权重针对「主英雄侧绝对值帧差」设计（非零和），因为 common_ai 不做任何操作。
    REWARD_WEIGHT_DICT = {
        "enemy_tower_hp": 10.0,     # 敌方外塔血量下降（推塔=赢，最高优先）
        "own_tower_hp": -5.0,       # 己方外塔被磨血惩罚（教防守）
        "money_gain": 3.0,          # 经济增量（鼓励补刀刷野）
        "exp_gain": 1.0,            # 经验增量（鼓励吃线）
        "kill": 2.0,                # 击杀敌方英雄（common_ai 不还手，低权重）
        "death": -8.0,              # 死亡惩罚（冲塔死 >> 击杀收益，教会敬畏防御塔）
        "forward": 0.1,             # 站位前压 [0,1]，纯位置比例，不做 hp 乘子
        "win": 20.0,                # 终局获胜（敌塔被摧毁的那一帧 = 1.0）
    }
    # 时间衰减：reward *= 0.75^(frame_no/TIME_SCALE_ARG)。
    # 使用 0.75 底数 + 20000 尺度，末期 ~20000 帧仍有 ~75% 奖励。
    # 对比 competitive 版的 0.6^4=0.13，给学习阶段更多探索时间。
    TIME_SCALE_ARG = 20000
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

    # ---- attack_target 语义关系（不编码 raw runtime_id）----
    # has_target, targets_me, targets_enemy_hero, targets_outer_tower, targets_soldier
    ATTACK_TARGET_DIM = 5

    # ---- 英雄 abilities：紧凑白名单 + 未知 raw bit ----
    # documented: NoControl, NoMove, NoSkill, NoMoveRotate, Blindness, Freeze,
    # ForbidSelect, Repressed. Observed undocumented bits 31/33 are retained as
    # raw unknown bits without naming their semantics.
    HERO_ABILITY_BITS = [0, 1, 2, 5, 7, 10, 15, 21, 31, 33]
    HERO_ABILITY_DIM = len(HERO_ABILITY_BITS)

    # ---- 装备 ----
    EQUIP_SLOTS = 6
    EQUIP_FEAT_PER_SLOT = 4  # exists, buyPrice_log, has_active, has_passive
    EQUIP_DIM = EQUIP_SLOTS * EQUIP_FEAT_PER_SLOT  # = 24

    # ---- 各实体 token 维度 ----
    # 英雄战斗属性归一化尺度（仅保留天然比率用到的 scale）
    MGC_VAMP_SCALE = 10000.0       # 万分比，同 crit_rate/phy_vamp
    CD_REDUCE_SCALE = 10000.0      # 万分比
    CTRL_REDUCE_SCALE = 10000.0    # 万分比

    HERO_DIM = (
        HERO_STATUS_DIM        # 4: exists, visible, alive, time_since_seen
        + 2                    # hp_ratio, ep_ratio
        + HERO_ID_ONEHOT_DIM   # 4: config one-hot (+unknown)
        + 2 + 2 + 1            # rel_pos(交战尺度), abs_pos(地图尺度), dist_to_main
        + 2                    # forward 朝向 (归一化 x,z)
        + 3                    # level_ratio, money_soft, exp_soft
        + 4                    # phy_atk/phy_def/mgc_atk/mgc_def 软饱和
        + 2                    # mov_spd, atk_spd 软饱和
        + 4                    # crit_rate, crit_effe, phy_vamp, mgc_vamp（万分比 /1e4）
        + 2                    # hp_recover, ep_recover 软饱和
        + 2                    # phy_armor_hurt, mgc_armor_hurt 软饱和
        + 2                    # cd_reduce, ctrl_reduce（万分比 /1e4）
        + 1                    # sight_area 视野范围软饱和
        + 1                    # is_in_grass 草丛隐身
        + SKILL_DIM            # 35 (8 槽 × 3 + 召唤师 one-hot 11)
        + 2                    # in_enemy_tower_range, enemy_in_my_atk_range
        + HERO_ABILITY_DIM     # 10: compact abilities / raw unknown bits
        + ATTACK_TARGET_DIM    # 5: semantic attack_target relations
        + EQUIP_DIM            # 24: 6 slots × 4 features
    )                          # = 114

    # STRUCT_DIM: 状态块(4) + hp_ratio(1) + rel_pos(2) + abs_pos(2) + dist(1)
    #             + attack_range_soft(1) + main_in_range(1) + attack_target(5)
    STRUCT_DIM = STRUCT_STATUS_DIM + 1 + 2 + 2 + 1 + 1 + 1 + ATTACK_TARGET_DIM   # = 17


    # MINION_DIM: exists(1) + hp_ratio(1) + hp_soft(1) + rel_pos(2) + dist(1)
    #             + in_my_atk_range(1) + kill_income_soft(1) + attack_target(5)
    #             + Arli mark 19900 presence/layer_ratio(2)
    ARLI_MARK_CONFIG_ID = 19900
    ARLI_MARK_DIM = 2
    MINION_DIM = (
        MINION_STATUS_DIM + 1 + 1 + 2 + 1 + 1 + 1
        + ATTACK_TARGET_DIM + ARLI_MARK_DIM
    )   # = 15

    # MONSTER_DIM: exists(1) + hp_ratio(1) + hp_soft(1) + rel_pos(2) + dist(1)
    #              + in_my_atk_range(1) + kill_income_soft(1)
    # Probe evidence did not show resource monsters attacking; no target bits.
    MONSTER_DIM = 1 + 1 + 1 + 2 + 1 + 1 + 1   # = 8

    # BULLET_DIM: exists, source_is_enemy, source hero one-hot(+unknown),
    #             slot_type one-hot(0-3 + unknown), rel_pos(2), dist, velocity(2).
    # Only hero-sourced bullets are encoded; skill_id was always zero in probes.
    BULLET_SLOT_TYPES = [0, 1, 2, 3]
    BULLET_SLOT_ONEHOT_DIM = len(BULLET_SLOT_TYPES) + 1
    BULLET_DIM = 1 + 1 + HERO_ID_ONEHOT_DIM + BULLET_SLOT_ONEHOT_DIM + 2 + 1 + 2   # = 16
    BULLET_VEL_SCALE = 1500.0

    # ---- token 数量 ----
    N_STRUCT_PER_CAMP = 1   # 仅外塔(21)
    N_MINION_PER_CAMP = 4   # enemy: 最近4个后按 runtime_id 对齐 Soldier1-4；own: 距离排序
    N_MONSTER = 1           # target 槽 Monster = 最近 1 个有收益野怪
    N_BULLETS = 4           # hero-sourced bullets, enemy hero bullets first

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
        ("bullets",       BULLET_DIM, N_BULLETS),
    ]

    # ---- 模型侧用到的语义映射（放这里做 single source of truth）----
    # 每个 type_key 的「类型」(决定共享投影) 与「阵营」(连同类型决定 AdaLN 条件)。
    TYPE_OF = {
        "main_hero": "hero", "enemy_hero": "hero",
        "own_tower": "structure", "enemy_tower": "structure",
        "own_minions": "minion", "enemy_minions": "minion",
        "monsters": "monster",
        "bullets": "bullet",
    }
    CAMP_OF = {
        "main_hero": "ego", "enemy_hero": "enemy",
        "own_tower": "own", "enemy_tower": "enemy",
        "own_minions": "own", "enemy_minions": "enemy",
        "monsters": "neutral",
        "bullets": "neutral",
    }
    # AdaLN 条件 = type_key（每个 type_key 唯一对应一个 (type,camp) 组合）。
    COND_KEYS = ["main_hero", "enemy_hero", "own_tower",
                 "enemy_tower", "own_minions", "enemy_minions",
                 "monsters", "bullets"]

    # ---- target 头 (label[5]) 的 9 个槽：含义 + key 来源 ----
    # key 来源为 token type_key 的，pointer 直接取该实体的 transformer 输出；
    # 为 None 的（None）使用可学习 null key。Soldier1-4 = 最近4个可见存活
    # 敌方 soldier 中按 runtime_id 升序的槽位；该规则由 feature.targeting 共享。
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
    NUM_TOKENS = sum(count for _, _, count in TOKEN_SEGMENTS)            # 17
    TOKEN_FEATURE_DIM = sum(dim * count for _, dim, count in TOKEN_SEGMENTS)  # 392
    FEATURE_DIM = TOKEN_FEATURE_DIM + GLOBAL_DIM                          # 401

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
