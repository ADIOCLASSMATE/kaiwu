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
     改为「按实体类型共享投影 + (type×camp) AdaLN 条件」在模型侧注入。
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
    | cake x2
    | global(GLOBAL_DIM) ]
  其中 main_hero 固定为 token index 0。
  每个 token 第 0 维为 exists（>0.5 表示槽位被占用），供模型构造 key_padding_mask。
"""


def _build_field_slices(fields):
    field_slices = {}
    offset = 0
    for name, width in fields:
        field_slices[name] = slice(offset, offset + width)
        offset += width
    return field_slices


def _build_token_slices(token_segments):
    token_slices = {}
    offset = 0
    for type_key, dim, count in token_segments:
        ranges = []
        for _ in range(count):
            ranges.append(slice(offset, offset + dim))
            offset += dim
        token_slices[type_key] = tuple(ranges)
    return token_slices


class GameConfig:
    # 目标优先的 shaping reward。终局胜负奖励单独配置，避免被时间衰减。
    #
    # 设计目标：
    #   1. 对线期高频信号（补刀、经济、经验、血量）要稳定压过站撸噪声；
    #   2. 推塔奖励只作为终局目标的中低频信号，越塔/无兵线硬点塔折价；
    #   3. 死亡规避不能只等 dead_cnt 发生后才反馈，低血处在威胁区时给轻量逐帧惩罚。
    REWARD_WEIGHT_DICT = {
        "tower_hp_point": 1.5,      # 双方外塔血量优势变化，越塔/无兵线推塔时折价
        "lane_progress": -0.1,      # 满血后场位置惩罚：泉水强，神符到中线极弱
        "hp_point": 2.0,            # 英雄对英雄伤害优势变化，只奖励自己打出的压制
        "danger_penalty": -0.5,     # 低血仍在敌英雄/敌塔威胁区的逐帧惩罚
        "kill": 2.5,                # 击杀数优势变化；不再与 death 重复计数
        "death": -2.0,              # 自身死亡增量，直接压制送死捷径
        "money": 0.8,               # 累计经济 money_cnt 优势变化
        "exp": 0.8,                 # 跨等级累计经验优势变化
        "last_hit": 1.0,            # dead_action 中英雄真实补刀/阻止敌方补刀事件
        "minion_hp_point": 0.1,     # 敌方英雄攻击己方兵的小惩罚；不奖励无脑清线
        "kill_monster": 0.3,        # dead_action 中中立野怪归属
        "idle_penalty": -0.1,       # 长时间停滞后的渐进式每帧惩罚
        "tower_attack": 0.02,       # 安全压塔时选择点塔动作的小奖励
    }
    TERMINAL_WIN_REWARD = 8.0
    TOWER_DIVE_DISCOUNT = 0.25
    TOWER_NO_MINION_DISCOUNT = 0.15
    TOWER_PUSH_MINION_RADIUS = 6500
    HERO_DAMAGE_REWARD_SCALE = 3000.0
    # 终局奖励已经提供结束压力；关闭全局时间衰减，保持 shaping 尺度稳定。
    TIME_SCALE_ARG = 0
    MODEL_SAVE_INTERVAL = 1800

    # ---- 越程攻击惩罚（distance shaping，与 action 相关，独立于上面的帧差子项）----
    OUT_OF_RANGE_PENALTY = 0.01      # 轻量动作惩罚；设 0 关闭
    ATTACK_BUTTONS = (3, 4, 5, 6, 8, 10, 11)

    # ---- 低血危险区惩罚 ----
    DANGER_HP_THRESHOLD = 0.45       # 低于该血量，若仍处于敌方威胁区则开始惩罚
    DANGER_RANGE_MULT = 1.15         # 敌方攻击距离的安全余量
    DANGER_FRAME_SCALE = 1.0 / 30.0  # 逐帧尺度，避免比终局/击杀奖励更尖锐
    # 敌方明显更残时允许低血反打/追击：只豁免敌英雄威胁，不豁免敌塔威胁。
    DANGER_COUNTERPLAY_HP_RATIO = 0.8

    # ---- 满血后场位置引导 ----
    # lane_progress 现在表示“惩罚强度”而不是正向势函数：
    #   泉水/深后场约 1.0；
    #   到己方神符高次衰减到一个很小的 epsilon；
    #   己方神符到中线只保留极弱线性引导；
    #   中线及以后为 0。仅满血启用，避免惩罚合理回撤。
    LANE_GUIDANCE_HP_THRESHOLD = 0.99
    LANE_GUIDANCE_EPSILON = 0.01
    LANE_GUIDANCE_BACK_EXPONENT = 4.0
    LANE_GUIDANCE_FOUNTAIN_T = -0.25
    LANE_GUIDANCE_FALLBACK_CAKE_T = -0.08

    # ---- 挂机检测参数（纯产出停滞判据）----
    # 判据：经济(money_cnt)与对英雄伤害(total_hurt_to_hero)帧间增量同时停滞 → 累计 inactive。
    # 叠加「非回城/非泉水」豁免（冻结计数而非清零）：在己方塔后方的安全回撤/泉水区不罚。
    IDLE_GRACE_FRAMES = 150          # 宽限期（帧，~5s）：赶路/等 CD/补刀间隙不罚
    IDLE_RAMP_FRAMES = 600           # 爬升期（帧，~20s）：从 0 线性到满额惩罚
    IDLE_MAX_VALUE = 1.0             # idle_penalty value 封顶：防累积成巨额负悬崖
    IDLE_FRAME_SCALE = 1.0 / 30.0    # 满额时约每秒 -0.1，而不是每帧 -0.1
    # 回撤/泉水豁免：英雄到敌方外塔的距离 > (己方外塔到敌方外塔距离 × 此比例) 时，
    # 视为在己方塔后方安全区（回血/回城/泉水），冻结挂机计数。1.0=己方塔位置，
    # >1.0 表示更靠后。设为 1.05 给己方塔身前一点余量仍算"在场"。
    IDLE_RETREAT_RATIO = 1.05
    # 挂机检测位置增量阈值：英雄帧间位移 > 此值则视为"有移动"，重置挂机计数。
    # 地图尺度 MAP_SCALE=46000，英雄移速 ~350，每帧(~1/30s)位移约 12 单位。
    # 设 30 约需 2-3 帧连续移动才能重置；过滤掉贴墙卡住的微小抖动。
    IDLE_POS_DELTA_THRESHOLD = 30

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

    # ---- 三英雄共用的私有状态协议 ----
    # 固定 10 维，不为不同英雄改变 token 长度：
    #   passive_phase: phase0..phase4 + unknown = 6
    #   active_variant: slot1/slot2/slot3 是否处于非基础 configId = 3
    #   private_unknown: 英雄或被动阶段无法识别 = 1
    HERO_PRIVATE_PHASE_DIM = 6
    HERO_ACTIVE_VARIANT_DIM = 3
    HERO_PRIVATE_DIM = HERO_PRIVATE_PHASE_DIM + HERO_ACTIVE_VARIANT_DIM + 1
    HERO_PASSIVE_PHASE_IDS = {
        112: {11200: 0, 11201: 1, 11202: 2, 11203: 3, 11204: 4},
        133: {13300: 0, 13301: 1, 13302: 2},
        199: {19900: 0, 19901: 1, 19902: 2, 19903: 3, 19904: 4},
    }
    HERO_ACTIVE_BASE_IDS = {
        112: {1: 11210, 2: 11220, 3: 11230},
        133: {1: 13310, 2: 13320, 3: 13330},
        199: {1: 19910, 2: 19920, 3: 19930},
    }

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
    EQUIP_FEAT_PER_SLOT = 4  # exists, buyPrice_soft, has_active, has_passive
    EQUIP_DIM = EQUIP_SLOTS * EQUIP_FEAT_PER_SLOT  # = 24

    # ---- 各实体 token 维度 ----
    # 连续特征归一化尺度。来源优先级：
    #   1) ../kaiwu-others/data/field_inventory.csv 的 12000 帧实测范围；
    #   2) ../kaiwu-others/code/agent_ppo/feature/feature_process/*_config.ini；
    #   3) 字段本身的万分比语义。
    # builder 侧会用 scale01/log01 把连续值压到 [0, 1]，从而允许模型输入
    # 不再依赖 token-level LayerNorm。
    HERO_MONEY_LOG_SCALE = 4000.0
    HERO_EXP_LOG_SCALE = 2202.0
    HERO_PHY_ATK_SCALE = 900.0
    HERO_PHY_DEF_SCALE = 1200.0
    HERO_MGC_ATK_SCALE = 1200.0
    HERO_MGC_DEF_SCALE = 700.0
    HERO_MOV_SPD_SCALE = 7000.0
    HERO_ATK_SPD_SCALE = 10000.0
    HERO_HP_RECOVER_SCALE = 220.0
    HERO_EP_RECOVER_SCALE = 70.0
    HERO_PHY_ARMOR_HURT_SCALE = 240.0
    HERO_MGC_ARMOR_HURT_SCALE = 170.0
    HERO_SIGHT_AREA_SCALE = 10000.0
    CRIT_RATE_SCALE = 10000.0
    CRIT_EFFE_SCALE = 17000.0
    PHY_VAMP_SCALE = 10000.0
    MGC_VAMP_SCALE = 10000.0
    CD_REDUCE_SCALE = 3600.0
    CTRL_REDUCE_SCALE = 3500.0
    EQUIP_BUY_PRICE_LOG_SCALE = 2500.0
    TOWER_ATTACK_RANGE_SCALE = 13000.0
    MINION_HP_LOG_SCALE = 12000.0
    MONSTER_HP_LOG_SCALE = 9000.0
    UNIT_KILL_INCOME_LOG_SCALE = 250.0

    HERO_FIELDS = (
        ("status", HERO_STATUS_DIM),
        ("vitals", 2),
        ("hero_id", HERO_ID_ONEHOT_DIM),
        ("position", 5),
        ("forward", 2),
        ("progress", 3),
        ("offense_defense", 4),
        ("movement_attack_speed", 2),
        ("crit_vamp", 4),
        ("recovery", 2),
        ("armor_hurt", 2),
        ("reductions", 2),
        ("sight", 1),
        ("in_grass", 1),
        ("skills", SKILL_DIM),
        ("private_state", HERO_PRIVATE_DIM),
        ("range_flags", 2),
        ("abilities", HERO_ABILITY_DIM),
        ("attack_target", ATTACK_TARGET_DIM),
        ("equipment", EQUIP_DIM),
    )
    HERO_FIELD_SLICES = _build_field_slices(HERO_FIELDS)
    HERO_DIM = sum(width for _, width in HERO_FIELDS)  # 124

    # STRUCT_DIM: 状态块(4) + hp_ratio(1) + rel_pos(2) + abs_pos(2) + dist(1)
    #             + attack_range_soft(1) + main_in_range(1) + attack_target(5)
    STRUCT_DIM = STRUCT_STATUS_DIM + 1 + 2 + 2 + 1 + 1 + 1 + ATTACK_TARGET_DIM   # = 17


    # MINION_DIM: exists(1) + hp_ratio(1) + hp_soft(1) + rel_pos(2) + dist(1)
    #             + in_my_atk_range(1) + kill_income_soft(1) + attack_target(5)
    #             + Arli mark 19900 presence/layer_ratio(2) + minion type(7)
    ARLI_MARK_CONFIG_ID = 19900
    ARLI_MARK_DIM = 2
    MINION_CONFIG_IDS = [6800, 6801, 6802, 6803, 6804, 6805]
    MINION_TYPE_DIM = len(MINION_CONFIG_IDS) + 1
    MINION_FIELDS = (
        ("exists", MINION_STATUS_DIM),
        ("hp_ratio", 1),
        ("hp_soft", 1),
        ("relative_position", 2),
        ("distance", 1),
        ("in_attack_range", 1),
        ("kill_income", 1),
        ("attack_target", ATTACK_TARGET_DIM),
        ("arli_mark", ARLI_MARK_DIM),
        ("minion_type", MINION_TYPE_DIM),
    )
    MINION_FIELD_SLICES = _build_field_slices(MINION_FIELDS)
    MINION_DIM = sum(width for _, width in MINION_FIELDS)  # 22

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

    # CAKE_DIM: exists + rel_pos(2) + abs_pos(2) + distance
    CAKE_FIELDS = (
        ("exists", 1),
        ("relative_position", 2),
        ("absolute_position", 2),
        ("distance", 1),
    )
    CAKE_FIELD_SLICES = _build_field_slices(CAKE_FIELDS)
    CAKE_DIM = sum(width for _, width in CAKE_FIELDS)  # 6

    # ---- token 数量 ----
    N_STRUCT_PER_CAMP = 1   # 仅外塔(21)
    N_MINION_PER_CAMP = 4   # enemy: 最近4个后按 runtime_id 对齐 Soldier1-4；own: 距离排序
    N_MONSTER = 1           # target 槽 Monster = 最近 1 个有收益野怪
    N_BULLETS = 4           # hero-sourced bullets, enemy hero bullets first
    N_CAKES = 2             # 地图上同时最多观测到两个神符

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
        ("cakes",         CAKE_DIM, N_CAKES),
    ]
    TOKEN_SLICES = _build_token_slices(TOKEN_SEGMENTS)

    # ---- 模型侧用到的语义映射（放这里做 single source of truth）----
    # 每个 type_key 的「类型」(决定共享投影) 与「阵营」(连同类型决定 AdaLN 条件)。
    TYPE_OF = {
        "main_hero": "hero", "enemy_hero": "hero",
        "own_tower": "structure", "enemy_tower": "structure",
        "own_minions": "minion", "enemy_minions": "minion",
        "monsters": "monster",
        "bullets": "bullet",
        "cakes": "cake",
    }
    CAMP_OF = {
        "main_hero": "ego", "enemy_hero": "enemy",
        "own_tower": "own", "enemy_tower": "enemy",
        "own_minions": "own", "enemy_minions": "enemy",
        "monsters": "neutral",
        "bullets": "neutral",
        "cakes": "neutral",
    }
    # AdaLN 条件 = type_key（每个 type_key 唯一对应一个 (type,camp) 组合）。
    COND_KEYS = ["main_hero", "enemy_hero", "own_tower",
                 "enemy_tower", "own_minions", "enemy_minions",
                 "monsters", "bullets", "cakes"]

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
    NUM_TOKENS = sum(count for _, _, count in TOKEN_SEGMENTS)            # 19
    TOKEN_FEATURE_DIM = sum(dim * count for _, dim, count in TOKEN_SEGMENTS)  # 542
    FEATURE_DIM = TOKEN_FEATURE_DIM + GLOBAL_DIM                          # 551

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
    LSTM_TIME_STEPS = 8
    LSTM_UNIT_SIZE = 256

    # ---- entity-transformer 编码器超参（model 读取）----
    EMBED_DIM = 192          # token 嵌入维度 d_model
    N_HEADS = 4              # 注意力头数
    N_LAYERS = 3            # encoder 层数
    FFN_MULT = 4            # FFN 隐层 = FFN_MULT * EMBED_DIM
    N_REGISTER = 2          # register token 数量（学习式池化）
    GLOBAL_PROJ_DIM = 64     # 全局特征投影维度

    # AdaLN gate 初始值。0 = 原始 AdaLN-Zero（block 初始恒等，实体信息被 gate 掉，
    # 训练早期策略头看不见单位）。小正数让 attention/mlp 残差从第一步就以小幅度
    # 流入，给策略梯度一条通路。0.1 是稳健起点；若想更接近原行为可设 0.05。
    ADALN_GATE_INIT = 0.1

    # 输出头配置：label/value 头的中间隐层维度列表（不含输入和最后一层）
    # 空列表 = 输入直连输出；[256] = 一层 256 隐层（旧行为）
    LABEL_HEAD_HIDDEN_DIMS = []     # [] = 直连，[256] = 恢复旧行为
    VALUE_HEAD_HIDDEN_DIMS = []
    ACTOR_ADAPTER_DIM = 64

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
    # 诊断结论：原值 0.025 在弱 advantage 下让 entropy bonus（≈0.30）远大于
    # policy_cost（≈0.007），把策略死死摁在合法-均匀分布上、熵不下降。降到
    # 0.005 让策略梯度能把熵拉开。确认能学之后可再回调平衡探索。
    BETA_START = 0.005
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
    GRAD_CLIP_RANGE = 1.0

    # ---- checkpoint diagnostics ----
    # 环境变量 KAIWU_DIAG_* 优先；如果提交 job 时不方便设置环境变量，可直接改这里。
    DIAG_ENABLE = False
    DIAG_FRAME_STRIDE = 30
    DIAG_EPISODE_INTERVAL = 20
    DIAG_MAX_RECORDS = 2000
    DIAG_SAMPLE_VALUES = 20000

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])


# 维度配置，构建模型时使用。
class DimConfig:
    DIM_OF_FEATURE = [FeatureConfig.FEATURE_DIM]
