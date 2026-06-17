#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

FeatureBuilder（v2）：构造 FEATURE_DIM 维特征向量。

相对上一版的改动：
  - 结构实体仅保留外塔(21)；own/enemy 各 1 个 tower token，删除 struct type one-hot。
  - present 解耦：每个 token 第 0 维是 exists（=padding，槽位是否被占用，唯一驱动 mask）。
    visible / alive / time_since_seen 作为普通特征，不可见实体不再整 token 清零：
    保留「最后已知位置」(仅来自本方真实观测，不偷看 ground-truth) 与「消失多久」。
  - 删除 camp_is_main 原始位（敌我由模型侧 type_key / AdaLN 条件区分）。
  - 增加 1 个资源野怪 token：仅收入 kill_income>0 的非小兵 NPC，排除泉水等
    无收益单位；Monster target 槽可 pointer 到该 token。
  - 英雄增加固定长度的三英雄通用私有状态块；小兵增加 config_id 类型 one-hot；
    地图 cake 作为独立 token 输入。

跨帧记忆：FeatureBuilder 每局由 FeatureProcess.reset 重建，故可安全地把
last-seen 记忆作为实例状态保存（不会跨局污染）。仅在「当前可观测」时更新记忆，
保证 last-known 信息全部来自本方过去的真实观测。

所有输入特征都约束到 [0, 1]。天然比率直接使用 ratio，战斗属性使用固定
scale01，经济/价格/血量/收益等重尾正数使用 log01。这样模型输入不依赖
token-level LayerNorm 也能保持尺度稳定。
"""

import math
from agent_diy.conf.conf import FeatureConfig as FC
from agent_diy.feature.targeting import (
    BUILDING_ACTOR_TYPE,
    MINION_ACTOR_TYPE,
    MINION_SUBTYPE,
    TOWER_SUBTYPE,
    actor_map,
    attack_target_features,
    target_slot_enemy_soldiers,
)


def _scale01(value, scale):
    if value is None:
        return 0.0
    if scale <= 0:
        return 0.0
    return _clip01(float(value) / scale)


def _log01(value, scale):
    """log(1+x) 压缩重尾正数，再按固定上界映射到 [0,1]。"""
    if value is None:
        return 0.0
    if scale <= 0:
        return 0.0
    v = float(value)
    if v <= 0.0:
        return 0.0
    return _clip01(math.log1p(v) / math.log1p(scale))


def _clip01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _rel_to01(diff, scale):
    """相对位移 diff，用交战尺度 scale 归一并 clip 到 [-1,1]，再映射到 [0,1]。"""
    r = diff / scale
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return 0.5 * (r + 1.0)


class FeatureBuilder:
    def __init__(self, camp):
        self.main_camp = camp                 # 1=蓝, 2=红
        self.mirror = (camp == 2)             # camp2 镜像到 camp1 视角
        self.vis_index = camp - 1             # camp_visible 主视角索引
        # 跨帧记忆：key -> {"frame": int, "x": float, "z": float, "alive": bool}
        # key ∈ {"main_hero","enemy_hero","own_tower","enemy_tower"}。
        # 野怪可能随机刷新，当前不可见/不存在时不沿用 last-seen，避免 stale 资源误导。
        self._mem = {}
        # bullet 速度记忆：FeatureProcess.reset 每局重建 builder，避免跨局污染。
        self._bullet_mem = {}

    # ---- 坐标工具 ----
    def _is_sentinel(self, loc):
        return abs(loc.get("x", 0)) >= FC.SENTINEL or abs(loc.get("z", 0)) >= FC.SENTINEL

    def _xz(self, loc):
        x = loc["x"]
        z = loc["z"]
        if self.mirror:
            x = -x
            z = -z
        return x, z

    def _visible_to_main(self, unit):
        cv = unit.get("camp_visible")
        if cv is None:
            return True
        if unit.get("camp") == self.main_camp:
            return True
        try:
            return bool(cv[self.vis_index])
        except (IndexError, TypeError):
            return True

    # ---- 记忆 ----
    def _update_mem(self, key, pos, frame_no, alive):
        self._mem[key] = {"frame": frame_no, "x": pos[0], "z": pos[1], "alive": alive}

    def _mem_pos(self, key):
        m = self._mem.get(key)
        if m is None:
            return None
        return (m["x"], m["z"])

    def _time_since_seen(self, key, frame_no):
        m = self._mem.get(key)
        if m is None:
            return 1.0   # 从未见过 → 视作「很久」
        return _clip01((frame_no - m["frame"]) / FC.TSS_SCALE)

    def _pos_block(self, pos):
        """[rel_x, rel_z, abs_x, abs_z, dist] 映射到 [0,1]；pos=None 时中性占位。"""
        if pos is None:
            return [0.5, 0.5, 0.5, 0.5, 0.0]
        out = []
        if self._main_pos is not None:
            out.append(_rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE))
            out.append(_rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE))
        else:
            out += [0.5, 0.5]
        out.append(_clip01((pos[0] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)))
        out.append(_clip01((pos[1] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)))
        d = self._dist_to_main(pos)
        out.append(_clip01(d / FC.DIST_SCALE) if d is not None else 0.0)
        return out

    # ---- 主入口 ----
    def build(self, frame_state):
        heroes = frame_state.get("hero_states", [])
        npcs = frame_state.get("npc_states", [])
        bullets = frame_state.get("bullets", []) or []
        cakes = frame_state.get("cakes", []) or []
        frame_no = frame_state.get("frame_no", 0)
        self._actors = actor_map(frame_state)

        main_hero, enemy_hero = None, None
        for h in heroes:
            if h.get("camp") == self.main_camp:
                main_hero = h
            else:
                enemy_hero = h
        self._main_hero_ref = main_hero
        self._enemy_hero_ref = enemy_hero

        # 主英雄基准位置
        self._main_pos = None
        self._main_atk_range = 0.0
        if main_hero is not None and not self._is_sentinel(main_hero.get("location", {})):
            self._main_pos = self._xz(main_hero["location"])
            self._main_atk_range = float(main_hero.get("attack_range", 0) or 0)

        # 建筑分类：只认外塔(21)，own / enemy 各一个
        own_tower, enemy_tower = None, None
        own_minions, enemy_minions, monsters = [], [], []
        for npc in npcs:
            at = npc.get("actor_type")
            st = npc.get("sub_type")
            camp = npc.get("camp")
            if at == BUILDING_ACTOR_TYPE and st == TOWER_SUBTYPE:
                if camp == self.main_camp:
                    own_tower = npc
                else:
                    enemy_tower = npc
            elif at == MINION_ACTOR_TYPE and st == MINION_SUBTYPE:
                if camp == self.main_camp:
                    own_minions.append(npc)
                else:
                    enemy_minions.append(npc)
            elif self._is_resource_monster(npc):
                monsters.append(npc)
            # 其它（二塔/水晶/出兵营/无收益中立单位等）忽略

        # 预计算「主英雄是否在敌方外塔攻击范围内」(危险)
        self._danger_cache = 0.0
        if enemy_tower is not None and self._main_pos is not None \
                and not self._is_sentinel(enemy_tower.get("location", {})):
            etpos = self._xz(enemy_tower["location"])
            d = self._dist_to_main(etpos)
            ar = float(enemy_tower.get("attack_range", 0) or 0)
            if d is not None and ar > 0 and d <= ar:
                self._danger_cache = 1.0

        feat = []
        # ---- hero tokens ----
        feat += self._hero_token(main_hero, "main_hero", True, enemy_hero, frame_no)
        feat += self._hero_token(enemy_hero, "enemy_hero", False, enemy_hero, frame_no)
        # ---- tower tokens（own / enemy 各 1）----
        feat += self._tower_token(own_tower, "own_tower", frame_no)
        feat += self._tower_token(enemy_tower, "enemy_tower", frame_no)
        # ---- minions（按到主英雄距离取最近 N）----
        feat += self._minion_tokens(own_minions)
        enemy_items = target_slot_enemy_soldiers(
            enemy_minions,
            self._main_pos,
            self.main_camp,
            FC.N_MINION_PER_CAMP,
            mirror=self.mirror,
            visible_fn=lambda unit, _camp: self._visible_to_main(unit),
        )
        feat += self._minion_tokens(enemy_minions, ordered_items=enemy_items)
        # ---- monster（按到主英雄距离取最近 N；当前 N=1）----
        feat += self._monster_tokens(monsters)
        # ---- hero-sourced bullets（enemy first）----
        feat += self._bullet_tokens(bullets, frame_no)
        # ---- cakes（按到主英雄距离排序，最多两个）----
        feat += self._cake_tokens(cakes)
        # ---- global ----
        feat += self._global_feature(frame_no, main_hero, enemy_hero, own_tower, enemy_tower)

        assert len(feat) == FC.FEATURE_DIM, "feature len %d != FEATURE_DIM %d" % (
            len(feat), FC.FEATURE_DIM)
        return feat

    # ---- 距离 ----
    def _dist_to_main(self, pos):
        if self._main_pos is None or pos is None:
            return None
        dx = pos[0] - self._main_pos[0]
        dz = pos[1] - self._main_pos[1]
        return math.sqrt(dx * dx + dz * dz)

    def _in_main_atk_range(self, pos):
        d = self._dist_to_main(pos)
        if d is None or self._main_atk_range <= 0:
            return 0.0
        return 1.0 if d <= self._main_atk_range else 0.0

    @staticmethod
    def _is_resource_monster(npc):
        return (npc.get("actor_type") == MINION_ACTOR_TYPE
                and npc.get("sub_type") != MINION_SUBTYPE
                and npc.get("kill_income", 0) > 0
                and npc.get("hp", 0) > 0)

    # ---- 英雄 token ----
    def _hero_token(self, hero, mem_key, is_main, enemy_hero, frame_no):
        dim = FC.HERO_DIM
        if hero is None:
            return [0.0] * dim                       # exists=0：槽位空（pre-spawn/缺失）

        loc = hero.get("location", {})
        sentinel = self._is_sentinel(loc)
        visible = self._visible_to_main(hero)
        alive = hero.get("hp", 0) > 0
        observable = (not sentinel) and visible and alive

        pos = None
        if observable:
            pos = self._xz(loc)
            self._update_mem(mem_key, pos, frame_no, alive)
        use_pos = pos if pos is not None else self._mem_pos(mem_key)

        f = []
        # 状态块(4): exists, visible, alive, time_since_seen —— 1v1 英雄恒存在 → exists=1
        f.append(1.0)
        f.append(1.0 if observable else 0.0)
        f.append(1.0 if alive else 0.0)
        f.append(self._time_since_seen(mem_key, frame_no))

        # hp/ep（动态，雾中不可知 → 0）
        if observable:
            max_hp = hero.get("max_hp", 0) or 1
            max_ep = hero.get("max_ep", 0) or 1
            f.append(_clip01(hero.get("hp", 0) / max_hp))
            f.append(_clip01(hero.get("ep", 0) / max_ep))
        else:
            f += [0.0, 0.0]

        # config one-hot（身份静态，始终已知）
        cid = hero.get("config_id")
        onehot = [1.0 if cid == h else 0.0 for h in FC.HERO_CONFIG_IDS]
        onehot.append(1.0 if cid not in FC.HERO_CONFIG_IDS else 0.0)
        f += onehot

        # 位置块(5)：可观测用当前，否则用最后已知，再否则中性
        f += self._pos_block(use_pos)

        # forward 朝向(2)（动态）
        if observable:
            fwd = hero.get("forward", {})
            fx, fz = fwd.get("x", 0), fwd.get("z", 0)
            if self.mirror:
                fx, fz = -fx, -fz
            norm = math.sqrt(fx * fx + fz * fz)
            if norm > 1e-6:
                f += [0.5 * (fx / norm + 1.0), 0.5 * (fz / norm + 1.0)]
            else:
                f += [0.5, 0.5]
        else:
            f += [0.5, 0.5]

        # 以下全部为动态战斗/成长属性：雾中不可知 → 0
        if observable:
            f.append(_clip01(hero.get("level", 1) / 15.0))
            f.append(_log01(hero.get("money", 0), FC.HERO_MONEY_LOG_SCALE))
            f.append(_log01(hero.get("exp", 0), FC.HERO_EXP_LOG_SCALE))
            f.append(_scale01(hero.get("phy_atk", 0), FC.HERO_PHY_ATK_SCALE))
            f.append(_scale01(hero.get("phy_def", 0), FC.HERO_PHY_DEF_SCALE))
            f.append(_scale01(hero.get("mgc_atk", 0), FC.HERO_MGC_ATK_SCALE))
            f.append(_scale01(hero.get("mgc_def", 0), FC.HERO_MGC_DEF_SCALE))
            f.append(_scale01(hero.get("mov_spd", 0), FC.HERO_MOV_SPD_SCALE))
            f.append(_scale01(hero.get("atk_spd", 0), FC.HERO_ATK_SPD_SCALE))
            f.append(_scale01(hero.get("crit_rate", 0), FC.CRIT_RATE_SCALE))
            f.append(_scale01(hero.get("crit_effe", 0), FC.CRIT_EFFE_SCALE))
            f.append(_scale01(hero.get("phy_vamp", 0), FC.PHY_VAMP_SCALE))
            f.append(_scale01(hero.get("mgc_vamp", 0), FC.MGC_VAMP_SCALE))
            f.append(_scale01(hero.get("hp_recover", 0), FC.HERO_HP_RECOVER_SCALE))
            f.append(_scale01(hero.get("ep_recover", 0), FC.HERO_EP_RECOVER_SCALE))
            f.append(_scale01(hero.get("phy_armor_hurt", 0), FC.HERO_PHY_ARMOR_HURT_SCALE))
            f.append(_scale01(hero.get("mgc_armor_hurt", 0), FC.HERO_MGC_ARMOR_HURT_SCALE))
            f.append(_scale01(hero.get("cd_reduce", 0), FC.CD_REDUCE_SCALE))
            f.append(_scale01(hero.get("ctrl_reduce", 0), FC.CTRL_REDUCE_SCALE))
            f.append(_scale01(hero.get("sight_area", 0), FC.HERO_SIGHT_AREA_SCALE))
            f.append(1.0 if hero.get("is_in_grass", False) else 0.0)
            f += self._skill_feature(hero)
            f += self._hero_private_feature(hero)
        else:
            f += [0.0] * (
                3 + 4 + 2 + 4 + 2 + 2 + 2 + 1 + 1
                + FC.SKILL_DIM + FC.HERO_PRIVATE_DIM
            )

        # 交互范围标志(2)（动态）
        if is_main:
            f.append(self._danger_cache if observable else 0.0)
            f.append(self._enemy_in_my_range(enemy_hero))
        else:
            f.append(self._enemy_in_my_range(hero))
            f.append(0.0)

        # abilities / attack_target / equip 是动态状态：敌方不可见或死亡时置 0，避免雾区泄露。
        if observable:
            f += self._ability_feature(hero)
            f += self._attack_target_feature(hero)
            f += self._equip_feature(hero)
        else:
            f += [0.0] * (FC.HERO_ABILITY_DIM + FC.ATTACK_TARGET_DIM + FC.EQUIP_DIM)

        return f

    def _hero_private_feature(self, hero):
        """Return the fixed-width private state block shared by all heroes."""
        hero_id = hero.get("config_id")
        slots = {
            slot.get("slot_type"): slot
            for slot in ((hero.get("skill_state", {}) or {}).get("slot_states", []) or [])
        }

        phase = [0.0] * FC.HERO_PRIVATE_PHASE_DIM
        passive = slots.get(0)
        passive_id = passive.get("configId") if passive else None
        phase_map = FC.HERO_PASSIVE_PHASE_IDS.get(hero_id)
        phase_index = phase_map.get(passive_id) if phase_map is not None else None
        unknown = phase_index is None
        if phase_index is None:
            phase[-1] = 1.0
        else:
            phase[phase_index] = 1.0

        active_variants = []
        active_base_ids = FC.HERO_ACTIVE_BASE_IDS.get(hero_id, {})
        for slot_type in (1, 2, 3):
            slot = slots.get(slot_type)
            config_id = slot.get("configId") if slot else None
            base_id = active_base_ids.get(slot_type)
            active_variants.append(
                1.0
                if config_id is not None and base_id is not None and config_id != base_id
                else 0.0
            )

        return phase + active_variants + [1.0 if unknown else 0.0]

    def _ability_feature(self, hero):
        abilities = hero.get("abilities", []) or []
        out = []
        for index in FC.HERO_ABILITY_BITS:
            out.append(1.0 if index < len(abilities) and abilities[index] else 0.0)
        return out

    def _attack_target_feature(self, actor):
        return attack_target_features(
            actor,
            self._actors,
            main_hero=self._main_hero_ref,
            enemy_hero=self._enemy_hero_ref,
            main_camp=self.main_camp,
            visible_fn=lambda unit, _camp: self._visible_to_main(unit),
        )

    def _equip_feature(self, hero):
        """6 格装备：每槽 [exists, buyPrice_log, has_active, has_passive]。

        不编码 configId onehot（装备 ID 空间过大）。装备价格作为档次代理；
        主动/被动技能标志影响 button 11（装备技能）可用性判断。
        """
        equips = ((hero.get("equip_state", {}) or {}).get("equips", []) or [])
        out = []
        for i in range(FC.EQUIP_SLOTS):
            if i < len(equips):
                eq = equips[i]
                has = 1.0 if eq.get("configId", 0) != 0 else 0.0
                out.append(has)
                out.append(_log01(eq.get("buyPrice", 0), FC.EQUIP_BUY_PRICE_LOG_SCALE))
                out.append(1.0 if eq.get("active_skill") else 0.0)
                out.append(1.0 if eq.get("passive_skill") else 0.0)
            else:
                out += [0.0] * FC.EQUIP_FEAT_PER_SLOT
        return out

    def _enemy_in_my_range(self, enemy_hero):
        if enemy_hero is None:
            return 0.0
        loc = enemy_hero.get("location", {})
        if self._is_sentinel(loc) or not self._visible_to_main(enemy_hero):
            return 0.0
        if enemy_hero.get("hp", 0) <= 0:
            return 0.0
        return self._in_main_atk_range(self._xz(loc))

    # ---- 技能 token ----
    def _skill_feature(self, hero):
        """每槽 3 维 [usable, cd_remaining, level_ratio]，按 FC.SKILL_SLOT_TYPES 顺序。
        额外：召唤师槽(SUMMONER_SLOT_TYPE) 的 configId one-hot（+unknown）拼在末尾。
        本命技能(0-3)/回城(5)/装备(7) 的 configId 由英雄 config_id 决定或全局唯一，
        是冗余信息，不编码。
        """
        slots = {}
        ss = hero.get("skill_state", {}) or {}
        for s in ss.get("slot_states", []) or []:
            slots[s.get("slot_type")] = s

        out = []
        for st in FC.SKILL_SLOT_TYPES:
            s = slots.get(st)
            if s is None:
                out += [0.0, 0.0, 0.0]
                continue
            usable = 1.0 if s.get("usable") else 0.0
            cdmax = s.get("cooldown_max", 0) or 0
            cd = s.get("cooldown", 0) or 0
            cd_remaining = _clip01(cd / cdmax) if cdmax > 0 else 0.0
            level_ratio = _clip01(s.get("level", 0) / 6.0)
            out += [usable, cd_remaining, level_ratio]

        # 召唤师技能 one-hot（+unknown）：从召唤师槽的 configId 取
        summoner = slots.get(FC.SUMMONER_SLOT_TYPE)
        cid = summoner.get("configId") if summoner else None
        onehot = [1.0 if cid == sid else 0.0 for sid in FC.SUMMONER_SKILL_IDS]
        onehot.append(1.0 if cid not in FC.SUMMONER_SKILL_IDS else 0.0)  # unknown
        out += onehot

        return out

    # ---- 外塔 token ----
    def _tower_token(self, npc, mem_key, frame_no):
        dim = FC.STRUCT_DIM
        if npc is None or npc.get("hp", 0) <= 0:
            return [0.0] * dim                       # exists=0：无塔 / 已被摧毁（真的没了）

        loc = npc.get("location", {})
        sentinel = self._is_sentinel(loc)
        visible = self._visible_to_main(npc)
        alive = npc.get("hp", 0) > 0
        observable = (not sentinel) and visible

        # 塔位置静态、公开：只要拿到非哨兵坐标就记忆/使用（不算偷看）
        pos = None
        if not sentinel:
            pos = self._xz(loc)
            self._update_mem(mem_key, pos, frame_no, alive)
        use_pos = pos if pos is not None else self._mem_pos(mem_key)

        f = []
        # 状态块(4)
        f.append(1.0)                                # exists
        f.append(1.0 if observable else 0.0)         # visible
        f.append(1.0 if alive else 0.0)              # alive（塔的存亡是公开事件）
        f.append(self._time_since_seen(mem_key, frame_no))
        # hp_ratio（动态，不可见 → 0）
        if observable:
            max_hp = npc.get("max_hp", 0) or 1
            f.append(_clip01(npc.get("hp", 0) / max_hp))
        else:
            f.append(0.0)
        # 位置块(5)
        f += self._pos_block(use_pos)
        # attack_range_soft(1)（塔的攻击范围是配置常量，可视为已知）
        atk_range = float(npc.get("attack_range", 0) or 0)
        f.append(_scale01(atk_range, FC.TOWER_ATTACK_RANGE_SCALE))
        # main_in_range(1)（动态：需要当前位置）
        in_range = 0.0
        d = self._dist_to_main(use_pos) if use_pos is not None else None
        if observable and d is not None and atk_range > 0 and d <= atk_range:
            in_range = 1.0
        f.append(in_range)
        f += self._attack_target_feature(npc) if observable else [0.0] * FC.ATTACK_TARGET_DIM
        return f

    # ---- 小兵 tokens ----
    def _minion_tokens(self, minions, ordered_items=None):
        if ordered_items is None:
            valid = []
            for m in minions:
                loc = m.get("location", {})
                if self._is_sentinel(loc):
                    continue
                if not self._visible_to_main(m):
                    continue
                if m.get("hp", 0) <= 0:
                    continue
                pos = self._xz(loc)
                d = self._dist_to_main(pos)
                valid.append((d if d is not None else 1e18, m, pos, d))
            valid.sort(key=lambda t: t[0])
        else:
            valid = [
                (item["distance"], item["unit"], item["pos"], item["distance"])
                for item in ordered_items
            ]

        out = []
        n = FC.N_MINION_PER_CAMP
        for i in range(n):
            if i < len(valid):
                _, m, pos, d = valid[i]
                out += self._minion_token(m, pos, d)
            else:
                out += [0.0] * FC.MINION_DIM         # exists=0：空槽
        return out

    def _minion_token(self, m, pos, d):
        f = []
        f.append(1.0)  # exists
        max_hp = m.get("max_hp", 0) or 1
        f.append(_clip01(m.get("hp", 0) / max_hp))
        f.append(_log01(m.get("hp", 0), FC.MINION_HP_LOG_SCALE))
        if self._main_pos is not None:
            f.append(_rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE))
            f.append(_rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE))
        else:
            f += [0.5, 0.5]
        f.append(_clip01(d / FC.DIST_SCALE) if d is not None else 0.0)
        f.append(self._in_main_atk_range(pos))
        f.append(_log01(m.get("kill_income", 0), FC.UNIT_KILL_INCOME_LOG_SCALE))
        f += self._attack_target_feature(m)
        f += self._arli_mark_feature(m)
        f += self._minion_type_feature(m)
        return f

    def _minion_type_feature(self, minion):
        config_id = minion.get("config_id")
        onehot = [
            1.0 if config_id == known_id else 0.0
            for known_id in FC.MINION_CONFIG_IDS
        ]
        onehot.append(1.0 if config_id not in FC.MINION_CONFIG_IDS else 0.0)
        return onehot

    def _arli_mark_feature(self, actor):
        marks = ((actor.get("buff_state", {}) or {}).get("buff_marks", []) or [])
        for mark in marks:
            if mark.get("configId") == FC.ARLI_MARK_CONFIG_ID:
                layer = mark.get("layer", 0) or 0
                return [1.0, _clip01(layer / 3.0)]
        return [0.0, 0.0]

    # ---- 野怪 token ----
    def _monster_tokens(self, monsters):
        valid = []
        for m in monsters:
            loc = m.get("location", {})
            if self._is_sentinel(loc):
                continue
            if not self._visible_to_main(m):
                continue
            pos = self._xz(loc)
            d = self._dist_to_main(pos)
            valid.append((d if d is not None else 1e18, m, pos, d))
        valid.sort(key=lambda t: t[0])

        out = []
        n = FC.N_MONSTER
        for i in range(n):
            if i < len(valid):
                _, m, pos, d = valid[i]
                out += self._monster_token(m, pos, d)
            else:
                out += [0.0] * FC.MONSTER_DIM
        return out

    def _monster_token(self, m, pos, d):
        f = []
        f.append(1.0)  # exists
        max_hp = m.get("max_hp", 0) or 1
        f.append(_clip01(m.get("hp", 0) / max_hp))
        f.append(_log01(m.get("hp", 0), FC.MONSTER_HP_LOG_SCALE))
        if self._main_pos is not None:
            f.append(_rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE))
            f.append(_rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE))
        else:
            f += [0.5, 0.5]
        f.append(_clip01(d / FC.DIST_SCALE) if d is not None else 0.0)
        f.append(self._in_main_atk_range(pos))
        f.append(_log01(m.get("kill_income", 0), FC.UNIT_KILL_INCOME_LOG_SCALE))
        return f

    # ---- hero bullet tokens ----
    def _bullet_tokens(self, bullets, frame_no):
        candidates = []
        current_ids = set()
        for bullet in bullets:
            source = self._actors.get(bullet.get("source_actor"))
            if source is None or source.get("actor_type") != 0:
                continue
            loc = bullet.get("location", {}) or {}
            if self._is_sentinel(loc):
                continue
            pos = self._xz(loc)
            d = self._dist_to_main(pos)
            runtime_id = bullet.get("runtime_id")
            current_ids.add(runtime_id)
            previous = self._bullet_mem.get(runtime_id)
            vx, vz = 0.0, 0.0
            if previous is not None and frame_no != previous["frame_no"]:
                delta = frame_no - previous["frame_no"]
                if delta > 0:
                    vx = (pos[0] - previous["x"]) / delta
                    vz = (pos[1] - previous["z"]) / delta
            self._bullet_mem[runtime_id] = {
                "frame_no": frame_no,
                "x": pos[0],
                "z": pos[1],
            }
            source_enemy = source.get("camp") != self.main_camp
            candidates.append(
                (
                    0 if source_enemy else 1,
                    d if d is not None else 1e18,
                    runtime_id if runtime_id is not None else 0,
                    bullet,
                    source,
                    pos,
                    d,
                    vx,
                    vz,
                )
            )

        self._bullet_mem = {
            rid: value for rid, value in self._bullet_mem.items()
            if rid in current_ids
        }
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        out = []
        for i in range(FC.N_BULLETS):
            if i < len(candidates):
                _, _, _, bullet, source, pos, d, vx, vz = candidates[i]
                out += self._bullet_token(bullet, source, pos, d, vx, vz)
            else:
                out += [0.0] * FC.BULLET_DIM
        return out

    def _bullet_token(self, bullet, source, pos, d, vx, vz):
        f = [1.0]
        f.append(1.0 if source.get("camp") != self.main_camp else 0.0)

        cid = source.get("config_id")
        hero_onehot = [1.0 if cid == h else 0.0 for h in FC.HERO_CONFIG_IDS]
        hero_onehot.append(1.0 if cid not in FC.HERO_CONFIG_IDS else 0.0)
        f += hero_onehot

        slot = bullet.get("slot_type")
        slot_onehot = [1.0 if slot == st else 0.0 for st in FC.BULLET_SLOT_TYPES]
        slot_onehot.append(1.0 if slot not in FC.BULLET_SLOT_TYPES else 0.0)
        f += slot_onehot

        if self._main_pos is not None:
            f.append(_rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE))
            f.append(_rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE))
        else:
            f += [0.5, 0.5]
        f.append(_clip01(d / FC.DIST_SCALE) if d is not None else 0.0)
        f.append(_rel_to01(vx, FC.BULLET_VEL_SCALE))
        f.append(_rel_to01(vz, FC.BULLET_VEL_SCALE))
        return f

    # ---- cake tokens ----
    def _cake_tokens(self, cakes):
        candidates = []
        for cake in cakes:
            loc = ((cake.get("collider", {}) or {}).get("location", {}) or {})
            if not loc or self._is_sentinel(loc):
                continue
            pos = self._xz(loc)
            distance = self._dist_to_main(pos)
            candidates.append(
                (
                    distance if distance is not None else 1e18,
                    pos[0],
                    pos[1],
                    pos,
                    distance,
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        out = []
        for index in range(FC.N_CAKES):
            if index < len(candidates):
                _, _, _, pos, distance = candidates[index]
                out += self._cake_token(pos, distance)
            else:
                out += [0.0] * FC.CAKE_DIM
        return out

    def _cake_token(self, pos, distance):
        if self._main_pos is None:
            relative_position = [0.5, 0.5]
        else:
            relative_position = [
                _rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE),
                _rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE),
            ]
        absolute_position = [
            _clip01((pos[0] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)),
            _clip01((pos[1] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)),
        ]
        normalized_distance = (
            _clip01(distance / FC.DIST_SCALE) if distance is not None else 0.0
        )
        return [1.0] + relative_position + absolute_position + [normalized_distance]

    # ---- 全局特征 ----
    def _global_feature(self, frame_no, main_hero, enemy_hero, own_tower, enemy_tower):
        g = []
        g.append(_clip01(frame_no / 20000.0))
        g += self._game_time_onehot(frame_no)

        own_alive = 1.0 if (own_tower is not None and own_tower.get("hp", 0) > 0) else 0.0
        enemy_alive = 1.0 if (enemy_tower is not None and enemy_tower.get("hp", 0) > 0) else 0.0
        g.append(own_alive)
        g.append(enemy_alive)

        g.append(self._danger_cache)
        g.append(self._enemy_in_my_range(enemy_hero))

        def hp_ratio(h, observable_only=False):
            if h is None or h.get("hp", 0) <= 0:
                return 0.0
            mh = h.get("max_hp", 0) or 1
            return _clip01(h.get("hp", 0) / mh)

        # 血量优势：敌英雄不可见时其 hp 未知，按 0 处理（保守）
        evis = (enemy_hero is not None
                and not self._is_sentinel(enemy_hero.get("location", {}))
                and self._visible_to_main(enemy_hero)
                and enemy_hero.get("hp", 0) > 0)
        main_hp = hp_ratio(main_hero)
        enemy_hp = hp_ratio(enemy_hero) if evis else 0.0
        g.append(0.5 * ((main_hp - enemy_hp) + 1.0))

        def lvl(h):
            return (h.get("level", 0) if h else 0)
        lvl_adv = (lvl(main_hero) - (lvl(enemy_hero) if evis else 0)) / 15.0
        g.append(_clip01(0.5 * (lvl_adv + 1.0)))

        def money(h):
            return (h.get("money", 0) if h else 0)
        money_adv = (money(main_hero) - (money(enemy_hero) if evis else 0)) / 10000.0
        money_adv = max(-1.0, min(1.0, money_adv))
        g.append(0.5 * (money_adv + 1.0))

        g.append(1.0 if evis else 0.0)
        return g

    @staticmethod
    def _game_time_onehot(frame_no):
        bucket = 0
        for boundary in FC.GAME_TIME_BUCKETS:
            if frame_no >= boundary:
                bucket += 1
            else:
                break
        return [1.0 if idx == bucket else 0.0 for idx in range(FC.GAME_TIME_ONEHOT_DIM)]
