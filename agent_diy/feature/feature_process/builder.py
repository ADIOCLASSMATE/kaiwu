#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

FeatureBuilder: 构造 FEATURE_DIM 维特征向量的具体实现。
所有输出值落在 [0, 1]（相对位移先 clip[-1,1] 再映射到 [0,1]）。
"""

import math
from agent_diy.conf.conf import FeatureConfig as FC


# ---- 建筑 sub_type -> 固定 token 顺序 / 类型 one-hot ----
# [外塔21, 二塔24, 水晶23]。外塔(21) 是判定胜负的塔；二塔(24)/水晶(23) 不作为攻击目标，
# 但仍编码进特征，避免被错分进小兵（修复原 ppo 只认 21 的 BUG）。
STRUCT_ORDER = [21, 24, 23]
STRUCT_TYPE_ONEHOT = {21: [1, 0, 0], 24: [0, 1, 0], 23: [0, 0, 1]}

MINION_SUBTYPE = 11        # 实测小兵 sub_type=11（actor_type=1）
# actor_type=1 sub_type=0（config 50004，attack_range=100，固定于出生角）是出兵营/基地，
# 不是战斗单位，也不作为目标，忽略。
BUILDING_ACTOR_TYPE = 2
MINION_ACTOR_TYPE = 1


def _soft(value, k):
    """软饱和归一化 value/(value+k)，value>=0。结果 ∈ [0,1)。"""
    if value is None:
        return 0.0
    if value <= 0:
        return 0.0
    return float(value) / (float(value) + float(k))


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

    # ---- 坐标工具 ----
    def _is_sentinel(self, loc):
        return abs(loc.get("x", 0)) >= FC.SENTINEL or abs(loc.get("z", 0)) >= FC.SENTINEL

    def _xz(self, loc):
        """取镜像后的 (x, z)。调用方需先确认非哨兵。"""
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
        # 主阵营单位永远可见；否则看 camp_visible[main_camp-1]
        if unit.get("camp") == self.main_camp:
            return True
        try:
            return bool(cv[self.vis_index])
        except (IndexError, TypeError):
            return True

    # ---- 主入口 ----
    def build(self, frame_state):
        heroes = frame_state.get("hero_states", [])
        npcs = frame_state.get("npc_states", [])
        frame_no = frame_state.get("frame_no", 0)

        # 区分主/敌英雄
        main_hero, enemy_hero = None, None
        for h in heroes:
            if h.get("camp") == self.main_camp:
                main_hero = h
            else:
                enemy_hero = h

        # 主英雄基准位置（用于相对位移 / 距离 / 攻击范围判定）
        self._main_pos = None
        self._main_atk_range = 0.0
        if main_hero is not None and not self._is_sentinel(main_hero.get("location", {})):
            self._main_pos = self._xz(main_hero["location"])
            self._main_atk_range = float(main_hero.get("attack_range", 0) or 0)

        # 建筑分类：own / enemy，按固定顺序 [21,24,23]
        own_struct = {st: None for st in STRUCT_ORDER}
        enemy_struct = {st: None for st in STRUCT_ORDER}
        own_minions = []
        enemy_minions = []
        for npc in npcs:
            at = npc.get("actor_type")
            st = npc.get("sub_type")
            camp = npc.get("camp")
            if at == BUILDING_ACTOR_TYPE and st in STRUCT_ORDER:
                if camp == self.main_camp:
                    own_struct[st] = npc
                else:
                    enemy_struct[st] = npc
            elif at == MINION_ACTOR_TYPE and st == MINION_SUBTYPE:
                if camp == self.main_camp:
                    own_minions.append(npc)
                else:
                    enemy_minions.append(npc)
            # 其它（出兵营 sub_type=0 / 野怪等）忽略

        # 预先计算「主英雄是否在敌方外塔(21)攻击范围内」(危险)，供 hero_token & global 复用，
        # 避免依赖方法调用顺序。
        self._danger_cache = 0.0
        et = enemy_struct.get(21)
        if et is not None and self._main_pos is not None and not self._is_sentinel(et.get("location", {})):
            etpos = self._xz(et["location"])
            d = self._dist_to_main(etpos)
            ar = float(et.get("attack_range", 0) or 0)
            if d is not None and ar > 0 and d <= ar:
                self._danger_cache = 1.0

        feat = []

        # ---- main_hero / enemy_hero token ----
        feat += self._hero_token(main_hero, is_main=True, enemy_hero=enemy_hero)
        feat += self._hero_token(enemy_hero, is_main=False, enemy_hero=enemy_hero)

        # ---- own / enemy structures（固定顺序 [外塔,二塔,水晶]）----
        for st in STRUCT_ORDER:
            feat += self._struct_token(own_struct[st], st, is_main_camp=True)
        for st in STRUCT_ORDER:
            feat += self._struct_token(enemy_struct[st], st, is_main_camp=False)

        # ---- minions（按到主英雄距离取最近 N 个）----
        feat += self._minion_tokens(own_minions, is_main_camp=True)
        feat += self._minion_tokens(enemy_minions, is_main_camp=False)

        # ---- global ----
        feat += self._global_feature(
            frame_no, main_hero, enemy_hero, own_struct, enemy_struct
        )

        # 安全：长度校验（构造器输出必须 == FEATURE_DIM）
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

    # ---- 英雄 token ----
    def _hero_token(self, hero, is_main, enemy_hero):
        dim = FC.HERO_DIM
        zeros = [0.0] * dim

        if hero is None:
            return zeros
        loc = hero.get("location", {})
        sentinel = self._is_sentinel(loc)
        visible = self._visible_to_main(hero)
        alive = hero.get("hp", 0) > 0
        present = (not sentinel) and visible and alive
        if not present:
            # present=0、整 token 置零（不参与任何位置/距离计算）
            return zeros

        f = []
        # present, hp_ratio, ep_ratio, camp_is_main
        f.append(1.0)
        max_hp = hero.get("max_hp", 0) or 1
        f.append(_clip01(hero.get("hp", 0) / max_hp))
        max_ep = hero.get("max_ep", 0) or 1
        f.append(_clip01(hero.get("ep", 0) / max_ep))
        f.append(1.0 if hero.get("camp") == self.main_camp else 0.0)

        # config one-hot (+unknown)
        cid = hero.get("config_id")
        onehot = [1.0 if cid == h else 0.0 for h in FC.HERO_CONFIG_IDS]
        onehot.append(1.0 if cid not in FC.HERO_CONFIG_IDS else 0.0)
        f += onehot

        # 位置
        pos = self._xz(loc)
        if self._main_pos is not None:
            f.append(_rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE))
            f.append(_rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE))
        else:
            f += [0.5, 0.5]
        f.append(_clip01((pos[0] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)))
        f.append(_clip01((pos[1] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)))
        d = self._dist_to_main(pos)
        f.append(_clip01(d / FC.DIST_SCALE) if d is not None else 0.0)

        # forward 朝向（归一化到单位向量后映射到 [0,1]）
        fwd = hero.get("forward", {})
        fx, fz = fwd.get("x", 0), fwd.get("z", 0)
        if self.mirror:
            fx, fz = -fx, -fz
        norm = math.sqrt(fx * fx + fz * fz)
        if norm > 1e-6:
            f.append(0.5 * (fx / norm + 1.0))
            f.append(0.5 * (fz / norm + 1.0))
        else:
            f += [0.5, 0.5]

        # 成长
        f.append(_clip01(hero.get("level", 1) / 15.0))
        f.append(_soft(hero.get("money", 0), 5000.0))
        f.append(_soft(hero.get("exp", 0), 2000.0))

        # 战斗属性（软饱和）
        f.append(_soft(hero.get("phy_atk", 0), 400.0))
        f.append(_soft(hero.get("phy_def", 0), 300.0))
        f.append(_soft(hero.get("mgc_atk", 0), 400.0))
        f.append(_soft(hero.get("mgc_def", 0), 300.0))
        f.append(_soft(hero.get("mov_spd", 0), 4000.0))
        f.append(_soft(hero.get("atk_spd", 0), 1500.0))

        # 万分比（/1e4，clip）
        f.append(_clip01(hero.get("crit_rate", 0) / 10000.0))
        f.append(_clip01(hero.get("crit_effe", 0) / 10000.0))
        f.append(_clip01(hero.get("phy_vamp", 0) / 10000.0))

        # 恢复
        f.append(_soft(hero.get("hp_recover", 0), 100.0))
        f.append(_soft(hero.get("ep_recover", 0), 50.0))

        # 技能槽（按 slot_type 显式索引 0..6）
        f += self._skill_feature(hero)

        # 主英雄是否处于敌方塔攻击范围内 / 敌英雄是否在我普攻范围内
        if is_main:
            f.append(self._danger_cache)                  # 主英雄在敌外塔范围内 = 危险
            f.append(self._enemy_in_my_range(enemy_hero))  # 敌英雄在我普攻范围内 = 可击
        else:
            # 敌英雄 token：第一位复用「敌英雄是否在我范围内」，第二位留 0（对称占位）
            f.append(self._enemy_in_my_range(hero))
            f.append(0.0)

        return f

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
        slots = {}
        ss = hero.get("skill_state", {}) or {}
        for s in ss.get("slot_states", []) or []:
            slots[s.get("slot_type")] = s

        out = []
        # 每槽基础 3 维：usable, cd_remaining, level_ratio（按 slot_type 显式索引）
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

        # 召唤师技能身份 one-hot（英雄 config_id 推不出，必须显式编码）
        out += self._summoner_onehot(slots.get(FC.SUMMONER_SLOT_TYPE))
        return out

    def _summoner_onehot(self, slot):
        dim = FC.SUMMONER_ONEHOT_DIM
        vec = [0.0] * dim
        if slot is None:
            return vec
        cid = slot.get("configId")
        try:
            idx = FC.SUMMONER_SKILL_IDS.index(cid)
        except ValueError:
            idx = dim - 1   # unknown 兜底桶
        vec[idx] = 1.0
        return vec

    # ---- 建筑 token ----
    def _struct_token(self, npc, sub_type, is_main_camp):
        dim = FC.STRUCT_DIM
        zeros = [0.0] * dim
        if npc is None:
            return zeros
        loc = npc.get("location", {})
        sentinel = self._is_sentinel(loc)
        visible = self._visible_to_main(npc)
        alive = npc.get("hp", 0) > 0
        present = (not sentinel) and visible and alive
        if not present:
            return zeros

        f = []
        f.append(1.0)  # present
        max_hp = npc.get("max_hp", 0) or 1
        f.append(_clip01(npc.get("hp", 0) / max_hp))  # hp_ratio
        f.append(1.0 if npc.get("camp") == self.main_camp else 0.0)  # camp_is_main
        f += [float(v) for v in STRUCT_TYPE_ONEHOT[sub_type]]  # type one-hot

        pos = self._xz(loc)
        if self._main_pos is not None:
            f.append(_rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE))
            f.append(_rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE))
        else:
            f += [0.5, 0.5]
        f.append(_clip01((pos[0] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)))
        f.append(_clip01((pos[1] + FC.MAP_SCALE) / (2 * FC.MAP_SCALE)))
        d = self._dist_to_main(pos)
        f.append(_clip01(d / FC.DIST_SCALE) if d is not None else 0.0)

        atk_range = float(npc.get("attack_range", 0) or 0)
        f.append(_soft(atk_range, 12000.0))  # attack_range_soft

        # 主英雄是否处于该建筑攻击范围内
        in_range = 0.0
        if d is not None and atk_range > 0 and d <= atk_range:
            in_range = 1.0
        f.append(in_range)
        return f

    # ---- 小兵 tokens ----
    def _minion_tokens(self, minions, is_main_camp):
        # 过滤哨兵/不可见/死亡，按到主英雄距离排序，取最近 N
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

        out = []
        n = FC.N_MINION_PER_CAMP
        for i in range(n):
            if i < len(valid):
                _, m, pos, d = valid[i]
                out += self._minion_token(m, pos, d, is_main_camp)
            else:
                out += [0.0] * FC.MINION_DIM
        return out

    def _minion_token(self, m, pos, d, is_main_camp):
        f = []
        f.append(1.0)  # present
        max_hp = m.get("max_hp", 0) or 1
        f.append(_clip01(m.get("hp", 0) / max_hp))
        f.append(1.0 if m.get("camp") == self.main_camp else 0.0)
        if self._main_pos is not None:
            f.append(_rel_to01(pos[0] - self._main_pos[0], FC.ENGAGE_SCALE))
            f.append(_rel_to01(pos[1] - self._main_pos[1], FC.ENGAGE_SCALE))
        else:
            f += [0.5, 0.5]
        f.append(_clip01(d / FC.DIST_SCALE) if d is not None else 0.0)
        f.append(self._in_main_atk_range(pos))
        return f

    # ---- 全局特征 ----
    def _global_feature(self, frame_no, main_hero, enemy_hero, own_struct, enemy_struct):
        g = []
        # 1) frame_no / 20000，封顶 1
        g.append(_clip01(frame_no / 20000.0))

        # 2/3) 双方建筑存活数 / 3
        own_alive = sum(1 for st in STRUCT_ORDER
                        if own_struct[st] is not None and own_struct[st].get("hp", 0) > 0)
        enemy_alive = sum(1 for st in STRUCT_ORDER
                          if enemy_struct[st] is not None and enemy_struct[st].get("hp", 0) > 0)
        g.append(_clip01(own_alive / 3.0))
        g.append(_clip01(enemy_alive / 3.0))

        # 4) 主英雄是否在敌方塔(外塔21)攻击范围内（危险）—— 复用 build() 预计算
        g.append(self._danger_cache)

        # 5) 敌英雄是否在我普攻范围内（可击）
        g.append(self._enemy_in_my_range(enemy_hero))

        # 6) 血量优势：主 hp_ratio - 敌 hp_ratio，映射到 [0,1]
        def hp_ratio(h):
            if h is None or h.get("hp", 0) <= 0:
                return 0.0
            mh = h.get("max_hp", 0) or 1
            return _clip01(h.get("hp", 0) / mh)
        hp_adv = hp_ratio(main_hero) - hp_ratio(enemy_hero)
        g.append(0.5 * (hp_adv + 1.0))

        # 7) 等级优势
        def lvl(h):
            return (h.get("level", 0) if h else 0)
        lvl_adv = (lvl(main_hero) - lvl(enemy_hero)) / 15.0
        g.append(_clip01(0.5 * (lvl_adv + 1.0)))

        # 8) 经济优势
        def money(h):
            return (h.get("money", 0) if h else 0)
        money_adv = (money(main_hero) - money(enemy_hero)) / 10000.0
        money_adv = max(-1.0, min(1.0, money_adv))
        g.append(0.5 * (money_adv + 1.0))

        # 9) 敌英雄是否可见
        evis = 0.0
        if enemy_hero is not None and not self._is_sentinel(enemy_hero.get("location", {})) \
                and self._visible_to_main(enemy_hero) and enemy_hero.get("hp", 0) > 0:
            evis = 1.0
        g.append(evis)

        return g