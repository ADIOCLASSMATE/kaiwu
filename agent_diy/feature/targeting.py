#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Shared target-slot and relationship helpers for agent_diy features/reward."""

import math


TOWER_SUBTYPE = 21
MINION_ACTOR_TYPE = 1
MINION_SUBTYPE = 11
BUILDING_ACTOR_TYPE = 2
SENTINEL = 99999


def is_sentinel_location(loc):
    return abs(loc.get("x", 0)) >= SENTINEL or abs(loc.get("z", 0)) >= SENTINEL


def visible_to_camp(unit, camp):
    visible = unit.get("camp_visible")
    if visible is None:
        return True
    if unit.get("camp") == camp:
        return True
    try:
        return bool(visible[camp - 1])
    except (IndexError, TypeError):
        return True


def mirrored_xz(loc, mirror=False):
    x = loc["x"]
    z = loc["z"]
    if mirror:
        x = -x
        z = -z
    return x, z


def actor_map(frame_state):
    actors = {}
    for actor in (frame_state.get("hero_states", []) or []) + (
        frame_state.get("npc_states", []) or []
    ):
        runtime_id = actor.get("runtime_id")
        if runtime_id is not None:
            actors[runtime_id] = actor
    return actors


def is_enemy_soldier(npc, main_camp):
    return (
        npc.get("actor_type") == MINION_ACTOR_TYPE
        and npc.get("sub_type") == MINION_SUBTYPE
        and npc.get("camp") != main_camp
    )


def target_slot_enemy_soldiers(
    npcs,
    main_pos,
    main_camp,
    limit,
    *,
    mirror=False,
    visible_fn=visible_to_camp,
):
    """Return enemy Soldier1-4 units in environment slot order.

    Current evidence confirms Soldier slot identity is ascending runtime_id
    within the targetable soldier set. The more-than-four selection rule is
    still provisional, so we conservatively select the nearest four visible,
    alive enemy soldiers first, then sort that selected set by runtime_id.
    """
    valid = []
    for npc in npcs or []:
        if not is_enemy_soldier(npc, main_camp):
            continue
        if npc.get("hp", 0) <= 0:
            continue
        loc = npc.get("location", {}) or {}
        if is_sentinel_location(loc):
            continue
        if not visible_fn(npc, main_camp):
            continue
        pos = mirrored_xz(loc, mirror)
        if main_pos is None:
            distance = 1e18
        else:
            distance = math.hypot(pos[0] - main_pos[0], pos[1] - main_pos[1])
        valid.append(
            {
                "unit": npc,
                "pos": pos,
                "distance": distance,
            }
        )

    nearest = sorted(
        valid,
        key=lambda item: (item["distance"], item["unit"].get("runtime_id", 0)),
    )[:limit]
    return sorted(nearest, key=lambda item: item["unit"].get("runtime_id", 0))


def attack_target_features(
    actor,
    actors,
    *,
    main_hero,
    enemy_hero,
    main_camp,
    visible_fn=visible_to_camp,
):
    """Semantic attack_target bits, never raw runtime IDs.

    Layout:
      has_target, targets_me, targets_enemy_hero, targets_outer_tower,
      targets_soldier

    If the referenced target is an invisible enemy actor, only has_target is
    retained; category bits are suppressed to avoid fog-of-war leakage.
    """
    target_id = actor.get("attack_target", 0) if actor else 0
    has_target = 1.0 if target_id else 0.0
    if not target_id:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    main_id = main_hero.get("runtime_id") if main_hero else None
    enemy_id = enemy_hero.get("runtime_id") if enemy_hero else None
    target = actors.get(target_id)
    if target is not None and target.get("camp") != main_camp:
        if not visible_fn(target, main_camp):
            return [has_target, 0.0, 0.0, 0.0, 0.0]

    targets_me = 1.0 if target_id == main_id else 0.0
    targets_enemy_hero = 1.0 if target_id == enemy_id else 0.0
    targets_outer_tower = 0.0
    targets_soldier = 0.0
    if target is not None:
        targets_outer_tower = 1.0 if (
            target.get("actor_type") == BUILDING_ACTOR_TYPE
            and target.get("sub_type") == TOWER_SUBTYPE
        ) else 0.0
        targets_soldier = 1.0 if (
            target.get("actor_type") == MINION_ACTOR_TYPE
            and target.get("sub_type") == MINION_SUBTYPE
        ) else 0.0

    return [
        has_target,
        targets_me,
        targets_enemy_hero,
        targets_outer_tower,
        targets_soldier,
    ]
