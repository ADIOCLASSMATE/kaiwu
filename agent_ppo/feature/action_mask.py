#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Button-conditioned action-mask adjustments.

These helpers keep inference-time sampling and training-time stored legal masks
aligned.  They intentionally only cover high-confidence global constraints.
"""

import numpy as np

from agent_ppo.conf.conf import Config


NORMAL_ATTACK_BUTTON = 3
TARGET_NONE = 0
TARGET_SELF = 2
ENTITY_TARGETS = (1, 3, 4, 5, 6, 7, 8)
FALLBACK_BUTTONS = (2, 1)


def normal_attack_entity_target_legal(target_mask):
    flat = np.asarray(target_mask).reshape(-1)
    if flat.size != Config.LABEL_SIZE_LIST[-1]:
        return False
    return bool(np.any(flat[list(ENTITY_TARGETS)] > 0))


def adjust_target_legal_for_button(button, target_mask):
    """Return a target mask adjusted for the selected top-level button.

    Normal attack should never choose None/Self.  If no concrete target is
    legal, the top-level normal-attack button must be masked before sampling.
    """
    adjusted = np.asarray(target_mask).copy()
    original_shape = adjusted.shape
    flat = adjusted.reshape(-1)
    if flat.size != Config.LABEL_SIZE_LIST[-1]:
        return adjusted

    try:
        button = int(button)
    except (TypeError, ValueError):
        return adjusted

    if button != NORMAL_ATTACK_BUTTON:
        return adjusted

    flat[TARGET_NONE] = 0
    flat[TARGET_SELF] = 0
    return flat.reshape(original_shape)


def adjust_raw_legal_action_for_button_targets(legal_action, return_stats=False):
    """Adjust a raw 184-dim legal-action mask before action sampling.

    For normal attack, target0(None) and target2(Self) are globally invalid. If
    no concrete normal-attack target remains legal, disable button3 itself so
    the policy falls back to another legal top-level action.
    """
    adjusted = np.asarray(legal_action).copy()
    stats = _empty_stats()
    raw_target_size = Config.LABEL_SIZE_LIST[-1] * Config.LABEL_SIZE_LIST[0]
    expected_raw = sum(Config.LEGAL_ACTION_SIZE_LIST)
    if adjusted.reshape(-1).size != expected_raw:
        return (adjusted, stats) if return_stats else adjusted

    flat = adjusted.reshape(-1)
    button_size = Config.LABEL_SIZE_LIST[0]
    target_size = Config.LABEL_SIZE_LIST[-1]
    button_mask = flat[:button_size]
    target_matrix = flat[-raw_target_size:].reshape(button_size, target_size)

    b = NORMAL_ATTACK_BUTTON
    if button_mask[b] > 0:
        stats["button3_legal_checked_cnt"] = 1
        if normal_attack_entity_target_legal(target_matrix[b]):
            stats["button3_entity_target_legal_cnt"] = 1
        else:
            stats["button3_no_entity_target_legal_cnt"] = 1
            stats["button3_masked_no_entity_target_cnt"] = 1
            button_mask[b] = 0

        if target_matrix[b, TARGET_NONE] > 0 or target_matrix[b, TARGET_SELF] > 0:
            stats["button3_target0_or_self_suppressed_cnt"] = 1

    target_matrix[b] = adjust_target_legal_for_button(b, target_matrix[b])
    if not np.any(button_mask > 0):
        fallback_button = _restore_fallback_button(button_mask, legal_action)
        if fallback_button is not None and not np.any(target_matrix[fallback_button] > 0):
            target_matrix[fallback_button, TARGET_NONE] = 1

    return (adjusted, stats) if return_stats else adjusted


def action_mask_stats_rates(stats):
    checked = stats.get("button3_legal_checked_cnt", 0)
    out = dict(stats)
    out["button3_entity_target_legal_rate"] = round(
        stats.get("button3_entity_target_legal_cnt", 0) / checked if checked > 0 else 0.0,
        4,
    )
    out["button3_no_entity_target_legal_rate"] = round(
        stats.get("button3_no_entity_target_legal_cnt", 0) / checked if checked > 0 else 0.0,
        4,
    )
    out["button3_target0_or_self_suppressed_rate"] = round(
        stats.get("button3_target0_or_self_suppressed_cnt", 0) / checked if checked > 0 else 0.0,
        4,
    )
    return out


def _empty_stats():
    return {
        "button3_legal_checked_cnt": 0,
        "button3_entity_target_legal_cnt": 0,
        "button3_no_entity_target_legal_cnt": 0,
        "button3_masked_no_entity_target_cnt": 0,
        "button3_target0_or_self_suppressed_cnt": 0,
    }


def _restore_fallback_button(button_mask, original_legal_action):
    original = np.asarray(original_legal_action).reshape(-1)
    for button in FALLBACK_BUTTONS:
        if button < original.size and original[button] > 0:
            button_mask[button] = original[button]
            return button
    button_mask[FALLBACK_BUTTONS[-1]] = 1
    return FALLBACK_BUTTONS[-1]
