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

from agent_diy.conf.conf import Config


NORMAL_ATTACK_BUTTON = 3
TARGET_NONE = 0
TARGET_SELF = 2
ENTITY_TARGETS = (1, 3, 4, 5, 6, 7, 8)


def adjust_target_legal_for_button(button, target_mask):
    """Return a target mask adjusted for the selected top-level button.

    Normal attack should not choose None/Self when at least one concrete attack
    target is legal.  If the environment exposes no concrete legal target, keep
    the original mask to avoid producing an all-zero legal-action row.
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

    if not np.any(flat[list(ENTITY_TARGETS)] > 0):
        return adjusted

    flat[TARGET_NONE] = 0
    flat[TARGET_SELF] = 0
    if not np.any(flat > 0):
        return np.asarray(target_mask).copy()
    return flat.reshape(original_shape)
