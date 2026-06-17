#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Use the agent_diy feature engineering path for agent_ppo experiments while
leaving PPO reward and model logic under agent_ppo.
"""

from agent_diy.feature.feature_process import FeatureProcess


__all__ = ["FeatureProcess"]
