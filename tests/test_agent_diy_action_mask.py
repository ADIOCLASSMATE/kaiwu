#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import types
import unittest

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.action_mask import adjust_target_legal_for_button


def _install_platform_stubs():
    common = sys.modules.setdefault("common_python", types.ModuleType("common_python"))
    utils = sys.modules.setdefault("common_python.utils", types.ModuleType("common_python.utils"))
    common.utils = utils

    common_func = types.ModuleType("common_python.utils.common_func")

    def create_cls(name, **defaults):
        def __init__(self, **kwargs):
            for key, value in defaults.items():
                setattr(self, key, kwargs.get(key, value))

        return type(name, (), {"__init__": __init__})

    class Frame:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    common_func.create_cls = create_cls
    common_func.Frame = Frame
    sys.modules["common_python.utils.common_func"] = common_func

    kaiwudrl = sys.modules.setdefault("kaiwudrl", types.ModuleType("kaiwudrl"))
    interface = sys.modules.setdefault("kaiwudrl.interface", types.ModuleType("kaiwudrl.interface"))
    agent_module = types.ModuleType("kaiwudrl.interface.agent")

    class BaseAgent:
        pass

    agent_module.BaseAgent = BaseAgent
    sys.modules["kaiwudrl.interface.agent"] = agent_module
    kaiwudrl.interface = interface


class AgentDiyActionMaskTests(unittest.TestCase):
    def test_normal_attack_masks_none_and_self_when_entity_target_exists(self):
        mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)

        adjusted = adjust_target_legal_for_button(3, mask)

        self.assertEqual(adjusted[0], 0.0)
        self.assertEqual(adjusted[2], 0.0)
        self.assertEqual(adjusted[1], 1.0)
        self.assertEqual(adjusted[3], 1.0)
        self.assertEqual(mask[0], 1.0)

    def test_normal_attack_keeps_original_mask_when_no_entity_target_exists(self):
        mask = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        adjusted = adjust_target_legal_for_button(3, mask)

        np.testing.assert_array_equal(adjusted, mask)

    def test_non_normal_attack_keeps_target_mask_unchanged(self):
        mask = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)

        adjusted = adjust_target_legal_for_button(4, mask)

        np.testing.assert_array_equal(adjusted, mask)

    def test_update_legal_action_stores_adjusted_normal_attack_target_mask(self):
        _install_platform_stubs()
        from agent_diy.feature.definition import _update_legal_action

        fixed_size = sum(Config.LABEL_SIZE_LIST[:-1])
        target_size = Config.LABEL_SIZE_LIST[-1]
        button_size = Config.LABEL_SIZE_LIST[0]
        original = np.ones(fixed_size + button_size * target_size, dtype=np.float32)
        original[-button_size * target_size:] = 0.0
        target_matrix = original[-button_size * target_size:].reshape(button_size, target_size)
        target_matrix[3, [0, 2, 3]] = 1.0

        compressed = _update_legal_action(original, [3, 15, 15, 15, 15, 0])
        target_mask = compressed[-target_size:]

        np.testing.assert_array_equal(
            target_mask,
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        )

    def test_sample_masked_action_does_not_pick_none_or_self_for_normal_attack(self):
        _install_platform_stubs()
        from agent_diy.agent import Agent

        agent = Agent.__new__(Agent)
        agent.label_size_list = Config.LABEL_SIZE_LIST
        agent.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        legal_action = np.zeros(sum(Config.LEGAL_ACTION_SIZE_LIST), dtype=np.float32)
        offset = 0
        for head, size in enumerate(Config.LEGAL_ACTION_SIZE_LIST):
            if head == 0:
                legal_action[offset + 3] = 1.0
            elif head < len(Config.LEGAL_ACTION_SIZE_LIST) - 1:
                legal_action[offset + size - 1] = 1.0
            offset += size

        target_matrix = legal_action[-Config.LEGAL_ACTION_SIZE_LIST[-1]:].reshape(
            Config.LABEL_SIZE_LIST[0],
            Config.LABEL_SIZE_LIST[-1],
        )
        target_matrix[3, [0, 2, 3]] = 1.0
        logits = np.zeros(Config.LABEL_SUM, dtype=np.float32)

        _, _, action, d_action = agent._sample_masked_action(logits, legal_action)

        self.assertEqual(action[0], 3)
        self.assertEqual(action[5], 3)
        self.assertEqual(d_action[0], 3)
        self.assertEqual(d_action[5], 3)
