#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

样本数据结构与序列化（FrameCollector / GAE / _format_data /
_reshape_lstm_batch_sample）。结构与 ppo 基线完全一致，仅维度从
FeatureConfig.FEATURE_DIM 派生。

一帧样本字段顺序（与 _format_data 写入顺序一致）：
  feature(FEATURE_DIM) | legal_action(85) | reward_sum(1) | advantage(1)
  | action(6) | prob(85=12+16+16+16+16+9) | sub_action(6) | is_train(1)
  → sample_one_size = FEATURE_DIM + 85 + 1 + 1 + 6 + 85 + 6 + 1
"""

from common_python.utils.common_func import create_cls, Frame
from agent_diy.conf.conf import Config
from agent_diy.feature.action_mask import (
    adjust_raw_legal_action_for_button_targets,
    adjust_target_legal_for_button,
)
import numpy as np
import collections
import random
import itertools


def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if not camp_heroes:
        raise Exception("camp_heroes is empty")
    camps = []
    for lineups in itertools.product(camp_heroes, camp_heroes):
        camp = []
        for lineup in lineups:
            camp.append(lineup)
        camps.append(camp)
    return _lineup_iterator_shuffle_cycle(camps)


ObsData = create_cls("ObsData", feature=None, legal_action=None, lstm_cell=None, lstm_hidden=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    d_prob=None,
    value=None,
    lstm_cell=None,
    lstm_hidden=None,
)

SampleData = create_cls("SampleData", sample=sum([shape[0] for shape in Config.data_shapes]))

NONE_ACTION = [0, 15, 15, 15, 15, 0]


def sample_process(collector):
    return collector.sample_process()


# 构建当前帧样本
def build_frame(agent, observation):
    obs_data, act_data = agent.obs_data, agent.act_data
    is_train = False
    frame_state = observation["frame_state"]
    hero_list = frame_state["hero_states"]
    frame_no = frame_state["frame_no"]
    for hero in hero_list:
        if hero["camp"] == agent.hero_camp:
            is_train = True if hero["hp"] > 0 else False

    if obs_data.feature is not None:
        feature_vec = np.array(obs_data.feature)
    else:
        feature_vec = np.array(observation["observation"])

    reward = observation["reward"]["reward_sum"]
    sub_action_mask = observation["sub_action_mask"]
    prob, value, action = act_data.prob, act_data.value, act_data.action
    lstm_cell, lstm_hidden = act_data.lstm_cell, act_data.lstm_hidden

    # 184 维原始 legal_action → 85 维压缩存样本
    legal_action = _update_legal_action(observation["legal_action"], action)
    frame = Frame(
        frame_no=frame_no,
        feature=feature_vec.reshape([-1]),
        legal_action=legal_action.reshape([-1]),
        action=action,
        reward=reward,
        reward_sum=0,
        value=value.flatten()[0],
        next_value=0,
        advantage=0,
        prob=prob,
        sub_action=sub_action_mask[str(action[0])],
        lstm_info=np.concatenate([lstm_cell.flatten(), lstm_hidden.flatten()]).reshape([-1]),
        is_train=False if action[0] < 0 else is_train,
    )
    return frame


# 184(76 固定 + 12×9 target 矩阵) → 85(76 固定 + 当前 which_button 对应的 9 维 target)
def _update_legal_action(original_la, action):
    target_size = Config.LABEL_SIZE_LIST[-1]   # 9
    top_size = Config.LABEL_SIZE_LIST[0]       # 12
    original_la = adjust_raw_legal_action_for_button_targets(np.array(original_la))
    fix_part = original_la[: -target_size * top_size]
    target_la = original_la[-target_size * top_size:]
    target_la = target_la.reshape([top_size, target_size])[action[0]]
    target_la = adjust_target_legal_for_button(action[0], target_la)
    return np.concatenate([fix_part, target_la], axis=0)


class FrameCollector:
    def __init__(self, num_agents):
        self._data_shapes = Config.data_shapes
        self._LSTM_FRAME = Config.LSTM_TIME_STEPS
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def save_frame(self, rl_data_info, agent_id):
        reward = self._clip_reward(rl_data_info.reward)
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = rl_data_info.value
            last_rl_data_info.reward = reward
        rl_data_info.reward = 0
        self.rl_data_map[agent_id][rl_data_info.frame_no] = rl_data_info

    def save_last_frame(self, reward, agent_id):
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = 0
            last_rl_data_info.reward = reward

    def sample_process(self):
        self._calc_reward()
        self._format_data()
        return self.m_replay_buffer

    def _calc_reward(self):
        # GAE
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae = 0.0
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)])
        idx, s_idx = 0, 0
        sample[-sample_lstm.shape[0]:] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx:s_idx + split_shape[0]] = sample_batch[:, idx:idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return sample.astype(np.float32)

    def _format_data(self):
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size])
        first_frame_no = -1

        for i in range(self.num_agents):
            sample_lstm = np.zeros([sample_lstm_size])
            cnt = 0
            for j in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][j]
                if cnt == 0:
                    first_frame_no = rl_info.frame_no

                idx, dlen = 0, 0
                dlen = rl_info.feature.shape[0]
                sample_batch[cnt, idx:idx + dlen] = rl_info.feature
                idx += dlen

                dlen = rl_info.legal_action.shape[0]
                sample_batch[cnt, idx:idx + dlen] = rl_info.legal_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1

                dlen = 6
                sample_batch[cnt, idx:idx + dlen] = rl_info.action
                idx += dlen

                for p in rl_info.prob:
                    dlen = len(p)
                    sample_batch[cnt, idx:idx + dlen] = p
                    idx += dlen

                dlen = 6
                sample_batch[cnt, idx:idx + dlen] = rl_info.sub_action
                idx += dlen

                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    sample_array = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append(SampleData(sample=sample_array))
                    sample_lstm = rl_info.lstm_info

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    def __len__(self):
        return max([len(agent_samples) for agent_samples in self.rl_data_map])

    def is_train_rate(self, agent_id):
        """该 agent 本局采样帧中 is_train=1 的占比（监控用，纯观测）。

        诊断 suspect C：PPO 的 policy/entropy 损失分母是 sum(weight*is_train)。
        若该占比偏低（英雄频繁死亡 / action[0]<0 的无效帧多），有效策略梯度样本
        就少，advantage 归一化也会被小分母放大噪声。这条曲线让该假设可证伪：
        若 >0.5，则样本有效性不是主因，AdaLN gate + entropy 系数足够解释停滞。
        """
        frames = self.rl_data_map[agent_id]
        if not frames:
            return 0.0
        total = len(frames)
        trained = sum(1 for rl in frames.values() if getattr(rl, "is_train", 0))
        return trained / total
