#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import os
import random
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
from agent_ppo.feature.action_mask import (
    action_mask_stats_rates,
    adjust_raw_legal_action_for_button_targets,
    adjust_target_legal_for_button,
)
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.conf.conf import Config, FeatureConfig, GameConfig
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.feature_process import FeatureProcess


# Available summoner skills / 可选召唤师技能
SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}
SUMMONER_SKILL_IDS = FeatureConfig.SUMMONER_SKILL_IDS


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self._action_mask_stats = {
            "button3_legal_checked_cnt": 0,
            "button3_entity_target_legal_cnt": 0,
            "button3_no_entity_target_legal_cnt": 0,
            "button3_masked_no_entity_target_cnt": 0,
            "button3_target0_or_self_suppressed_cnt": 0,
        }
        self._reset_recall_exploration_state()

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        # Select summoner skill for each hero based on hero lineup of both camps
        # 根据双方阵营英雄阵容，为己方每个英雄选择召唤师技能
        my_heroes = config_data.get("my_heroes", [])
        select_skills = {}
        for hero_id in my_heroes:
            skill_id = random.choice(SUMMONER_SKILL_IDS)
            select_skills[hero_id] = skill_id
        return select_skills

    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用
        self.hero_camp = observation["camp"]
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp)
        self._reset_recall_exploration_state()

    def _model_inference(self, list_obs_data):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)
            target_logits_by_button = getattr(self.model, "target_logits_by_button", None)

        np_output = []
        for output in output_list:
            np_output.append(output.detach().cpu().numpy())
        if target_logits_by_button is not None:
            target_logits_by_button = target_logits_by_button.detach().cpu().numpy()

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            target_logits = None
            if target_logits_by_button is not None:
                target_logits = target_logits_by_button[i]
            prob, d_prob, action, d_action = self._sample_masked_action(
                logits[i],
                legal_action[i],
                target_logits_by_button=target_logits,
            )
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    def predict(self, observation):
        # Prediction function, usually called during training
        # Returns a random sampling action
        # 预测函数，通常在训练时调用，返回随机采样动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        return action

    def exploit(self, observation):
        # Exploitation function, usually called during evaluation
        # Returns the action with the highest probability
        # 利用函数，在评估时调用，返回最大概率动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return d_action

    def observation_process(self, observation):
        feature = self.feature_processes.process_feature(observation)
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def action_process(self, observation, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            act_data.action = self._maybe_apply_recall_exploration(
                observation,
                act_data.action,
                act_data,
            )
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        # Framework provides loading opponent agent function, no need to implement function content
        # 框架提供的加载对手模型功能，无需实现函数内容
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def get_feature_stats(self):
        """Return per-episode feature health stats and reset the accumulators."""
        return self.feature_processes.get_stats()

    def consume_action_mask_stats(self):
        stats = action_mask_stats_rates(self._action_mask_stats)
        need = self._action_mask_stats.get("recall_explore_need_cnt", 0)
        legal = self._action_mask_stats.get("recall_explore_legal_cnt", 0)
        prob_sum = self._action_mask_stats.get("recall_explore_button9_prob_sum", 0.0)
        stats["recall_explore_legal_rate"] = round(legal / need if need > 0 else 0.0, 4)
        stats["recall_explore_override_rate"] = round(
            self._action_mask_stats.get("recall_explore_override_cnt", 0) / need
            if need > 0 else 0.0,
            4,
        )
        stats["recall_explore_button9_prob_avg"] = round(prob_sum / need if need > 0 else 0.0, 6)
        for key in self._action_mask_stats:
            self._action_mask_stats[key] = 0
        return stats

    def _reset_recall_exploration_state(self):
        self._recall_explore_starts_this_episode = 0
        if not hasattr(self, "_action_mask_stats"):
            self._action_mask_stats = {}
        self._action_mask_stats.update(
            {
                "recall_explore_need_cnt": 0,
                "recall_explore_legal_cnt": 0,
                "recall_explore_override_cnt": 0,
                "recall_explore_hold_cnt": 0,
                "recall_explore_button9_prob_sum": 0.0,
            }
        )

    def _maybe_apply_recall_exploration(self, observation, action, act_data):
        if not getattr(GameConfig, "RECALL_EXPLORATION_ENABLED", False):
            return action
        if self.reward_manager is None or observation is None:
            return action
        frame_state = observation.get("frame_state")
        legal_action = observation.get("legal_action")
        if frame_state is None or legal_action is None:
            return action

        context = self._recall_exploration_context(frame_state)
        if context is None or not context["should_recall"]:
            return action

        adjusted_legal = adjust_raw_legal_action_for_button_targets(np.array(legal_action))
        legal_actions = np.split(
            adjusted_legal,
            [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list) - 1)],
        )
        button_mask = legal_actions[0]
        button9_legal = self._button_is_legal(button_mask, GameConfig.RECALL_BUTTON)
        button9_prob = self._action_prob(act_data, GameConfig.RECALL_BUTTON)

        self._action_mask_stats["recall_explore_need_cnt"] += 1
        self._action_mask_stats["recall_explore_button9_prob_sum"] += button9_prob
        if button9_legal:
            self._action_mask_stats["recall_explore_legal_cnt"] += 1

        active = getattr(self.reward_manager, "_recall_channel_steps", 0) > 0
        if active:
            return self._maybe_hold_recall_channel(action, legal_actions)

        if not button9_legal:
            return action
        if int(action[0]) == GameConfig.RECALL_BUTTON:
            return action
        if (
            self._recall_explore_starts_this_episode
            >= GameConfig.RECALL_EXPLORATION_MAX_STARTS_PER_EPISODE
        ):
            return action
        if random.random() >= GameConfig.RECALL_EXPLORATION_PROB:
            return action

        self._recall_explore_starts_this_episode += 1
        self._action_mask_stats["recall_explore_override_cnt"] += 1
        return self._replace_action_button(
            action,
            GameConfig.RECALL_BUTTON,
            legal_actions[-1],
        )

    def _maybe_hold_recall_channel(self, action, legal_actions):
        if int(action[0]) in (GameConfig.RECALL_BUTTON, GameConfig.RECALL_NOOP_BUTTON):
            return action
        button_mask = legal_actions[0]
        if not self._button_is_legal(button_mask, GameConfig.RECALL_NOOP_BUTTON):
            return action
        if random.random() >= GameConfig.RECALL_EXPLORATION_HOLD_PROB:
            return action
        self._action_mask_stats["recall_explore_override_cnt"] += 1
        self._action_mask_stats["recall_explore_hold_cnt"] += 1
        return self._replace_action_button(
            action,
            GameConfig.RECALL_NOOP_BUTTON,
            legal_actions[-1],
        )

    def _recall_exploration_context(self, frame_state):
        try:
            main_hero = self.reward_manager._main_hero(frame_state)
            if main_hero is None:
                return None
            main_camp = main_hero.get("camp")
            main_hero, main_tower = self.reward_manager._get_camp_units(frame_state, main_camp)
            enemy_hero, enemy_tower = self.reward_manager._get_camp_units(frame_state, 3 - main_camp)
            should_recall = self.reward_manager.should_recall_recover(
                frame_state,
                main_camp,
                main_hero,
                main_tower,
                enemy_hero,
                enemy_tower,
            )
            return {"should_recall": should_recall}
        except (KeyError, TypeError, ValueError):
            return None

    def _replace_action_button(self, action, button, target_legal_action):
        replaced = list(action)
        replaced[0] = int(button)
        replaced[-1] = self._preferred_target_for_button(button, target_legal_action)
        return replaced

    def _preferred_target_for_button(self, button, target_legal_action):
        target_matrix = np.asarray(target_legal_action).reshape(
            self.legal_action_size[0],
            self.legal_action_size[-1] // self.legal_action_size[0],
        )
        target_mask = adjust_target_legal_for_button(button, target_matrix[int(button)])
        flat = np.asarray(target_mask).reshape(-1)
        if flat.size > 0 and flat[0] > 0:
            return 0
        legal = np.flatnonzero(flat > 0)
        return int(legal[0]) if legal.size > 0 else 0

    @staticmethod
    def _button_is_legal(button_mask, button):
        return button < len(button_mask) and button_mask[int(button)] > 0

    @staticmethod
    def _action_prob(act_data, button):
        try:
            probs = act_data.prob[0]
            return float(probs[int(button)])
        except (AttributeError, IndexError, TypeError, ValueError):
            return 0.0

    def _sample_masked_action(self, logits, legal_action, target_logits_by_button=None):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional list
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_action, mask_stats = adjust_raw_legal_action_for_button_targets(
            legal_action,
            return_stats=True,
        )
        if not hasattr(self, "_action_mask_stats"):
            self._action_mask_stats = {
                "button3_legal_checked_cnt": 0,
                "button3_entity_target_legal_cnt": 0,
                "button3_no_entity_target_legal_cnt": 0,
                "button3_masked_no_entity_target_cnt": 0,
                "button3_target0_or_self_suppressed_cnt": 0,
            }
        for key, value in mask_stats.items():
            self._action_mask_stats[key] = self._action_mask_stats.get(key, 0) + value
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        target_legal_action = adjust_target_legal_for_button(action_list[0], target_legal_action)

        legal_actions[index] = target_legal_action
        sample_target_logits = logits_split[-1]
        if target_logits_by_button is not None:
            sample_target_logits = target_logits_by_button[action_list[0]]
        probs = self._legal_soft_max(sample_target_logits, target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        target_legal_action_d = adjust_target_legal_for_button(d_action_list[0], target_legal_action_d)

        deterministic_target_logits = logits_split[-1]
        if target_logits_by_button is not None:
            deterministic_target_logits = target_logits_by_button[d_action_list[0]]
        probs = self._legal_soft_max(deterministic_target_logits, target_legal_action_d)
        d_prob_list += list(probs)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
