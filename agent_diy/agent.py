#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import random
import numpy as np

from kaiwudrl.interface.agent import BaseAgent
from torch.optim.lr_scheduler import LambdaLR

from agent_diy.model.model import Model
from agent_diy.feature.definition import *
from agent_diy.feature.action_mask import adjust_target_legal_for_button
from agent_diy.conf.conf import Config, FeatureConfig
from agent_diy.feature.reward_process import GameRewardManager
from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.feature.feature_process import FeatureProcess
from agent_diy.diagnostics import AgentDiagnostics


SUMMONER_SKILL_MAP = {
    80102: "治疗", 80109: "疾跑", 80104: "惩击", 80108: "终结", 80110: "狂暴",
    80105: "干扰", 80103: "晕眩", 80107: "净化", 80121: "弱化", 80115: "闪现",
}
# 技能池以 FeatureConfig 为唯一真源，保证「选技能」与「特征 one-hot」对齐。
SUMMONER_SKILL_IDS = FeatureConfig.SUMMONER_SKILL_IDS


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        # learning info
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.diagnostics = AgentDiagnostics.from_env()
        self.algorithm = Algorithm(
            self.model,
            self.optimizer,
            self.scheduler,
            self.device,
            self.logger,
            self.monitor,
            diagnostics=self.diagnostics,
        )

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        if step > self.target_step:
            return self.target_lr / self.lr
        return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        my_heroes = config_data.get("my_heroes", [])
        select_skills = {}
        for hero_id in my_heroes:
            select_skills[hero_id] = random.choice(SUMMONER_SKILL_IDS)
        return select_skills

    def reset(self, observation):
        self.hero_camp = observation["camp"]
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp)

    def _model_inference(self, list_obs_data):
        feature = [obs.feature for obs in list_obs_data]
        legal_action = [obs.legal_action for obs in list_obs_data]
        lstm_cell = [obs.lstm_cell for obs in list_obs_data]
        lstm_hidden = [obs.lstm_hidden for obs in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            torch_inputs[i] = data.reshape(-1).float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = [output.detach().cpu().numpy() for output in output_list]
        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]
        target_logits_by_button = getattr(self.model, "target_logits_by_button", None)
        if target_logits_by_button is not None:
            target_logits_by_button = target_logits_by_button.detach().cpu().numpy()
        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = []
        for i in range(len(legal_action)):
            target_logits = (
                target_logits_by_button[i]
                if target_logits_by_button is not None and len(target_logits_by_button) > i
                else None
            )
            prob, d_prob, action, d_action = self._sample_masked_action(
                logits[i],
                legal_action[i],
                target_logits_by_button=target_logits,
            )
            self.diagnostics.record_policy(
                logits=logits[i],
                legal_action=legal_action[i],
                prob=prob,
                d_prob=d_prob,
                action=action,
                d_action=d_action,
                value=value[i] if len(value) > i else value,
            )
            list_act_data.append(ActData(
                action=action, d_action=d_action, prob=prob, d_prob=d_prob,
                value=value, lstm_cell=_lstm_cell[i], lstm_hidden=_lstm_hidden[i]))
        return list_act_data

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

    def observation_process(self, observation):
        # 特征处理；legal_action 传给推理的是原始 184 维（契约 A）
        feature = self.feature_processes.process_feature(observation)
        self.diagnostics.record_feature(feature)
        return ObsData(
            feature=feature,
            legal_action=observation["legal_action"],
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
        )

    def action_process(self, observation, act_data, is_stochastic):
        if is_stochastic:
            return act_data.action
        return act_data.d_action

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.diagnostics.save_checkpoint(
            f"{path}/model.ckpt-{str(id)}",
            extra_meta={
                "model_id": str(id),
                "train_step": self.algorithm.train_step,
                "param_count": sum(p.numel() for p in self.model.parameters()),
            },
        )
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def get_feature_stats(self):
        """返回整局特征健康度聚合指标，调用后内部累加器归零。"""
        return self.feature_processes.get_stats()

    def record_episode_step(self, episode, frame_no, observation, action, is_eval=False):
        if not self.diagnostics.enabled:
            return
        act_data = self.act_data
        head_entropy = []
        try:
            probs = np.array(act_data.prob).reshape(-1)
            offset = 0
            for size in self.label_size_list:
                p = probs[offset:offset + size]
                head_entropy.append(float(-np.sum(p * np.log(np.maximum(p, 1e-12)))))
                offset += size
        except Exception:
            head_entropy = []
        self.diagnostics.record_episode_step(
            episode=episode,
            frame_no=frame_no,
            observation=observation,
            action=action,
            d_action=act_data.d_action,
            head_entropy=head_entropy,
            value=act_data.value,
            is_eval=is_eval,
        )

    # ---- 以下采样契约函数原样保留（不得改动语义）----
    def _sample_masked_action(self, logits, legal_action, target_logits_by_button=None):
        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: i + 1]) for i in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            action_list.append(self._legal_sample(probs, use_max=False))
            d_action_list.append(self._legal_sample(probs, use_max=True))

        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [self.legal_action_size[0], self.legal_action_size[-1] // self.legal_action_size[0]])
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        target_legal_action = adjust_target_legal_for_button(action_list[0], target_legal_action)

        legal_actions[index] = target_legal_action
        if target_logits_by_button is not None:
            target_logits_by_button = np.asarray(target_logits_by_button)
            target_logits = target_logits_by_button[action_list[0]]
        else:
            target_logits = logits_split[-1]
        probs = self._legal_soft_max(target_logits, target_legal_action)
        prob_list += list(probs)
        action_list.append(self._legal_sample(probs, use_max=False))

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        target_legal_action_d = adjust_target_legal_for_button(d_action_list[0], target_legal_action_d)
        if target_logits_by_button is not None:
            target_logits_d = target_logits_by_button[d_action_list[0]]
        else:
            target_logits_d = logits_split[-1]
        probs = self._legal_soft_max(target_logits_d, target_legal_action_d)
        d_prob_list += list(probs)
        d_action_list.append(self._legal_sample(probs, use_max=True))

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
        if use_max:
            return np.argmax(probs)
        return np.argmax(np.random.multinomial(1, probs, size=1))
