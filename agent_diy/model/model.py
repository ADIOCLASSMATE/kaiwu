#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

增强版模型：
  实体分投影 → 加类型嵌入 → Transformer 跨实体融合(带 key_padding_mask)
  → masked-mean 池化 + 主英雄 token 拼接 + 全局特征 → LSTM(256) → 多头输出。

与 learner / 环境的契约：
  - forward(inference=True) 返回 [logits(85), value(1), lstm_cell, lstm_hidden]，
    logits 切分顺序 = LABEL_SIZE_LIST = [12,16,16,16,16,9]。
  - compute_loss 实现 dual-clip / value-clip / adv-norm PPO，
    legal_action 从 feature 后段(85)切出，与 ppo 基线签名一致。
  - 训练路径 B×T，推理路径 T=1（set_eval_mode 把 lstm_time_steps 置 1）。
"""

import torch
import torch.nn as nn
from torch.nn import ModuleDict
import numpy as np
from typing import List

from agent_diy.conf.conf import Config, FeatureConfig, DimConfig


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(self, fc_feat_dim_list: List[int], name: str,
                 non_linearity: nn.Module = nn.ReLU, non_linearity_last: bool = False):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.dual_clip_param = Config.DUAL_CLIP_PARAM
        self.value_clip_param = Config.VALUE_CLIP_PARAM
        self.use_adv_norm = Config.USE_ADV_NORM
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])   # FEATURE_DIM (247)

        # ---- 实体 token 切分元信息（来自 FeatureConfig.TOKEN_SEGMENTS）----
        self.token_segments = FeatureConfig.TOKEN_SEGMENTS
        self.num_tokens = FeatureConfig.NUM_TOKENS
        self.token_feature_dim = FeatureConfig.TOKEN_FEATURE_DIM
        self.global_dim = FeatureConfig.GLOBAL_DIM

        self.embed_dim = 128

        # ---- 各 type_key 一个投影 + 一个可学习类型嵌入 ----
        type_keys = []
        for type_key, dim, count in self.token_segments:
            if type_key not in type_keys:
                type_keys.append(type_key)
        self.type_keys = type_keys
        self.entity_proj = ModuleDict({
            tk: make_fc_layer(self._dim_of_type(tk), self.embed_dim) for tk in type_keys
        })
        self.type_embedding = nn.Parameter(torch.zeros(len(type_keys), self.embed_dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.type_key_to_idx = {tk: i for i, tk in enumerate(type_keys)}

        # 展开每个 token 的 (type_key, start, dim)，main_hero 固定 index 0
        self.token_layout = []
        off = 0
        for type_key, dim, count in self.token_segments:
            for _ in range(count):
                self.token_layout.append((type_key, off, dim))
                off += dim
        assert off == self.token_feature_dim

        # ---- Transformer 跨实体融合 ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

        # ---- 池化 + 主英雄 + 全局 → LSTM 输入 ----
        self.global_proj = MLP([self.global_dim, 64, 64], "global_proj", non_linearity_last=True)
        fused_dim = self.embed_dim * 2 + 64
        self.fuse_mlp = MLP([fused_dim, self.lstm_unit_size], "fuse_mlp", non_linearity_last=True)

        self.lstm = nn.LSTM(
            input_size=self.lstm_unit_size, hidden_size=self.lstm_unit_size,
            num_layers=1, bias=True, batch_first=True)

        # ---- 多头输出 ----
        self.label_mlp = ModuleDict({
            "hero_label{0}_mlp".format(i): MLP(
                [self.lstm_unit_size, 256, self.label_size_list[i]], "hero_label{0}_mlp".format(i))
            for i in range(len(self.label_size_list))
        })
        self.value_mlp = MLP([self.lstm_unit_size, 256, 1], "hero_value_mlp")

    def _dim_of_type(self, type_key):
        for tk, dim, count in self.token_segments:
            if tk == type_key:
                return dim
        raise KeyError(type_key)

    # ---- 实体编码：feature -> fused state (B*T, lstm_unit_size) ----
    def _encode(self, feature_vec):
        token_part = feature_vec[:, : self.token_feature_dim]
        global_part = feature_vec[:, self.token_feature_dim:]

        embeds = []
        present_list = []
        for type_key, start, dim in self.token_layout:
            seg = token_part[:, start:start + dim]              # (bt, dim)
            present = (seg[:, 0:1] > 0.5).float()               # 第 0 位 = present
            proj = self.entity_proj[type_key](seg)              # (bt, embed_dim)
            tidx = self.type_key_to_idx[type_key]
            proj = proj + self.type_embedding[tidx].unsqueeze(0)
            embeds.append(proj.unsqueeze(1))
            present_list.append(present)

        tokens = torch.cat(embeds, dim=1)                       # (bt, num_tokens, embed_dim)
        present_mat = torch.cat(present_list, dim=1).clone()    # (bt, num_tokens)
        # 强制 main_hero(index 0)=present，防止整行被 mask
        present_mat[:, 0] = 1.0
        key_padding_mask = (present_mat < 0.5)                  # True=屏蔽

        fused = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        # masked-mean 池化（只对 present token 求均值）
        mask = present_mat.unsqueeze(-1)
        summed = (fused * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom

        main_hero_embed = fused[:, 0, :]
        global_embed = self.global_proj(global_part)

        concat = torch.cat([pooled, main_hero_embed, global_embed], dim=1)
        state = self.fuse_mlp(concat)
        return state

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        state = self._encode(feature_vec)                       # (B*T, H)

        t = self.lstm_time_steps
        bt = state.shape[0]
        b = bt // t
        lstm_in = state.reshape(b, t, self.lstm_unit_size)

        h0 = lstm_hidden_init.reshape(b, self.lstm_unit_size).unsqueeze(0).contiguous()
        c0 = lstm_cell_init.reshape(b, self.lstm_unit_size).unsqueeze(0).contiguous()

        lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))
        self.lstm_hidden_output = hn
        self.lstm_cell_output = cn
        flat = lstm_out.reshape(bt, self.lstm_unit_size)

        result_list = []
        for i in range(len(self.label_size_list)):
            result_list.append(self.label_mlp["hero_label{0}_mlp".format(i)](flat))
        value_result = self.value_mlp(flat)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3:3 + len(self.label_size_list)]
        for k in range(len(self.label_size_list)):
            usq_label_list[k] = usq_label_list[k].reshape(-1, self.data_split_shape[3 + k]).long()

        old_label_probability_list = data_list[3 + len(self.label_size_list):3 + 2 * len(self.label_size_list)]
        for k in range(len(self.label_size_list)):
            old_label_probability_list[k] = old_label_probability_list[k].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + k])

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list):3 + 3 * len(self.label_size_list)]
        for k in range(len(self.label_size_list)):
            usq_weight_list[k] = usq_weight_list[k].reshape(
                -1, self.data_split_shape[3 + 2 * len(self.label_size_list) + k])

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        # 保留未标准化的 advantage，用于重建 old_value（reward_sum = advantage + value）。
        raw_advantage = advantage
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [w.squeeze(dim=1) for w in usq_weight_list]
        frame_is_train = usq_is_train.squeeze(dim=1)

        if self.use_adv_norm:
            mask = frame_is_train
            denom = mask.sum().clamp(min=1.0)
            mean = (advantage * mask).sum() / denom
            var = ((advantage - mean) ** 2 * mask).sum() / denom
            advantage = (advantage - mean) / (torch.sqrt(var) + 1e-8)

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [int(np.prod(self.seri_vec_split_shape[0])), int(np.prod(self.seri_vec_split_shape[1]))],
            dim=1)
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        # value-clip：old_value = reward_sum - advantage（用未标准化的 advantage 重建），
        old_value = reward - raw_advantage
        v_clipped = old_value + torch.clamp(
            fc2_value_result_squeezed - old_value, -self.value_clip_param, self.value_clip_param)
        v_loss_unclipped = torch.square(reward - fc2_value_result_squeezed)
        v_loss_clipped = torch.square(reward - v_clipped)
        self.value_cost = 0.5 * torch.mean(torch.maximum(v_loss_unclipped, v_loss_clipped), dim=0)

        epsilon = 1e-5
        label_probability_list = []
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(
                    label_list[task_index].long(), self.label_size_list[task_index])
                legal_max_mask = (1 - legal_action_flag_list[task_index]) * boundary
                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(label_result[task_index] - legal_max_mask, dim=1, keepdim=True).values,
                    -boundary, 1)
                label_exp_logits = (legal_action_flag_list[task_index]
                                    * torch.exp(label_logits_subtract_max) + self.min_policy)
                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                ratio = torch.exp(policy_log_p - old_policy_log_p)

                surr1 = ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                clipped = torch.minimum(surr1, surr2)
                # dual-clip：advantage<0 时损失下界 = dual_clip_param * advantage
                dual = torch.maximum(clipped, self.dual_clip_param * advantage)
                surrogate = torch.where(advantage < 0, dual, clipped)

                temp_policy_loss = -torch.sum(
                    surrogate * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))
                self.policy_cost = self.policy_cost + temp_policy_loss

        current = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp = -torch.sum(
                    label_probability_list[current] * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current] + epsilon), dim=1)
                temp = -torch.sum(temp * weight_list[task_index].float() * frame_is_train) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))
                entropy_loss_list.append(temp)
                current += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0))

        self.entropy_cost = torch.tensor(0.0)
        for e in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + e
        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()
