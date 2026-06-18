#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
from torch.nn import ModuleDict, ModuleList

import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config
from agent_diy.conf.conf import FeatureConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
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
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        # NETWORK DIM
        # 网络维度
        self.hero_data_len = sum(Config.data_shapes[0])
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        self.encoder_output_dim = Config.PPO_ENCODER_OUTPUT_DIM
        self.n_categorical_heads = len(self.label_size_list) - 1

        self.concat_mlp = MLP(
            [self.feature_dim, self.encoder_output_dim, self.encoder_output_dim],
            "concat_mlp",
            non_linearity=nn.GELU,
            non_linearity_last=True,
        )

        self.token_segments = FeatureConfig.TOKEN_SEGMENTS
        self.num_tokens = FeatureConfig.NUM_TOKENS
        self.token_feature_dim = FeatureConfig.TOKEN_FEATURE_DIM
        self.global_dim = FeatureConfig.GLOBAL_DIM
        self.embed_dim = Config.EMBED_DIM
        self.n_register = Config.N_REGISTER

        self.token_layout = []
        self.token_keys = []
        offset = 0
        for type_key, dim, count in self.token_segments:
            for _ in range(count):
                self.token_layout.append((type_key, offset, dim))
                self.token_keys.append(type_key)
                offset += dim
        assert offset == self.token_feature_dim

        self.proj_key_of = dict(FeatureConfig.TYPE_OF)
        proj_in = {}
        for type_key, dim, count in self.token_segments:
            proj_in[self.proj_key_of[type_key]] = dim
        self.entity_proj = ModuleDict(
            {
                proj_key: make_fc_layer(in_dim, self.embed_dim)
                for proj_key, in_dim in proj_in.items()
            }
        )
        self.input_norm = ModuleDict(
            {
                proj_key: nn.Identity()
                for proj_key in proj_in
            }
        )

        cond_keys = list(FeatureConfig.COND_KEYS)
        self.cond_key_to_idx = {key: index for index, key in enumerate(cond_keys)}
        self.register_cond_idx = len(cond_keys)
        cond_idx = [self.register_cond_idx] * self.n_register + [
            self.cond_key_to_idx[key] for key in self.token_keys
        ]
        self.register_buffer("cond_idx", torch.tensor(cond_idx, dtype=torch.long))

        self.register_tokens = nn.Parameter(torch.zeros(self.n_register, self.embed_dim))
        nn.init.normal_(self.register_tokens, std=0.02)

        ffn_dim = self.embed_dim * Config.FFN_MULT
        self.blocks = ModuleList(
            [
                AdaLNBlock(self.embed_dim, Config.N_HEADS, ffn_dim, len(cond_keys) + 1)
                for _ in range(Config.N_LAYERS)
            ]
        )

        self.global_proj = MLP(
            [self.global_dim, Config.GLOBAL_PROJ_DIM, Config.GLOBAL_PROJ_DIM],
            "global_proj",
            non_linearity=nn.GELU,
            non_linearity_last=True,
        )
        fused_dim = self.embed_dim * self.n_register + Config.GLOBAL_PROJ_DIM
        self.feature_encoder = MLP(
            [fused_dim, self.encoder_output_dim],
            "feature_encoder",
            non_linearity=nn.GELU,
            non_linearity_last=True,
        )
        self.token_residual_gate = nn.Parameter(
            torch.tensor(float(Config.TOKEN_RESIDUAL_INIT))
        )

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.lstm_input_proj = make_fc_layer(self.encoder_output_dim, self.lstm_unit_size)
        self.lstm_output_proj = make_fc_layer(self.lstm_unit_size, self.encoder_output_dim)
        self.lstm_residual_gate = nn.Parameter(
            torch.tensor(float(Config.LSTM_RESIDUAL_INIT))
        )

        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [self.encoder_output_dim, 256, self.label_size_list[label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(self.n_categorical_heads)
            }
        )
        self.target_base_mlp = MLP(
            [self.encoder_output_dim, 256, self.label_size_list[-1]],
            "hero_label{0}_base_mlp".format(self.n_categorical_heads),
        )
        self.target_pointer_gate = nn.Parameter(
            torch.tensor(float(Config.TARGET_POINTER_INIT))
        )
        self.value_mlp = MLP([self.encoder_output_dim, 256, 1], "hero_value_mlp")

        self._build_target_pointer()

    def _build_target_pointer(self):
        desc = FeatureConfig.TARGET_SLOT_DESC
        key_positions = {}
        for index, key in enumerate(self.token_keys):
            key_positions.setdefault(key, []).append(index)
        counters = {key: 0 for key in key_positions}

        target_indices = []
        real_slots = []
        null_slots = []
        for slot, (_name, key) in enumerate(desc):
            if key is None:
                null_slots.append(slot)
                continue

            token_index = key_positions[key][counters[key]]
            counters[key] += 1
            target_indices.append(token_index)
            real_slots.append(slot)

        self.num_target_slots = len(desc)
        self.register_buffer("tgt_idx", torch.tensor(target_indices, dtype=torch.long))
        self.register_buffer("tgt_real_slots", torch.tensor(real_slots, dtype=torch.long))
        self.register_buffer("tgt_null_slots", torch.tensor(null_slots, dtype=torch.long))

        self.target_query_proj = make_fc_layer(self.encoder_output_dim, self.embed_dim)
        self.target_button_embed = nn.Embedding(self.label_size_list[0], self.embed_dim)
        nn.init.normal_(self.target_button_embed.weight, std=0.02)
        self.null_keys = nn.Parameter(torch.zeros(len(null_slots), self.embed_dim))
        nn.init.normal_(self.null_keys, std=0.02)

    def _encode(self, feature_vec):
        batch_size = feature_vec.shape[0]
        token_part = feature_vec[:, :self.token_feature_dim]
        global_part = feature_vec[:, self.token_feature_dim:]

        embeds = []
        exists_list = []
        for type_key, start, dim in self.token_layout:
            segment = token_part[:, start:start + dim]
            exists = (segment[:, 0:1] > 0.5).float()
            segment = self.input_norm[self.proj_key_of[type_key]](segment)
            projected = self.entity_proj[self.proj_key_of[type_key]](segment)
            embeds.append(projected.unsqueeze(1))
            exists_list.append(exists)

        entity_tokens = torch.cat(embeds, dim=1)
        exists_mat = torch.cat(exists_list, dim=1)
        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        encoded = torch.cat([register_tokens, entity_tokens], dim=1)

        register_mask = torch.zeros(
            batch_size,
            self.n_register,
            dtype=torch.bool,
            device=encoded.device,
        )
        entity_mask = exists_mat < 0.5
        key_padding_mask = torch.cat([register_mask, entity_mask], dim=1)

        for block in self.blocks:
            encoded = block(encoded, self.cond_idx, key_padding_mask)

        register_out = encoded[:, :self.n_register, :].reshape(batch_size, -1)
        entity_out = encoded[:, self.n_register:, :]
        global_embed = self.global_proj(global_part)
        structured_state = self.feature_encoder(torch.cat([register_out, global_embed], dim=1))
        return structured_state, entity_out

    def _target_keys(self, entity_out):
        batch_size = entity_out.shape[0]
        keys = entity_out.new_zeros(batch_size, self.num_target_slots, self.embed_dim)
        keys[:, self.tgt_real_slots, :] = entity_out[:, self.tgt_idx, :]
        keys[:, self.tgt_null_slots, :] = self.null_keys.unsqueeze(0).expand(batch_size, -1, -1)
        return keys

    def _target_pointer_by_button(self, state, entity_out):
        query_base = self.target_query_proj(state)
        button_ids = torch.arange(
            self.label_size_list[0],
            dtype=torch.long,
            device=state.device,
        )
        query = query_base.unsqueeze(1) + self.target_button_embed(button_ids).unsqueeze(0)
        keys = self._target_keys(entity_out)
        return torch.einsum("bqd,btd->bqt", query, keys) / (self.embed_dim ** 0.5)

    def _gather_target_logits(self, target_logits_by_button, button_labels):
        button_labels = button_labels.long().clamp(0, self.label_size_list[0] - 1)
        batch_index = torch.arange(
            target_logits_by_button.shape[0],
            dtype=torch.long,
            device=target_logits_by_button.device,
        )
        return target_logits_by_button[batch_index, button_labels, :]

    def _apply_lstm_residual(self, state, lstm_hidden_init, lstm_cell_init, time_steps):
        batch_time = state.shape[0]
        if batch_time % time_steps != 0:
            raise ValueError(
                "feature batch size {} is not divisible by LSTM time steps {}".format(
                    batch_time,
                    time_steps,
                )
            )
        batch_size = batch_time // time_steps

        lstm_input = self.lstm_input_proj(state).reshape(
            batch_size,
            time_steps,
            self.lstm_unit_size,
        )
        h0 = lstm_hidden_init.reshape(batch_size, self.lstm_unit_size).unsqueeze(0).contiguous()
        c0 = lstm_cell_init.reshape(batch_size, self.lstm_unit_size).unsqueeze(0).contiguous()
        lstm_out, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        self.lstm_hidden_output = hn
        self.lstm_cell_output = cn

        temporal_state = self.lstm_output_proj(lstm_out.reshape(batch_time, self.lstm_unit_size))
        return state + self.lstm_residual_gate * temporal_state

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        result_list = []

        # Raw MLP keeps direct access to all 561 feature fields; the token
        # encoder adds structured entity relations through a small residual.
        raw_state = self.concat_mlp(feature_vec)
        structured_state, entity_out = self._encode(feature_vec)
        fc_public_result = raw_state + self.token_residual_gate * structured_state
        time_steps = 1 if inference else self.lstm_time_steps
        fc_public_result = self._apply_lstm_residual(
            fc_public_result,
            lstm_hidden_init,
            lstm_cell_init,
            time_steps,
        )

        # output label
        # 输出标签
        for label_index in range(self.n_categorical_heads):
            label_mlp_out = self.label_mlp["hero_label{0}_mlp".format(label_index)](fc_public_result)
            result_list.append(label_mlp_out)

        target_base_logits = self.target_base_mlp(fc_public_result)
        target_pointer_logits = self._target_pointer_by_button(fc_public_result, entity_out)
        target_logits_by_button = (
            target_base_logits.unsqueeze(1)
            + self.target_pointer_gate * target_pointer_logits
        )
        self.target_logits_by_button = target_logits_by_button
        result_list.append(target_logits_by_button)

        # output value
        # 输出价值
        value_result = self.value_mlp(fc_public_result)
        result_list.append(value_result)

        # prepare for infer graph
        # 准备推理图
        target_logits_public = target_logits_by_button.mean(dim=1)
        logits = torch.flatten(
            torch.cat(result_list[:self.n_categorical_heads] + [target_logits_public], 1),
            start_dim=1,
        )
        value = result_list[-1]

        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # loss of value net
        # 值网络的损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        # for entropy loss calculate
        # 用于熵损失计算
        label_probability_list = []

        epsilon = 1e-5

        # policy loss: ppo clip loss
        # 策略损失：PPO剪辑损失
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
                task_logits = label_result[task_index]
                if task_index == self.n_categorical_heads and task_logits.dim() == 3:
                    task_logits = self._gather_target_logits(task_logits, label_list[0])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    task_logits
                    - torch.max(
                        task_logits - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * frame_is_train
                ) / torch.maximum(torch.sum((weight_list[task_index].float()) * frame_is_train), torch.tensor(1.0))

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * frame_is_train)
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


class AdaLNBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_dim: int, n_cond: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            make_fc_layer(d_model, ffn_dim),
            nn.GELU(),
            make_fc_layer(ffn_dim, d_model),
        )

        mod = torch.zeros(n_cond, 6 * d_model)
        gate_init = float(Config.ADALN_GATE_INIT)
        if gate_init != 0.0:
            mod[:, 2 * d_model:3 * d_model] = gate_init
            mod[:, 5 * d_model:6 * d_model] = gate_init
        self.mod_table = nn.Parameter(mod)

    def forward(self, x, cond_idx, key_padding_mask):
        key_padding_mask = key_padding_mask.to(dtype=torch.bool).contiguous()
        mod = self.mod_table[cond_idx]
        g1, b1, k1, g2, b2, k2 = mod.chunk(6, dim=-1)
        g1, b1, k1 = g1.unsqueeze(0), b1.unsqueeze(0), k1.unsqueeze(0)
        g2, b2, k2 = g2.unsqueeze(0), b2.unsqueeze(0), k2.unsqueeze(0)

        hidden = self.norm1(x) * (1.0 + g1) + b1
        attn_out, _ = self.attn(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + k1 * attn_out

        hidden = self.norm2(x) * (1.0 + g2) + b2
        return x + k2 * self.mlp(hidden)
