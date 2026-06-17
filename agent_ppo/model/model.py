#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

agent_ppo model with the agent_diy architecture idea adapted to the original
10-dimensional PPO feature contract.

The input feature/reward/sample layout is unchanged:
  - feature[0:3]  = main hero feature token
  - feature[3:10] = enemy tower feature token
  - feature[0:10] is also kept as a compact global feature

The network side mirrors agent_diy's structure: type-specific token projection,
register-token Transformer encoder, recurrent state, MLP actor/value heads, and
a button-conditioned pointer target head. Target slots without a corresponding
entity in the original PPO features use learned null keys.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict, ModuleList

from agent_ppo.conf.conf import Config, DimConfig


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
        non_linearity: nn.Module = nn.GELU,
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
    """pre-LN Transformer block with per-token AdaLN modulation."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, n_cond: int):
        super().__init__()
        self.d_model = d_model
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

        h = self.norm1(x) * (1.0 + g1) + b1
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + k1 * attn_out

        h = self.norm2(x) * (1.0 + g2) + b2
        x = x + k2 * self.mlp(h)
        return x


class PpoFeatureConfig:
    HERO_DIM = 3
    TOWER_DIM = 7
    GLOBAL_DIM = DimConfig.DIM_OF_FEATURE[0]
    TOKEN_SEGMENTS = [
        ("main_hero", HERO_DIM, 1),
        ("enemy_tower", TOWER_DIM, 1),
    ]
    TYPE_OF = {
        "main_hero": "hero",
        "enemy_tower": "structure",
    }
    COND_KEYS = ["main_hero", "enemy_tower"]
    TARGET_SLOT_DESC = [
        ("None", None),
        ("EnemyHero", None),
        ("Self", "main_hero"),
        ("Soldier1", None),
        ("Soldier2", None),
        ("Soldier3", None),
        ("Soldier4", None),
        ("Tower", "enemy_tower"),
        ("Monster", None),
    ]


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
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE
        self.hero_data_len = sum(Config.data_shapes[0])

        self.token_segments = PpoFeatureConfig.TOKEN_SEGMENTS
        self.num_tokens = sum(count for _, _, count in self.token_segments)
        self.global_dim = PpoFeatureConfig.GLOBAL_DIM
        self.embed_dim = Config.EMBED_DIM
        self.n_register = Config.N_REGISTER

        self.token_layout = []
        self.token_keys = []
        off = 0
        for type_key, dim, count in self.token_segments:
            for _ in range(count):
                self.token_layout.append((type_key, off, dim))
                self.token_keys.append(type_key)
                off += dim
        assert off == self.feature_dim

        self.proj_key_of = dict(PpoFeatureConfig.TYPE_OF)
        proj_in = {}
        for type_key, dim, _ in self.token_segments:
            proj_in[self.proj_key_of[type_key]] = dim
        self.entity_proj = ModuleDict({
            pk: make_fc_layer(in_dim, self.embed_dim) for pk, in_dim in proj_in.items()
        })
        self.input_norm = ModuleDict({pk: nn.Identity() for pk in proj_in})

        cond_keys = list(PpoFeatureConfig.COND_KEYS)
        self.cond_key_to_idx = {key: i for i, key in enumerate(cond_keys)}
        self.register_cond_idx = len(cond_keys)
        n_cond = len(cond_keys) + 1
        cond_idx = [self.register_cond_idx] * self.n_register + [
            self.cond_key_to_idx[key] for key in self.token_keys
        ]
        self.register_buffer("cond_idx", torch.tensor(cond_idx, dtype=torch.long))

        self.register_tokens = nn.Parameter(torch.zeros(self.n_register, self.embed_dim))
        nn.init.normal_(self.register_tokens, std=0.02)

        ffn_dim = self.embed_dim * Config.FFN_MULT
        self.blocks = ModuleList([
            AdaLNBlock(self.embed_dim, Config.N_HEADS, ffn_dim, n_cond)
            for _ in range(Config.N_LAYERS)
        ])

        self.global_proj = MLP(
            [self.global_dim, Config.GLOBAL_PROJ_DIM, Config.GLOBAL_PROJ_DIM],
            "global_proj",
            non_linearity_last=True,
        )
        fused_dim = self.embed_dim * self.n_register + Config.GLOBAL_PROJ_DIM
        self.fuse_mlp = MLP([fused_dim, self.lstm_unit_size], "fuse_mlp", non_linearity_last=True)

        self.lstm = nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.n_categorical_heads = len(self.label_size_list) - 1
        self.label_mlp = ModuleDict({
            "hero_label{0}_mlp".format(label_index): MLP(
                [self.lstm_unit_size] + Config.LABEL_HEAD_HIDDEN_DIMS + [self.label_size_list[label_index]],
                "hero_label{0}_mlp".format(label_index),
            )
            for label_index in range(self.n_categorical_heads)
        })
        self.value_mlp = MLP(
            [self.lstm_unit_size] + Config.VALUE_HEAD_HIDDEN_DIMS + [1],
            "hero_value_mlp",
        )

        self._build_target_pointer()

    def _build_target_pointer(self):
        desc = PpoFeatureConfig.TARGET_SLOT_DESC
        key_positions = {}
        for i, key in enumerate(self.token_keys):
            key_positions.setdefault(key, []).append(i)
        counters = {key: 0 for key in key_positions}

        tgt_idx, tgt_real_slots, tgt_null_slots = [], [], []
        for slot, (_, key) in enumerate(desc):
            if key is None:
                tgt_null_slots.append(slot)
                continue
            toks = key_positions[key]
            counter = counters[key]
            counters[key] += 1
            assert counter < len(toks), "target slot needs more tokens than available"
            tgt_idx.append(toks[counter])
            tgt_real_slots.append(slot)

        self.num_target_slots = len(desc)
        self.register_buffer("tgt_idx", torch.tensor(tgt_idx, dtype=torch.long))
        self.register_buffer("tgt_real_slots", torch.tensor(tgt_real_slots, dtype=torch.long))
        self.register_buffer("tgt_null_slots", torch.tensor(tgt_null_slots, dtype=torch.long))

        self.target_query_proj = make_fc_layer(self.lstm_unit_size, self.embed_dim)
        self.target_button_embed = nn.Embedding(self.label_size_list[0], self.embed_dim)
        nn.init.normal_(self.target_button_embed.weight, std=0.02)
        self.null_keys = nn.Parameter(torch.zeros(len(tgt_null_slots), self.embed_dim))
        nn.init.normal_(self.null_keys, std=0.02)

    def _encode(self, feature_vec):
        batch_size = feature_vec.shape[0]
        embeds = []
        exists_list = []
        for type_key, start, dim in self.token_layout:
            seg = feature_vec[:, start:start + dim]
            exists = (seg[:, 0:1] > 0.5).float()
            seg = self.input_norm[self.proj_key_of[type_key]](seg)
            proj = self.entity_proj[self.proj_key_of[type_key]](seg)
            embeds.append(proj.unsqueeze(1))
            exists_list.append(exists)

        entity_tokens = torch.cat(embeds, dim=1)
        exists_mat = torch.cat(exists_list, dim=1)

        reg = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([reg, entity_tokens], dim=1)

        reg_mask = torch.zeros(batch_size, self.n_register, dtype=torch.bool, device=x.device)
        ent_mask = exists_mat < 0.5
        key_padding_mask = torch.cat([reg_mask, ent_mask], dim=1)

        for block in self.blocks:
            x = block(x, self.cond_idx, key_padding_mask)

        reg_out = x[:, :self.n_register, :].reshape(batch_size, -1)
        entity_out = x[:, self.n_register:, :]
        global_embed = self.global_proj(feature_vec)
        state = self.fuse_mlp(torch.cat([reg_out, global_embed], dim=1))
        return state, entity_out

    def _target_keys(self, entity_out):
        batch_size = entity_out.shape[0]
        keys = entity_out.new_zeros(batch_size, self.num_target_slots, self.embed_dim)
        keys[:, self.tgt_real_slots, :] = entity_out[:, self.tgt_idx, :]
        keys[:, self.tgt_null_slots, :] = self.null_keys.unsqueeze(0).expand(batch_size, -1, -1)
        return keys

    def _target_pointer_by_button(self, query_in, entity_out):
        q_base = self.target_query_proj(query_in)
        button_ids = torch.arange(
            self.label_size_list[0],
            dtype=torch.long,
            device=query_in.device,
        )
        q = q_base.unsqueeze(1) + self.target_button_embed(button_ids).unsqueeze(0)
        keys = self._target_keys(entity_out)
        return torch.einsum("bqd,btd->bqt", q, keys) / (self.embed_dim ** 0.5)

    def _gather_target_logits(self, target_logits_by_button, button_labels):
        button_labels = button_labels.long().clamp(0, self.label_size_list[0] - 1)
        batch_index = torch.arange(
            target_logits_by_button.shape[0],
            dtype=torch.long,
            device=target_logits_by_button.device,
        )
        return target_logits_by_button[batch_index, button_labels, :]

    def _actor_logits(self, recurrent_state):
        return [
            self.label_mlp["hero_label{0}_mlp".format(index)](recurrent_state)
            for index in range(self.n_categorical_heads)
        ]

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        state, entity_out = self._encode(feature_vec)

        time_steps = self.lstm_time_steps
        batch_time = state.shape[0]
        batch_size = batch_time // time_steps
        lstm_in = state.reshape(batch_size, time_steps, self.lstm_unit_size)

        h0 = lstm_hidden_init.reshape(batch_size, self.lstm_unit_size).unsqueeze(0).contiguous()
        c0 = lstm_cell_init.reshape(batch_size, self.lstm_unit_size).unsqueeze(0).contiguous()

        lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))
        self.lstm_hidden_output = hn
        self.lstm_cell_output = cn
        flat = lstm_out.reshape(batch_time, self.lstm_unit_size)

        result_list = self._actor_logits(flat)
        target_logits_by_button = self._target_pointer_by_button(flat, entity_out)
        self.target_logits_by_button = target_logits_by_button
        result_list.append(target_logits_by_button)

        value_result = self.value_mlp(flat)
        result_list.append(value_result)

        target_logits_public = target_logits_by_button.mean(dim=1)
        logits = torch.flatten(
            torch.cat(result_list[:self.n_categorical_heads] + [target_logits_public], dim=1),
            start_dim=1,
        )
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
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list):3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list):3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]
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

        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        epsilon = 1e-5
        boundary = seri_vec.new_tensor(1e20)
        label_probability_list = []
        self.policy_cost = seri_vec.new_tensor(0.0)

        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                task_logits = label_result[task_index]
                if task_index == len(self.label_size_list) - 1 and task_logits.dim() == 3:
                    task_logits = self._gather_target_logits(task_logits, label_list[0])

                one_hot_actions = nn.functional.one_hot(
                    label_list[task_index].long(),
                    self.label_size_list[task_index],
                )
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
                ratio = torch.exp(policy_log_p - old_policy_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * frame_is_train
                ) / torch.maximum(
                    torch.sum((weight_list[task_index].float()) * frame_is_train),
                    seri_vec.new_tensor(1.0),
                )

                self.policy_cost = self.policy_cost + temp_policy_loss

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
                    temp_entropy_loss * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train),
                    seri_vec.new_tensor(1.0),
                )

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                entropy_loss_list.append(seri_vec.new_tensor(0.0))

        self.entropy_cost = seri_vec.new_tensor(0.0)
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
