#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

增强版模型（v2）：

  实体编码：按「类型」共享投影(hero/structure/minion 各 1 个) → 拼接 R 个 register
  token → pre-LN + AdaLN-Zero Transformer(条件 = type×camp，逐 token 调制
  scale/shift/gate) → 取 register token 作为学习式池化向量，拼全局特征 → LSTM(256)。

  动作输出：
    - label[0..4]：button / 方向，沿用 MLP(LSTM 输出) 头。
    - label[5]   ：target 选择，改为 pointer —— query=LSTM 输出，key=各 target 槽
      对应实体的 transformer 输出（槽 None/Monster 用可学习 null key），logits =
      query·key / sqrt(d)。这样逐实体隐状态被 target 头直接消费，而非池化丢弃。

设计要点（相对 v1）：
  - 取消「加性 type embedding」：类型身份由 (a) 按类型共享的投影、(b) AdaLN 的
    per-(type×camp) 条件注入；比纯加性更具表达力（可缩放方差 + 门控残差）。
  - 取消 masked-mean 池化与「单独拼 main_hero」：改为 register token 学习式池化，
    重要实体（含敌英雄）由注意力自然加权，main_hero 的 ego 身份由其 AdaLN 条件
    (hero×ego) 与 target 头的 Self 槽保留。
  - present 解耦：token 第 0 维 exists 仅用于构造 key_padding_mask（屏蔽空槽）；
    可见性/存活/消失时长是普通特征，雾中实体不再被强制清零。

与 learner / 环境的契约（不变）：
  - forward(inference=True) 返回 [logits(85), value(1), lstm_cell, lstm_hidden]，
    logits 切分顺序 = LABEL_SIZE_LIST = [12,16,16,16,16,9]。
  - compute_loss 与基线签名一致（dual-clip / value-clip / adv-norm PPO）。
  - 训练路径 B×T，推理路径 T=1（set_eval_mode 把 lstm_time_steps 置 1）。
"""

import torch
import torch.nn as nn
from torch.nn import ModuleDict, ModuleList
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


class AdaLNBlock(nn.Module):
    """pre-LN + AdaLN-Zero Transformer block。

    条件按「逐 token 的 (type×camp) / register」索引，提供 per-condition 的
    scale(γ)、shift(β)、gate(α)。AdaLN-Zero：全部 0 初始化 → block 初始为恒等，
    训练稳定（DiT 同款）。
    """

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
        # [γ1, β1, gate1, γ2, β2, gate2]，逐 condition 一行，0 初始化（AdaLN-Zero）
        self.mod_table = nn.Parameter(torch.zeros(n_cond, 6 * d_model))

    def forward(self, x, cond_idx, key_padding_mask):
        # x: (B, S, D); cond_idx: (S,) long; key_padding_mask: (B, S) bool, True=屏蔽
        mod = self.mod_table[cond_idx]                       # (S, 6D)
        g1, b1, k1, g2, b2, k2 = mod.chunk(6, dim=-1)        # each (S, D)
        g1, b1, k1 = g1.unsqueeze(0), b1.unsqueeze(0), k1.unsqueeze(0)
        g2, b2, k2 = g2.unsqueeze(0), b2.unsqueeze(0), k2.unsqueeze(0)

        h = self.norm1(x) * (1.0 + g1) + b1
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + k1 * attn_out

        h = self.norm2(x) * (1.0 + g2) + b2
        x = x + k2 * self.mlp(h)
        return x


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

        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])

        # ---- token 元信息 ----
        self.token_segments = FeatureConfig.TOKEN_SEGMENTS
        self.num_tokens = FeatureConfig.NUM_TOKENS
        self.token_feature_dim = FeatureConfig.TOKEN_FEATURE_DIM
        self.global_dim = FeatureConfig.GLOBAL_DIM

        self.embed_dim = Config.EMBED_DIM
        self.n_register = Config.N_REGISTER

        # 展开每个 token 的 (type_key, start, dim) 与 token 顺序的 type_key 列表
        self.token_layout = []
        self.token_keys = []
        off = 0
        for type_key, dim, count in self.token_segments:
            for _ in range(count):
                self.token_layout.append((type_key, off, dim))
                self.token_keys.append(type_key)
                off += dim
        assert off == self.token_feature_dim

        # ---- 按「类型」共享的投影（hero/structure/minion 各一个）----
        self.proj_key_of = dict(FeatureConfig.TYPE_OF)
        proj_in = {}
        for type_key, dim, count in self.token_segments:
            proj_in[self.proj_key_of[type_key]] = dim   # 同类型 dim 一致
        self.entity_proj = ModuleDict({
            pk: make_fc_layer(in_dim, self.embed_dim) for pk, in_dim in proj_in.items()
        })

        # ---- AdaLN 条件索引：register 用独立条件 ----
        cond_keys = list(FeatureConfig.COND_KEYS)
        self.cond_key_to_idx = {k: i for i, k in enumerate(cond_keys)}
        self.register_cond_idx = len(cond_keys)          # register 专属条件
        n_cond = len(cond_keys) + 1
        cond_idx = [self.register_cond_idx] * self.n_register + \
                   [self.cond_key_to_idx[k] for k in self.token_keys]
        self.register_buffer("cond_idx", torch.tensor(cond_idx, dtype=torch.long))

        # ---- register tokens（学习式池化）----
        self.register_tokens = nn.Parameter(torch.zeros(self.n_register, self.embed_dim))
        nn.init.normal_(self.register_tokens, std=0.02)

        # ---- pre-LN + AdaLN-Zero Transformer ----
        ffn_dim = self.embed_dim * Config.FFN_MULT
        self.blocks = ModuleList([
            AdaLNBlock(self.embed_dim, Config.N_HEADS, ffn_dim, n_cond)
            for _ in range(Config.N_LAYERS)
        ])

        # ---- 池化向量 + 全局 → LSTM 输入 ----
        self.global_proj = MLP(
            [self.global_dim, Config.GLOBAL_PROJ_DIM, Config.GLOBAL_PROJ_DIM],
            "global_proj", non_linearity_last=True)
        fused_dim = self.embed_dim * self.n_register + Config.GLOBAL_PROJ_DIM
        self.fuse_mlp = MLP([fused_dim, self.lstm_unit_size], "fuse_mlp", non_linearity_last=True)

        self.lstm = nn.LSTM(
            input_size=self.lstm_unit_size, hidden_size=self.lstm_unit_size,
            num_layers=1, bias=True, batch_first=True)

        # ---- 动作头：label[0..4] 用 MLP，label[5] 用 pointer ----
        self.n_categorical_heads = len(self.label_size_list) - 1   # 前 5 个头
        self.label_mlp = ModuleDict({
            "hero_label{0}_mlp".format(i): MLP(
                [self.lstm_unit_size] + Config.LABEL_HEAD_HIDDEN_DIMS + [self.label_size_list[i]],
                "hero_label{0}_mlp".format(i))
            for i in range(self.n_categorical_heads)
        })
        self.value_mlp = MLP(
            [self.lstm_unit_size] + Config.VALUE_HEAD_HIDDEN_DIMS + [1],
            "hero_value_mlp")

        # ---- pointer target 头 ----
        self._build_target_pointer()

    def _build_target_pointer(self):
        """根据 TARGET_SLOT_DESC 建立 target 槽 → token 的映射 + null key。"""
        desc = FeatureConfig.TARGET_SLOT_DESC
        key_positions = {}
        for i, k in enumerate(self.token_keys):
            key_positions.setdefault(k, []).append(i)
        counters = {k: 0 for k in key_positions}

        tgt_idx, tgt_real_slots, tgt_null_slots = [], [], []
        for slot, (name, key) in enumerate(desc):
            if key is None:
                tgt_null_slots.append(slot)
            else:
                toks = key_positions[key]
                c = counters[key]
                counters[key] += 1
                assert c < len(toks), "target 槽 '%s' 需要的 token 超过该 type_key 的数量" % name
                tgt_idx.append(toks[c])
                tgt_real_slots.append(slot)

        self.num_target_slots = len(desc)
        self.register_buffer("tgt_idx", torch.tensor(tgt_idx, dtype=torch.long))
        self.register_buffer("tgt_real_slots", torch.tensor(tgt_real_slots, dtype=torch.long))
        self.register_buffer("tgt_null_slots", torch.tensor(tgt_null_slots, dtype=torch.long))

        self.target_query_proj = make_fc_layer(self.lstm_unit_size, self.embed_dim)
        n_null = len(tgt_null_slots)
        self.null_keys = nn.Parameter(torch.zeros(n_null, self.embed_dim))
        nn.init.normal_(self.null_keys, std=0.02)

    # ---- 实体编码：feature -> (fused state, per-entity transformer 输出) ----
    def _encode(self, feature_vec):
        bt = feature_vec.shape[0]
        token_part = feature_vec[:, :self.token_feature_dim]
        global_part = feature_vec[:, self.token_feature_dim:]

        embeds = []
        exists_list = []
        for type_key, start, dim in self.token_layout:
            seg = token_part[:, start:start + dim]                 # (bt, dim)
            exists = (seg[:, 0:1] > 0.5).float()                   # 第 0 位 = exists
            proj = self.entity_proj[self.proj_key_of[type_key]](seg)
            embeds.append(proj.unsqueeze(1))
            exists_list.append(exists)
        entity_tokens = torch.cat(embeds, dim=1)                   # (bt, N, D)
        exists_mat = torch.cat(exists_list, dim=1)                 # (bt, N)

        reg = self.register_tokens.unsqueeze(0).expand(bt, -1, -1)  # (bt, R, D)
        x = torch.cat([reg, entity_tokens], dim=1)                 # (bt, R+N, D)

        # key_padding_mask：register 永不屏蔽；entity 由 exists 决定（屏蔽空槽）
        reg_mask = torch.zeros(bt, self.n_register, dtype=torch.bool, device=x.device)
        ent_mask = exists_mat < 0.5
        key_padding_mask = torch.cat([reg_mask, ent_mask], dim=1)  # (bt, R+N)

        for blk in self.blocks:
            x = blk(x, self.cond_idx, key_padding_mask)

        reg_out = x[:, :self.n_register, :].reshape(bt, -1)        # (bt, R*D)
        entity_out = x[:, self.n_register:, :]                     # (bt, N, D)

        global_embed = self.global_proj(global_part)               # (bt, G)
        state = self.fuse_mlp(torch.cat([reg_out, global_embed], dim=1))
        return state, entity_out

    def _target_pointer(self, query_in, entity_out):
        # query_in: (bt, lstm_unit); entity_out: (bt, N, D) -> logits: (bt, num_target_slots)
        bt = entity_out.shape[0]
        d = self.embed_dim
        q = self.target_query_proj(query_in)                       # (bt, D)
        keys = q.new_zeros(bt, self.num_target_slots, d)
        keys[:, self.tgt_real_slots, :] = entity_out[:, self.tgt_idx, :]
        keys[:, self.tgt_null_slots, :] = self.null_keys.unsqueeze(0).expand(bt, -1, -1)
        logits = (keys * q.unsqueeze(1)).sum(dim=-1) / (d ** 0.5)  # (bt, num_target_slots)
        return logits

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        state, entity_out = self._encode(feature_vec)              # (bt,H), (bt,N,D)

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
        for i in range(self.n_categorical_heads):                 # label[0..4]
            result_list.append(self.label_mlp["hero_label{0}_mlp".format(i)](flat))
        result_list.append(self._target_pointer(flat, entity_out))  # label[5] = pointer
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