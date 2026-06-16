#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
import os
import time
from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None, diagnostics=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.train_step = 0
        self.logger = logger
        self.monitor = monitor
        self.diagnostics = diagnostics

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        _input_datas = torch.stack([s.sample for s in list_sample_data]).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data_list[i] = data.reshape(-1).float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [int(np.prod(self.seri_vec_split_shape[0])), int(np.prod(self.seri_vec_split_shape[1]))], dim=1)
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # 梯度范数（裁剪前），检测梯度消失/爆炸
        raw_grad_norm = self._compute_grad_norm()

        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1
        self.scheduler.step(self.train_step)

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info_list.append([i.item() for i in info])
            else:
                _info_list.append(info.item())

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)

            # 梯度范数
            results["grad_norm"] = round(raw_grad_norm, 4)

            # 学习率
            results["learning_rate"] = round(self.optimizer.param_groups[0]["lr"], 8)

            # 各动作头熵（检测策略是否过早坍缩）
            head_names = ["head_0", "head_1", "head_2", "head_3", "head_4", "head_5"]
            if hasattr(self.model, "entropy_cost_list"):
                for name, ent in zip(head_names, self.model.entropy_cost_list):
                    results["entropy_" + name] = round(ent.item(), 4)

            # advantage 统计量（检测 advantage 是否有意义）
            adv_tensor = data_list[2]  # usq_advantage
            results["adv_mean"] = round(adv_tensor.mean().item(), 4)
            results["adv_std"] = round(adv_tensor.std().item(), 4)

            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
        if self.diagnostics is not None and self.diagnostics.enabled:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            diag_metrics = {
                "total_loss": results["total_loss"],
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "entropy_loss": entropy_loss,
                "grad_norm": raw_grad_norm,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self.diagnostics.record_train_step(
                diag_metrics,
                reward=data_list[1],
                advantage=data_list[2],
                value=rst_list[-1],
                grad_norms=self._compute_module_grad_norms(),
            )
        return results

    def _compute_grad_norm(self):
        total_norm = 0.0
        for p in self.parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _compute_module_grad_norms(self):
        groups = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            group = name.split(".", 1)[0]
            groups[group] = groups.get(group, 0.0) + param.grad.data.norm(2).item() ** 2
        return {key: value ** 0.5 for key, value in groups.items()}
