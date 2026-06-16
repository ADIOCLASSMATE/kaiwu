#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import random
import tomllib
from agent_diy.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_diy.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


TRAIN_ENV_CONFIG_PATH = "agent_diy/conf/train_env_conf.toml"


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path=TRAIN_ENV_CONFIG_PATH,
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie, 199:Arli)
    # 阵容迭代器 (112:鲁班, 133:狄仁杰, 199:公孙离)
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112, 133, 199])

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.train_opponent_mix = self._load_train_opponent_mix(TRAIN_ENV_CONFIG_PATH)

    def _load_train_opponent_mix(self, config_path):
        default = {
            "enable": False,
            "selfplay": 1.0,
            "common_ai": 0.0,
            "model_pool": 0.0,
        }
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            self.logger.info(f"failed to load train opponent mix config: {exc}; disabled")
            return default

        mix = data.get("episode", {}).get("train_opponent_mix", {})
        if not mix:
            return default

        cfg = {
            "enable": bool(mix.get("enable", default["enable"])),
            "selfplay": max(0.0, float(mix.get("selfplay", default["selfplay"]))),
            "common_ai": max(0.0, float(mix.get("common_ai", default["common_ai"]))),
            "model_pool": max(0.0, float(mix.get("model_pool", default["model_pool"]))),
        }
        if cfg["selfplay"] + cfg["common_ai"] + cfg["model_pool"] <= 0:
            cfg["enable"] = False
        self.logger.info(f"train opponent mix config: {cfg}")
        return cfg

    def _select_train_opponent_agent(self, configured_opponent_agent, is_eval, is_train_test):
        if is_eval or is_train_test or not self.train_opponent_mix.get("enable", False):
            return configured_opponent_agent

        choices = []
        weights = []
        if self.train_opponent_mix.get("selfplay", 0.0) > 0:
            choices.append("selfplay")
            weights.append(self.train_opponent_mix["selfplay"])
        if self.train_opponent_mix.get("common_ai", 0.0) > 0:
            choices.append("common_ai")
            weights.append(self.train_opponent_mix["common_ai"])
        if self.train_opponent_mix.get("model_pool", 0.0) > 0:
            candidate_models = get_valid_model_pool(self.logger)
            if candidate_models:
                choices.append(str(random.choice(candidate_models)))
                weights.append(self.train_opponent_mix["model_pool"])
            else:
                self.logger.info("train opponent mix skips model_pool because kaiwu.json model_pool is empty")

        if not choices:
            return configured_opponent_agent
        selected = random.choices(choices, weights=weights, k=1)[0]
        self.logger.info(f"train opponent selected: {selected}")
        return selected

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        monitor_side = self.env_conf_manager.get_monitor_side()
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def _episode_outcome(self, observation, monitor_side):
        """从最后一帧提取 monitor_side 的终局结果指标。

        胜负判定按外塔(sub_type=21)存活：己方塔存活且敌方塔被毁 → 胜(1)；
        反之为负(0)；双方塔都在（超时）→ 平(0.5)。同时上报英雄终局养成数据。
        """
        out = {}
        TOWER_SUBTYPE = 21
        try:
            fs = observation[str(monitor_side)]["frame_state"]
        except (KeyError, TypeError):
            return out

        my_hero = None
        my_camp = observation[str(monitor_side)].get("camp")
        for h in fs.get("hero_states", []):
            if h.get("camp") == my_camp:
                my_hero = h

        if my_hero is not None:
            out["final_level"] = my_hero.get("level", 0)
            out["final_money"] = my_hero.get("money", 0)
            out["kill_cnt"] = my_hero.get("kill_cnt", 0)
            out["dead_cnt"] = my_hero.get("dead_cnt", 0)
            mh = my_hero.get("max_hp", 0) or 1
            out["final_hp_ratio"] = round(my_hero.get("hp", 0) / mh, 3)

        own_tower_alive = enemy_tower_alive = False
        for npc in fs.get("npc_states", []):
            if npc.get("sub_type") != TOWER_SUBTYPE:
                continue
            alive = npc.get("hp", 0) > 0
            if npc.get("camp") == my_camp:
                own_tower_alive = own_tower_alive or alive
            else:
                enemy_tower_alive = enemy_tower_alive or alive

        if own_tower_alive and not enemy_tower_alive:
            out["win"] = 1.0
        elif enemy_tower_alive and not own_tower_alive:
            out["win"] = 0.0
        else:
            out["win"] = 0.5
        return out

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)

            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)

            # Start a new environment
            # 启动新对局，返回初始环境状态

            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation, is_eval=is_eval)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # ---- 监控统计累加器（仅追踪 monitor_side，按局聚合）----
            # reward 子项累计
            reward_item_sum = {}          # 各 reward 子项的整局累计值

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])
                        agent.record_episode_step(
                            episode=self.episode_cnt,
                            frame_no=frame_no,
                            observation=observation[str(index)],
                            action=actions[index],
                            is_eval=is_eval,
                        )

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            # 距离整形：基于做出决策时的 frame_state 计算越程攻击惩罚
                            agent.reward_manager.set_distance_penalty(
                                actions[index],
                                observation[str(index)]["frame_state"],
                            )
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]

                        # ---- 监控统计：只聚合 monitor_side ----
                        if i == monitor_side:
                            for k, v in reward.items():
                                if k == "reward_sum":
                                    continue
                                reward_item_sum[k] = reward_item_sum.get(k, 0.0) + v

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    # 终局奖励必须写入最后一帧样本，且不参与 shaping 的时间衰减。
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not do_sample:
                            continue
                        terminal_bonus = agent.reward_manager.apply_terminal_outcome(
                            observation[str(i)]["reward"],
                            observation[str(i)]["frame_state"],
                            observation[str(i)].get("win"),
                        )
                        reward_sum_list[i] += terminal_bonus
                        if i == monitor_side:
                            reward_item_sum["terminal"] = (
                                reward_item_sum.get("terminal", 0.0)
                                + observation[str(i)]["reward"]["terminal"]
                            )
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            # 整局聚合指标（训练/评估均上报，便于横向对比）。
                            monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                            monitor_data["episode_len"] = frame_no

                            # reward 子项分解：看清回报由什么驱动，而非只看总和。
                            for k, v in reward_item_sum.items():
                                monitor_data["rwd_" + k] = round(v, 3)

                            # reward 健康度：越程动作与挂机触发的逐局统计。
                            monitor_data.update(
                                self.agents[monitor_side].reward_manager.consume_monitor_stats()
                            )

                            # 对局结果指标：从最后一帧取 monitor_side 英雄的终局状态。
                            outcome = self._episode_outcome(observation, monitor_side)
                            monitor_data.update(outcome)

                            # 特征健康度：整局聚合指标（NaN/Inf/负值检测 + 各 token 组统计）
                            feat_stats = self.agents[monitor_side].get_feature_stats()
                            monitor_data.update(feat_stats)

                            # 样本有效性：本局 is_train=1 占比（诊断策略梯度被掩码
                            # 削弱的程度；偏低说明无效帧太多）。仅训练对局有意义。
                            if not is_eval and self.do_samples[monitor_side]:
                                monitor_data["is_train_rate"] = round(
                                    frame_collector.is_train_rate(monitor_side), 4
                                )

                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def reset_agents(self, observation, is_eval=False):
        configured_opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
        opponent_agent = self._select_train_opponent_agent(
            configured_opponent_agent,
            is_eval,
            is_train_test,
        )

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    opponent_model_id = "latest" if is_eval or is_train_test else "random"
                    self.logger.info(
                        f"selfplay opponent loads model_id={opponent_model_id}, eval={is_eval}"
                    )
                    agent.load_model(id=opponent_model_id)
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])
