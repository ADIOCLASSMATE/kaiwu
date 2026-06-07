# CLAUDE.md - 腾讯开悟王者荣耀1v1强化学习实训项目

## 项目概述

本项目是腾讯开悟（Tencent AI Arena）强化学习开发框架下的**王者荣耀 1v1 实训项目 baseline**。使用 PPO 算法在墨家机关道地图上进行 1v1 对战，任务目标是摧毁对手防御塔。

### 环境简述

- **英雄池**：鲁班七号(112)、狄仁杰(133)、公孙离(199)
- **终止条件**：一方防御塔被摧毁 / 超过20000帧 / 异常
- **评分**：获胜方得1分，失败/超时/异常均不得分
- **动作空间**：分为多级标签（12+16+16+16+16+9），含合法动作掩码

## 项目结构

```
kaiwu/
├── agent_ppo/              # PPO 智能体完整实现（参考/基线）
│   ├── agent.py            # 智能体核心：predict/exploit/learn/save/load
│   ├── algorithm/algorithm.py  # PPO 算法：loss 计算与训练 step
│   ├── model/model.py      # 网络结构：MLP + LSTM + 多头输出
│   ├── workflow/train_workflow.py  # 训练工作流：对局循环与样本收集
│   ├── data_dump.py        # 可选导出真实 rollout 样本和完整 raw frame
│   ├── feature/
│   │   ├── definition.py   # 数据结构定义：ObsData/ActData/SampleData/FrameCollector
│   │   ├── reward_process.py   # 奖励管理：塔血量、前进奖励（零和）
│   │   └── feature_process/    # 特征工程：英雄特征 + 防御塔特征 + 归一化
│   └── conf/
│       ├── conf.py         # 模型/算法超参数与数据集维度配置
│       ├── train_env_conf.toml  # 环境配置（对手类型、阵容等）
│       └── monitor_builder.py   # 监控面板配置
│
├── agent_diy/              # 自定义智能体模板（骨架代码，待填充）
│   ├── agent.py            # 仅有框架，需要实现模型推理和数据处理
│   ├── algorithm/algorithm.py  # 空壳
│   ├── model/model.py      # 空壳
│   ├── workflow/train_workflow.py  # 仅导入框架，需实现完整训练流程
│   ├── feature/definition.py  # 仅有数据结构声明，无完整实现
│   └── conf/               # 基础配置
│
├── conf/                   # 全局配置
│   ├── configure_app.toml  # 训练超参数（replay buffer、batch size、预加载等）
│   ├── app_conf_hok1v1.toml    # RL Helper / Policy Builder 配置
│   └── algo_conf_hok1v1.toml   # 算法→Agent/Workflow 模块映射
│
├── train_test.py           # 代码正确性验证脚本
└── kaiwu.json              # 模型池配置
```

## 核心架构

### 框架分层

本项目基于腾讯开悟分布式强化学习框架，使用 `kaiwudrl` 库：

```
智能体(Agent) ←→ 环境(Env) ←→ 样本处理(Sample) ←→ 训练(Learner)
```

框架配置项在 `conf/algo_conf_hok1v1.toml` 中指定各个模块的类路径。

### Agent 接口 (继承 `kaiwudrl.interface.agent.BaseAgent`)

```
agent.py 中的核心方法调用链:
reset() → observation_process() → _model_inference() → action_process() → predict()/exploit()
                                                                                  ↓
learn() ← sample_process() ← FrameCollector ← build_frame()
```

关键方法：
- **`predict(obs)`** — 训练时调用，随机采样动作
- **`exploit(obs)`** — 评估时调用，取最大概率动作
- **`learn(samples)`** — 接收 SampleData 列表进行训练
- **`save_model(path, id)` / `load_model(path, id)`** — 模型持久化
- **`init_config(config_data)`** — 根据阵容选择召唤师技能

### PPO 模型结构

```
输入 feature_vec (10维) → concat_mlp(10→256→256) → 多头输出
                                                    ├── label_mlp[0]→12维 (主标签/英雄选择)
                                                    ├── label_mlp[1]→16维
                                                    ├── label_mlp[2]→16维
                                                    ├── label_mlp[3]→16维
                                                    ├── label_mlp[4]→16维
                                                    ├── label_mlp[5]→9维 (目标选择，依赖于label[0])
                                                    └── value_mlp→1维 (状态价值)
```

注意：模型包含 LSTM 结构（`lstm_unit_size=512`），但 forward 中实际未使用 LSTM 推理（仅保留了 hidden/cell 的传递拼接）。

### 样本数据流

1. `env.step()` → observation（含 frame_state, legal_action）
2. `FeatureProcess.process_feature()` → 提取英雄特征(3维) + 防御塔特征(7维) = 10维特征向量
3. `Agent.predict()` → 产生 ActData（含 action, prob, value, lstm_hidden/cell）
4. `build_frame()` → 构建 Frame（拼接 feature + legal_action + reward + action + prob + value 等）
5. `FrameCollector` → 收集 LSTM_TIME_STEPS=16 帧后，打包为一个 SampleData 送训练

### 真实环境可用信息

不要把 PPO baseline 的 10 维特征误认为环境的全部信息。真实 `env.reset()` / `env.step()` 返回的是结构化游戏状态；baseline 只是从中取了很小一部分用于快速跑通 PPO。

`env.step(actions)` 返回 `(env_reward, env_obs)`。`env_obs` 顶层通常包含：

- `frame_no`：当前环境帧号
- `observation`：按智能体编号划分的观测字典，例如 `observation["0"]` / `observation["1"]`
- `terminated`：正常结束标记
- `truncated`：异常或超时中断标记
- `extra_info`：环境额外信息，本项目主要是 `result_code` 和 `result_message`

每个智能体的 `observation[agent_id]` 可用信息包括：

- `player_id`：英雄运行时 ID
- `camp` / `player_camp`：所属阵营，实际字段名以平台返回为准
- `legal_action`：合法动作掩码
- `sub_action_mask`：不同 button 对应的子动作掩码
- `frame_state`：当前帧完整结构化状态
- `score` / `win`：分数或胜负信息，通常终局时才有明确意义
- `reward`：代码侧 reward manager 计算后写入的奖励分项，不是环境天然字段

`frame_state` 是最重要的信息来源，包含：

- `hero_states`：双方英雄状态
- `npc_states`：小兵、防御塔、水晶、野怪等 NPC 状态
- `bullets`：技能弹道/子弹状态
- `cakes`：神符等功能物件，1v1 中可能为空
- `frame_action`：帧事件，主要用于死亡事件和伤害/收益明细
- `map_state`：地图状态，1v1 默认基本不使用

英雄和 NPC 状态里可挖掘的字段很多，包括：

- 身份与阵营：`config_id`、`runtime_id`、`actor_type`、`sub_type`、`camp`
- 位置与朝向：`location`、`forward`
- 生存和战斗属性：`hp`、`max_hp`、`attack_range`、`attack_target`、`phy_atk`、`phy_def`、`mgc_atk`、`mgc_def`、`mov_spd`、`atk_spd`
- 资源与成长：`level`、`exp`、`money`、`money_cnt`、`kill_income`
- 技能/装备/Buff：`skill_state`、`equip_state`、`buff_state`、`passive_skill`
- 视野与控制：`camp_visible`、`sight_area`、`abilities`、`is_in_grass`
- 战斗统计：`kill_cnt`、`dead_cnt`、`assist_cnt`、`total_hurt`、`total_hurt_to_hero`、`total_be_hurt_by_hero`
- 事件细节：`hit_target_info`、`take_hurt_infos`、`hurt_hero_info`、`real_cmd`

子结构中尤其值得关注：

- `SkillSlotState`：技能 ID、槽位、等级、是否可用、CD、命中次数、多段技能信息
- `EquipSlot`：装备 ID、价格、数量、主动/被动技能
- `BuffSkillState` / `BuffMarkState`：Buff ID、生效次数、印记层数
- `Bullet`：来源 actor、技能槽、技能 ID、当前位置
- `FrameAction.dead_action`：死亡对象、击杀者、助攻者、伤害与收益明细
- `CmdPkg`：英雄实际执行的移动、普攻、技能、买装等指令

当前 baseline 实际只用了：

- 己方英雄 3 维：是否存活、位置 x、位置 z
- 敌方防御塔 7 维：是否存活、阵营、绝对位置、相对位置、血量比例
- `legal_action` / `sub_action_mask`：用于动作采样和 loss mask
- `reward_process.py` 中的 `tower_hp_point` 和 `forward`

因此，后续做特征工程时可以优先考虑：双方英雄完整属性、技能 CD 与可用性、经济/等级差、兵线状态、防御塔血量与距离、子弹/技能飞行物、Buff/印记、伤害事件、真实执行指令、击杀/死亡事件、视野可见性。

### 数据导出与边界

`agent_ppo/data_dump.py` 支持两类文本导出，方便在平台 IDE 无法下载文件时逐个打开复制：

- `KAIWU_DUMP_DATASET=1`：导出 PPO `SampleData`，目录默认 `real_game_dataset/`。这是可直接喂当前 PPO loss 的 4144 维样本，但强绑定当前模型结构、特征工程、旧策略概率、value 和 advantage。
- `KAIWU_DUMP_RAW_FRAMES=1`：导出完整 raw frame，目录默认 `real_game_raw_frames/`。这是更接近环境本体的结构化轨迹，适合挖新特征、做行为克隆或分析字段覆盖。

离线导出的 raw frame 不是可交互模拟环境。它只记录历史轨迹 `(state, action, next_state, reward)`，无法回答“同一状态下如果新模型选择另一个动作会发生什么”。因此它不能替代 `env.step()` 做在线 RL，也不能直接作为完整游戏引擎。可以用它做：

- 行为克隆：学习历史策略的 `observation -> action`
- 离线数据分析：找更有价值的特征和奖励项
- 小规模 loss/entropy smoke test：确认训练管线能稳定跑
- 世界模型研究：近似学习 `(state, action) -> next_state`，但需要远大于当前小样本规模的数据

如果目标是训练强模型，仍应以平台真实环境 rollout 为主；离线数据主要用于预训练、调试和特征工程。

### 奖励设计

在 `reward_process.py` 中定义，采用**零和**设计：
- **tower_hp_point** (权重5.0)：己方塔血量比例 - 对方塔血量比例
- **forward** (权重0.01)：英雄是否在向敌方塔移动（仅血量>99%时计算）

奖励添加时间衰减因子：`reward *= 0.6^(frame_no / TIME_SCALE_ARG)`

### 对手类型

在 `train_env_conf.toml` 中配置：
- **selfplay**：自对弈，双方都用最新模型
- **common_ai**：规则AI，不需要加载模型
- **自定义模型ID**：加载模型池中的指定模型

## 关键框架依赖

- `kaiwudrl` — 腾讯开悟分布式强化学习框架
- `tools.env_conf_manager.EnvConfManager` — 环境配置管理
- `common_python.utils.workflow_disaster_recovery` — 容灾处理
- `common_python.utils.common_func.Frame` / `create_cls` — 数据结构工具

## 远程 WebIDE 操作

本课程 WebIDE 容器禁止普通公网出站访问。远程操作只使用 VS Code 端口代理：在 WebIDE 内启动本地服务 `127.0.0.1:8765`，本机通过 `https://tencentarena.com/p5/ide/<experiment_id>/proxy/8765/` 访问。

1. 本地生成启动脚本：

   ```bash
   python3 script/kaiwu_remote.py bootstrap-proxy-server-command
   ```

2. 将 `script/bootstrap-proxy-server.remote.sh` 内容粘贴到 Kaiwu WebIDE 终端执行，启动 `127.0.0.1:8765` 上的 `proxy_env_server.py`。

3. 本地通过平台代理操作远程：

   ```bash
   python3 script/kaiwu_remote.py proxy-health
   python3 script/kaiwu_remote.py proxy-command --cmd 'pwd && hostname'
   python3 script/kaiwu_remote.py sync
   ```

`train_test.py` 应从 `/data/projects/hok1v1` 执行，而不是直接从 `/workspace/code` 执行，因为平台自带的 `tools/` 目录在前者下。

## 开发注意事项

- `agent_ppo` 是完整可运行的 PPO 基线实现
- `agent_diy` 是空模板，所有方法体都是 `pass` 或返回空值，需要自行填充
- 代码中大量使用中英双语注释，这是项目规范
- Torch 模型使用 `channels_last` 内存格式以优化性能
- LSTM 时间步固定为 16 帧，一个完整训练样本包含 16 帧拼接
- 训练脚本入口：`train_test.py`，通过修改 `algorithm_name` 切换 ppo/diy
- 合法动作通过 `_legal_soft_max` 和 `_legal_sample` 来确保输出动作有效
- 模型文件命名规范：`model.ckpt-{id}.pkl`，框架通过文件名解析模型ID
