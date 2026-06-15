# agent_diy 训练停滞诊断（基于 logs 对应代码版本）

## 一句话结论

不是 reward 太复杂（现在只有 4 项，已经够干净）。问题是**策略梯度的信号
被三个因素叠加削弱到几乎为零**：(1) AdaLN-Zero 让 Transformer 初始为恒等，
实体信息一开始流不到策略头；(2) entropy bonus 相对过大，在弱 advantage 下
主动顶住熵不让它下降；(3) 熵的"高"是相对**合法动作**而言的假象。先按证据
排掉，再决定要不要碰 reward。

## 推翻原分析的一个关键误读

原分析说"熵 = 11.97 ≈ 76% 最大熵 → 策略停在随机"。但最大熵应按**合法动作**
算，不是按 85 维满空间算。

从 `diag_feature_probes` 的真实 `legal_action` 解码（15 帧样本）：

| 头 | 满空间 | 实测平均合法数 |
|---|---|---|
| button(0) | 12 | **4.8** |
| move1-4 | 16 each | ~15 each |
| target(5) | 9 | 取决于场上实体 |

合法-均匀熵 ≈ ln(4.8)+4·ln(15) ≈ 12.4（仅前 5 头）。你观测的 11.97 已经
**接近合法-均匀上限**，不是满空间上限。也就是说策略其实"几乎均匀地在合法
动作里乱选"，离随机初始化没远到哪去——但这恰恰说明梯度没能把它从均匀拉开，
而不是 reward 信号本身的问题。

## 证据链

1. **value_loss 正常（0.78，在动）**：encoder→LSTM→value 这条路是通的，
   特征/前向/GAE 数值口径都对（已核对 `old_value = reward - raw_advantage`
   与 `reward_sum = gae + value` 一致）。问题被隔离到**策略头这一路**。

2. **policy_loss ≈ 0 且熵不降**：value 能学、policy 不能学，差别只在
   policy 这条分支独有的东西——AdaLN gate、entropy bonus、advantage 归一化。

3. **target ordering bug 已修**：`targeting.py` 的 `target_slot_enemy_soldiers`
   已采用「最近4个 → 按 runtime_id 升序」，与 probe 结论一致。这一项排除。

## 三个根因（按确信度 + 改动小到大）

### A. AdaLN-Zero 把实体信息在初期完全 gate 掉（最可能、最该先试）

`AdaLNBlock.mod_table` 全 0 初始化 → 初始时：

```
x = x + gate1 * attn_out   # gate1 = 0 → attention 输出乘 0
x = x + gate2 * mlp(h)     # gate2 = 0 → mlp 输出乘 0
```

整个 Transformer 是**恒等映射**。register pooling 向量与场上实体无关，
pointer 的 key 也只是输入投影本身。策略头此刻基本"看不见"敌我单位。
DiT 里 AdaLN-Zero 能 work 是因为强监督 + 大数据把 gate 顶起来；RL 弱信号
+ 1245 步，gate 可能还几乎没动。

**改法**：gate 不从 0 起，给一个小正初始（identity-ish 但非零）。这样实体
信息从第一步就能以小幅度流到策略头，梯度有路可走。改动仅在 `__init__`。

### B. entropy bonus 在弱 advantage 下过大（一行，可逆）

`BETA_START = 0.025`。在 advantage 信号弱时，entropy 项对 loss 的贡献
（≈0.30）和 policy_cost（≈0.007）完全不在一个量级——**熵 bonus 把策略
死死摁在均匀分布上**。先调到 `0.005` 观察熵是否开始下降。

### C.（次要）advantage 归一化在 is_train 掩码下可能不稳

`compute_loss` 用 masked mean/var 归一化 advantage，但归一化作用于**全张量**
（含被 mask 的位置）。若 `sum(is_train)` 偏小，mean/var 估计噪声大。建议
先在 monitor 加一个 **`is_train` 占比**面板确认有效样本比例；若 >0.5 则
此项不是主因，A/B 足够。

## 建议执行顺序

1. 先加 `is_train` 占比监控（确认样本有效性，纯观测、零风险）。
2. 应用 A（AdaLN gate 松绑）+ B（BETA_START=0.005）。
3. 重训，盯三个面板：**entropy 是否开始下降**、**policy_loss 是否变量级**、
   **reward 是否抬头**。若是 → reward 一直是好的，只是被淹没。
4. 仍想动 reward：不要砍项，把 `tower_hp_point` 从 5.0 降到 ~2.0，让对线期
   高频信号（last_hit/hp_point）数值上能被感知（推塔是低频大 spike，会盖住
   高频小信号）。这是第二位的。