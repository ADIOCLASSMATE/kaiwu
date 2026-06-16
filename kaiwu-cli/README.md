# kaiwu-cli

腾讯开悟（KaiWu）王者荣耀 1v1 比赛工作台 API 的纯 Node CLI。

平台：https://tencentarena.com/p/competition/...
全部命令通过全局 `kaiwu xxx` 调用。零 opencli 依赖、零浏览器依赖、零外部 npm 依赖（仅 Node 18+ 内置模块）。

---

## 安装

```bash
cd kaiwu-cli
npm link              # 全局装 kaiwu 命令
kaiwu help            # 看全部命令
```

## 第一次使用：登录

```bash
kaiwu login
# 交互：
#   请选择登录方式:
#     1) 手机号 + 密码
#     2) 邮箱 + 密码
#   输入 1 / 2: 1
#   手机号（如 15084735093）: ...
#   密码（12-20 字符）: ********
```

或非交互一行登：

```bash
kaiwu login --phone 15084735093 --password '12chars+'
# 或环境变量
KAIWU_PHONE=... KAIWU_PASSWORD=... kaiwu login
```

登录成功后：
- token 落 `~/.kaiwu/session.json`（JWT，6 天有效，chmod 600）
- 凭据落 `~/.kaiwu/credentials.json`（phone/email + password 明文 chmod 600）
- token 过期时下一条命令自动用持久化密码续期，**用户只输一次**

不想留密码：`kaiwu login --no-save-password`，token 失效后需重 login。

---

## 命令一览

```
kaiwu help                            看分类列表
kaiwu <cmd> --help                    看子命令参数
kaiwu --format json|yaml|table <cmd>  切换输出格式（默认 yaml）
```

| 类别 | 命令 |
|---|---|
| 会话 | login / logout / whoami |
| 元信息 | experiment / team-info / list-versions / ide-status |
| 资源 | resource-balance / resource-stat |
| 训练任务 | list-train-task / get-train-task / create-train-task / release-train-task |
| 训练产物 | list-models |
| 模型库 | list-ai-models / submit-model |
| 监控 | metric / log / open-monitor |
| 评估对战 | list-battle-tasks / get-battle-task / list-games / create-battle-task / stop-battle-task |
| 比赛 | list-races |
| IDE 保活 | start-ide / keep-ide-alive |
| 下载 | download-ai-model / download-train-task / download-game / download-abs-tool / download-battle-archive |

写命令均带 `--dry-run` 默认开 + `--yes` 显式确认，避免误触。

### 下载命令（2026-05-08 通过 opencli 浏览器抓 Network 逆向）

5 类下载全部走统一接口 `POST /api/v5/Competition/GetAuthDownloadURL`，返回 `{url, size}`，url 是 cos presigned。`download_source.type` 区分：

| 命令 | source.type | file_key 来源 | 用途 |
|---|---|---|---|
| `download-ai-model --id <ai_model_id>` | `ai_model` | list-ai-models 的 `file[0].key` | 模型库永久模型 |
| `download-train-task --task-id <id>` | `train_task` | list-train-task 的 `file.key` | 训练任务上传的代码包 |
| `download-game --battle-task-id X --game-id Y --kind abs\|log` | `game` | list-games 的 `file[].key`（按 type 选 abs/log） | 单局录像/日志 |
| `download-abs-tool` | `common` (id=0) | 固定 `public/abs_tools/ABSParsingTool_hok_g_shelled_v1.0.2.zip` | ABS 录像播放器（2GB） |
| `download-battle-archive --battle-task-id X --kind abs\|log` | 异步流 | 先 `ArchiveBattleTaskFile` 触发打包，轮询 `GetBattleTask` 拿 file_key，再 GetAuthDownloadURL | 整 battle 全部录像/日志打包 |

实测落盘大小：ai_model 10MB / train_task 代码包 132KB / game abs 36KB / abs-tool 2.0GB。

---

## 接口分析过程

### 1. 起点：探活

`tencentarena.com` 是 React + axios SPA。直接 `opencli explore` 探到 0 endpoints —— 缓存 SPA navigate 同 URL 不再发 XHR。

改用浏览器 `performance.getEntriesByType('resource')` 抓页面已发请求：

```js
performance.getEntriesByType('resource')
  .filter(e => e.initiatorType === 'fetch' || e.initiatorType === 'xmlhttprequest')
  .map(e => e.name)
```

抓到 `/api/v5/Competition/ListTrainTask` / `GetExperimentHistory` / `GetWebIDE` 等。URL 模式 `/api/v5/<Module>/<Action>`，腾讯内部 RPC 风格。

### 2. 注入 fetch 拦截器看真实 body / headers

```js
const _f = window.fetch;
window.fetch = async function(input, init) {
  const url = (typeof input === 'string') ? input : input.url;
  const res = await _f.apply(this, arguments);
  // 记录 url, headers, body, response
  return res;
};
```

注意：`location.reload()` 会清掉 monkey patch（page JS 重建）。要触发新 XHR 必须通过 UI 交互（搜索框 / 刷新按钮），不能硬刷。

抓到第一个完整 SignIn 请求的 headers：

```
Authorization: Bearer eyJhbGciOiJIUzUxMi...
x-kaiwu-ts: 1778171671
x-kaiwu-auth: 809187321
```

`Authorization` 是 JWT，但 `x-kaiwu-auth` 是个签名整数，每次请求重算。

### 3. 列全量 endpoint（webpack 字符串挖掘）

webpack chunk 在 `window.webpackChunkcompetition_v5`，push 一个 dummy chunk 拿到 `__webpack_require__`：

```js
window.webpackChunkcompetition_v5.push([
  ['__probe_' + Math.random()],
  {},
  function(req) { window.__opencliReq = req; }
]);
```

production webpack 5 的 `req.c`（模块缓存）被剔除，但 `req.m`（factory map）还在。遍历 1942 个模块，正则 `/['"]\/api\/v5\/[A-Za-z]+\/[A-Za-z]+['"]/g` 提取字符串字面量，挖到 91 个 Competition endpoint + User / Course 模块全集。

---

## 签名解密

`x-kaiwu-auth` 是 31-bit 整数，同 body 不同 ts 出不同 auth，跟 ts 强相关。

webpack module 73916 里搜 `x-kaiwu-auth` 找到这段：

```js
function s(e) {
  for (var t = 5381, n = 0, r = e.length; n < r; ++n)
    t += (t << 5) + e.charCodeAt(n);
  return 2147483647 & t;
}

var sig = s(a + o.slice(-32) + e.split("/").slice(-1)[0]);
c["x-kaiwu-auth"] = sig;
c["x-kaiwu-ts"] = a;
```

变量含义：
- `a` = unix timestamp（秒）
- `o` = 整个 JWT，取末尾 32 字符
- `e` = 请求 path，split('/').slice(-1)[0] 取最后一段

算法：DJB2 hash 变种（`t = t*33 + c`），最后 `& 0x7FFFFFFF` 取 31-bit。

Python 复现 4 组样本 100% 命中：

| ts | expected | got |
|---|---|---|
| 1778171671 | 809187321 | 809187321 |
| 1778171727 | 1102027483 | 1102027483 |
| 1778171730 | 616062261 | 616062261 |
| 1778171735 | 109629786 | 109629786 |

破解后 curl 直调通过：

```bash
curl -sS -X POST https://tencentarena.com/api/v5/Competition/ListTrainTask \
  -H "Authorization: Bearer $TOKEN" \
  -H "x-kaiwu-ts: $TS" \
  -H "x-kaiwu-auth: $SIG" \
  -d '{"page":{"current":1,"size":3},...}'
# code:0  data: {train_task: [...]}
```

完全脱离浏览器，纯 Node `_session.js` 实现签名 + 鉴权封装。

---

## 字段 fuzz 方法论

### 错误码语义

| code | 含义 | 价值 |
|---|---|---|
| 51 | protobuf schema 校验错 | 金矿——错误信息直接暴露字段名 |
| 1601 | 业务参数错 | 黑盒——不暴露字段名，需翻 webpack |
| 1031 | 权限 / 字段名错 | 歧义——可能权限拒，也可能缺字段 |
| 1115 | 配额满 | 好信号——schema 完全合法 |
| 1002 | 业务创建失败 | 字段对了但语义错 |
| 1700 | 账号密码错 | SignIn 通过，密码不对 |
| 1040 | token 校验失败 | session 过期，需 refresh |

### Fuzz 进阶

1. 空 body → 拿首个必填字段名（51）
2. 逐字段加 → 51 消失证明 schema 完整
3. 变错误码 → 51 → 1601/1031/1115 业务层校验
4. 1031 + 加齐字段后还 1031 = 真权限拒；从 1031 变其他码 = 之前是字段缺失

### 实例：CreateAiModel

```
空 body                     → 51: name / share_type / file 必填
+ name + share_type=team    → 51: file 必填
+ file                      → 1031（误判：以为权限拒）
+ training_mode=distributed → 0 success! id=254214
```

教训：1031 不一定是权限——加齐 `training_mode` 字段后通过。判别 1031 真假必须 fuzz 完整字段。

### 实例：手机号登录字段

```
phone:"15369958967"     → 1601 参数异常（误判：以为字段不支持）
phone:"+86-15369958967" → 1700 账号密码错（schema 通过）
```

服务端要 `+86-` 国家码前缀。CLI 自动加。

---

## 关键陷阱与判错

### 陷阱 1：GetTrainTask 用 uuid 不是 id

直接 `train_task_id` 报 51 提示要 `uuid`。CLI 加 `--task-id` 自动从 ListTrainTask 反查 uuid。

### 陷阱 2：ListBattleTask owner.type 默认是 "battle"

`team` / `user` / `experiment` 都返回空，正确是 `{type: "battle", id: 0}`（webpack 默认值）。误用 owner=team 会让用户以为"从未创建过对战"。

### 陷阱 3：CreateBattleTask 第二 camp 不能用 opponent_agent

`opponent_agent: "common_ai"` 是训练时评估的 toml 字段，不是对战 API 字段。手动评估对战要求两个 camp 都是 ai_model_id。

### 陷阱 4：submit-model 的 filename 必须构造格式

list-models 返回 `hok1v1-145229-ppo-948-2026_05_08_00_43_42-61.1.3.zip`（带 task_id + 日期），但服务端期望 `hok1v1-ppo-948.zip`。CLI 自动用 `${project}-${algorithm}-${step}.zip` 构造。

### 陷阱 5：BatchCheckAuth 不可信

`auth_map[v5.User.Competition.CreateRace]=false` 不一定意味着真拒——resource_code 跟 endpoint 不一一对应。判断有无权限必须实测 endpoint。

### 陷阱 6：监控指标 metric_exp 是 PromQL

`GetTrainMetric` 的 `metric_exp` 不是简单 `reward`，而是完整 PromQL 如 `sum(max_over_time(kaiwu_actor_predict_succ_cnt{}[1h]))`。`time` / `start_time` / `end_time` 是 `{timestamp: "ISO8601"}` 对象，不是 unix number。

### 陷阱 7：reload 清拦截器

`location.reload()` 后 page JS 重建，注入的 fetch monkey-patch 全丢。要触发新 XHR 必须通过 UI 交互。

### 陷阱 8：监控页 panel 懒加载

监控页 5 个 panel 组（基础/硬件/环境/评估/算法）默认只首屏加载，IntersectionObserver lazy load + React Query 缓存。要抓全 metric query 列表只能逐个 panel 点击触发。

### 陷阱 9：手机号登录国家码 + 密码长度

- 手机号必须 `+86-` 前缀（裸 `15369958967` 报 1601）
- 密码必须 12-20 字符（前后端双重校验）
- 没有滑块 / 极验 captcha，纯 `{phone, password}` 即可登录

---

## 架构

```
kaiwu-cli/
├── bin/kaiwu.js          # 主入口：参数解析 + 子命令 dispatch + 输出格式化
├── commands/             # 27 个子命令，每个 export default { name, args, run }
│   ├── login.js          # 交互式登录 + 凭据持久化
│   ├── logout.js
│   ├── whoami.js
│   ├── list-train-task.js
│   ├── get-train-task.js
│   ├── create-train-task.js
│   ├── release-train-task.js
│   ├── list-models.js
│   ├── list-ai-models.js
│   ├── submit-model.js
│   ├── list-battle-tasks.js
│   ├── get-battle-task.js
│   ├── list-games.js
│   ├── create-battle-task.js
│   ├── stop-battle-task.js
│   ├── list-races.js
│   ├── metric.js
│   ├── log.js
│   ├── open-monitor.js
│   ├── experiment.js
│   ├── team-info.js
│   ├── list-versions.js
│   ├── ide-status.js
│   ├── start-ide.js
│   ├── keep-ide-alive.js
│   ├── resource-balance.js
│   └── resource-stat.js
├── _session.js           # 共享：DJB2 签名 + token + auto-refresh + api()
├── package.json
└── README.md             # 本文件
```

`_session.js` 是核心：
- `sign(ts, token, path)` — DJB2 算签名
- `rawSignIn(creds)` — 匿名 SignIn 拿 token
- `loadSession({autoRefresh})` — 加载 session.json，过期自动用 credentials 续期
- `refreshSession()` — 强制 SignIn 刷 token
- `api(urlPath, body, opts)` — 调任何 endpoint，1040 自动 retry refresh
- `loadCredentials() / saveCredentials() / clearCredentials()` — 凭据持久化

---

## 持久化文件

| 文件 | 内容 | 何时写 | 权限 |
|---|---|---|---|
| `~/.kaiwu/session.json` | token + 比赛上下文 + user_id | login + 每次 refresh | 600 |
| `~/.kaiwu/credentials.json` | phone/email + password | login（非 --no-save-password） | 600 |

token 6 天有效，credentials 永久（直到 `kaiwu logout`）。

---

## 完整命令速查

| 命令 | 类别 | 写操作 | 说明 |
|---|---|---|---|
| login | 会话 | yes | 交互式登录 + 持久化 |
| logout | 会话 | yes | 清 session + credentials |
| whoami | 会话 | - | 当前登录状态 |
| experiment | 元 | - | 实验配置 / 算法白名单 |
| team-info | 元 | - | 队伍详情 / 成员 |
| list-versions | 元 | - | 项目版本 |
| ide-status | 元 | - | WebIDE 容器状态 |
| resource-balance | 资源 | - | 单模块配额 |
| resource-stat | 资源 | - | 全模块配额 |
| list-train-task | 训练 | - | 任务列表 |
| get-train-task | 训练 | - | 任务详情（自动 task-id → uuid） |
| create-train-task | 训练 | yes | 创建任务（默认复用最近 file，支持 --pre-train-model-id） |
| release-train-task | 训练 | yes | 释放（running 必须 --yes） |
| list-models | 训练 | - | 单 task 产出 checkpoint |
| list-ai-models | 模型库 | - | 永久模型库 |
| submit-model | 模型库 | yes | checkpoint → 永久库 |
| metric | 监控 | - | 拉指标范围 |
| log | 监控 | - | 错误日志按 module 聚合 |
| open-monitor | 监控 | - | 跳浏览器看 Grafana |
| list-battle-tasks | 评估 | - | 对战列表（默认 owner=battle） |
| get-battle-task | 评估 | - | 详情 + 解析 end_info 出 KDA / 胜场 |
| list-games | 评估 | - | 单局明细 |
| create-battle-task | 评估 | yes | 创建对战 |
| stop-battle-task | 评估 | yes | 停止对战 |
| list-races | 比赛 | - | 天梯 / 测评（创建权限只 leader 有） |
| start-ide | IDE 保活 | yes | StartWebIDE（IDE running 时返 1008，CLI 当 noop） |
| keep-ide-alive | IDE 保活 | yes | 常驻轮询 + 自动重启（搭配 .vscode/tasks.json runOn:folderOpen 自动起 env_agent） |

---

## WebIDE 回收 + 保活方案

实测 IDE pod **每 30 分钟左右被平台重部署**一次（实测期间 deploy_at 跳了 1:24:51 → 1:54:52，跟 idle / API 调用无关）。开悟 webpack 里所有 WebIDE endpoint：`GetWebIDE / StartWebIDE / CloseWebIDE / StartWebIDEUpgrade / UpgradeWebIDECode / BatchGetWebIDEChangeFile / ListWebIDEVersion`，**没有 heartbeat / extend / touch 任何接口**。`web_ide_max_run_time` 配额是 `-1`（不限），所以也不是 hard timeout —— 平台自己的固定调度。

回收后的新容器跟原来同一个 NFS 卷（`/workspace/code/`）但其他全是 fresh container layer，所以：
- `/workspace/code/.vscode/` 内容跨重启保留（软链 `/data/projects/hok1v1/.vscode -> /workspace/code/.vscode`）
- `/root/`、`/data/projects/hok1v1/` 模板 + 软链都重建
- env_agent 进程（python3 env_agent.py）肯定没了

`ps -ef` 实测 code-server 启动参数带 `--disable-workspace-trust`，所以 `runOn: folderOpen` 任务会**直接 fire**，不要 trust prompt。

落地的两层保活：

```bash
# L1：IDE 容器自动重启
kaiwu keep-ide-alive --interval 60       # stopped → 自动 StartWebIDE，restarts_triggered 计数

# L2：env_agent 自动起 — 已写入 /workspace/code/.vscode/tasks.json
# {"version":"2.0.0","tasks":[{"label":"auto-start env_agent","runOptions":{"runOn":"folderOpen"},...}]}
# 触发条件：有 client（浏览器 tab 或 headless chromium）打开 IDE 网页让 code-server 进入 workspace
```

**limitation**：完全无人值守保活需要 L3（常驻 headless chromium 打开 `https://tencentarena.com/p5/ide/<experiment_id>/?folder=/data/projects/hok1v1`），当前未实装。

---

## License

MIT。代码逆向出来的 API 协议属于腾讯开悟平台，仅作个人参赛工具使用。
