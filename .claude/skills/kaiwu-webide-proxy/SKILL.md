---
name: kaiwu-webide-proxy
description: Use for this Tencent Kaiwu/Tencent Arena HoK 1v1 course project when operating the remote WebIDE, syncing local code, running remote commands, checking IDE status, running train_test.py, or fetching training task logs/metrics from Kaiwu training management. This project must use the WebIDE port-proxy workflow for remote shell work and Course/GetTrainLog + Course/GetTrainMetricRange for course training data.
---

# Kaiwu WebIDE Proxy Workflow

This project cannot run `kaiwudrl` locally. Remote work is done through Kaiwu WebIDE's VS Code port proxy:

```text
local helper -> tencentarena.com/p5/ide/<experiment_id>/proxy/8765 -> 127.0.0.1:8765 inside WebIDE
```

Do not use any workflow that requires the WebIDE container to call out to a public service. The WebIDE container cannot connect to ordinary public HTTPS hosts or public custom ports.

## Setup

1. Make sure local login/session is valid:

   ```bash
   python3 script/kaiwu_remote.py login
   python3 script/kaiwu_remote.py ide-status
   ```

2. Generate the WebIDE bootstrap script:

   ```bash
   python3 script/kaiwu_remote.py bootstrap-proxy-server-command
   ```

3. Paste the contents of `script/bootstrap-proxy-server.remote.sh` into the Kaiwu WebIDE terminal.

   Expected remote output:

   ```text
   started pid=... url=http://127.0.0.1:8765 workspace=/data/projects/hok1v1 log=/tmp/kaiwu-proxy-env-server.log
   proxy env server listening on http://127.0.0.1:8765 workspace=/data/projects/hok1v1
   ```

4. Verify from local:

   ```bash
   python3 script/kaiwu_remote.py proxy-health
   python3 script/kaiwu_remote.py proxy-command --cmd 'pwd && hostname'
   ```

## Daily Commands

Sync local code to remote `/workspace/code` and run Python syntax checks:

```bash
python3 script/kaiwu_remote.py sync
```

Run a remote command. Commands execute with server workspace root `/data/projects/hok1v1`:

```bash
python3 script/kaiwu_remote.py proxy-command --cmd 'cd /data/projects/hok1v1 && ls -la'
```

Run the platform smoke test:

```bash
python3 script/kaiwu_remote.py train-test --nohup
python3 script/kaiwu_remote.py train-test-log
```

For synchronous short checks:

```bash
python3 script/kaiwu_remote.py train-test --timeout 180 --wait 210
```

If `train-test` returns an empty response but processes/logs show activity, use:

```bash
python3 script/kaiwu_remote.py proxy-command --cmd 'ps -ef | grep train_test.py | grep -v grep || true'
python3 script/kaiwu_remote.py proxy-command --cmd 'cd /data/projects/hok1v1 && grep -R "ERROR\\|Traceback\\|Exception\\|Execution error" -n log/learner log/aisrv 2>/dev/null | tail -80 || true'
```

## Training Management (Logs & Metrics)

For platform training-management data in this course project, use `Course/GetTrainLog` and `Course/GetTrainMetricRange`, **not** `Competition/GetTrainLog` / `Competition/GetTrainMetricRange`.

The verified course context is:

```text
domain.type=course
domain.id=2383
experiment_id=15823
```

Use the local Kaiwu session token from `~/.kaiwu/session.json`; do not use the WebIDE proxy for platform APIs.

### Logs CLI

Fetch all raw logs for a task by name and save JSONL:

```bash
node kaiwu-cli/bin/kaiwu.js log \
  --domain-type course \
  --domain-id 2383 \
  --experiment-id 15823 \
  --name train-diy-v0_0 \
  --all \
  --output logs/train-diy-v0_0.jsonl
```

Filter to errors:

```bash
node kaiwu-cli/bin/kaiwu.js log \
  --domain-type course \
  --domain-id 2383 \
  --experiment-id 15823 \
  --name train-diy-v0_0 \
  --level ERROR \
  --all \
  --output logs/train-diy-v0_0-errors.jsonl
```

List available log levels/modules:

```bash
node kaiwu-cli/bin/kaiwu.js log --domain-type course --domain-id 2383 --experiment-id 15823 --name train-diy-v0_0 --query var_level
node kaiwu-cli/bin/kaiwu.js log --domain-type course --domain-id 2383 --experiment-id 15823 --name train-diy-v0_0 --query var_module
```

Get 15-second log counts:

```bash
node kaiwu-cli/bin/kaiwu.js log \
  --domain-type course \
  --domain-id 2383 \
  --experiment-id 15823 \
  --name train-diy-v0_0 \
  --query stat_log \
  --interval 15 \
  --all
```

### Metrics CLI

Fetch all known metrics for a task by ID (default: summary with min/max/last):

```bash
node kaiwu-cli/bin/kaiwu.js metric \
  --domain-type course \
  --domain-id 2383 \
  --experiment-id 15823 \
  --task-id 218419
```

Fetch specific metrics by name:

```bash
node kaiwu-cli/bin/kaiwu.js metric \
  --domain-type course \
  --domain-id 2383 \
  --experiment-id 15823 \
  --task-id 218419 \
  --names win_rate,reward,total_loss
```

Dump raw values for a metric:

```bash
node kaiwu-cli/bin/kaiwu.js metric \
  --domain-type course \
  --domain-id 2383 \
  --experiment-id 15823 \
  --task-id 218419 \
  --names win_rate \
  --raw
```

Custom PromQL expression:

```bash
node kaiwu-cli/bin/kaiwu.js metric \
  --domain-type course \
  --domain-id 2383 \
  --experiment-id 15823 \
  --task-id 218419 \
  --expr 'avg(kaiwu_win_rate{model_id="selfplay"})' \
  --step 60
```

Available metric names: `win_rate`, `reward`, `total_loss`, `value_loss`, `policy_loss`, `entropy_loss`, `self_tower_hp`, `enemy_tower_hp`, `frame`, `money_per_frame`, `kill`, `death`, `hurt_by_hero`, `hurt_to_hero`, `predict_succ_cnt`, `train_global_step`, `episode_cnt`, `batch_train_cost_time_ms`, `sample_production_and_consumption_ratio`, and more (37 total).

### Raw Log API Contract

Valid `Course/GetTrainLog` queries:

```text
var_level
var_module
query_log
stat_log
```

Use this raw-log payload shape:

```json
{
  "domain": {"type": "course", "id": 2383},
  "experiment_id": 15823,
  "train_task_id": 204664,
  "start_time": {"timestamp": "2026-06-07T07:37:11.764Z"},
  "end_time": {"timestamp": "2026-06-07T07:39:53.877Z"},
  "query": "query_log",
  "page": {"size": 20, "current": 1},
  "var": {"message": "*", "level": "*", "module": "*"}
}
```

Use this aggregate payload shape:

```json
{
  "domain": {"type": "course", "id": 2383},
  "experiment_id": 15823,
  "train_task_id": 204664,
  "start_time": {"timestamp": "2026-06-07T07:37:11.764Z"},
  "end_time": {"timestamp": "2026-06-07T07:39:53.877Z"},
  "query": "stat_log",
  "page": {"size": 100, "current": 1},
  "var": {"message": "*", "level": "*", "module": "*", "interval": "15"}
}
```

Do not use `stat_level_log`; it is the old competition-side guess and does not fetch course raw logs. Do not guess enum names; unsupported values return protobuf enum errors. If task times are missing, use the task `created_at` as `start_time` and `end_time || now` as `end_time`.

## Important Constraints

- Run `train_test.py` from `/data/projects/hok1v1`, not `/workspace/code`; platform `tools/` lives under `/data/projects/hok1v1`.
- WebIDE POST requests through the platform proxy can be swallowed as empty `200` responses. The helper intentionally uses GET with base64 query payloads for command/write operations.
- `script/bootstrap-proxy-server.remote.sh` is generated and ignored. Regenerate it after changing `script/proxy_env_server.py` or `script/remote-proxy-server.sh`.
- `.kaiwu-remote.env` is local/private. The checked example should contain course IDs, proxy port, and remote paths only.
