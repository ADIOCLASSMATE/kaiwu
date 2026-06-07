---
name: kaiwu-webide-proxy
description: Use for this Tencent Kaiwu/Tencent Arena HoK 1v1 course project when operating the remote WebIDE, syncing local code, running remote commands, checking IDE status, or running train_test.py. This project must use the WebIDE port-proxy workflow because the WebIDE container has no general outbound network access.
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

## Important Constraints

- Run `train_test.py` from `/data/projects/hok1v1`, not `/workspace/code`; platform `tools/` lives under `/data/projects/hok1v1`.
- WebIDE POST requests through the platform proxy can be swallowed as empty `200` responses. The helper intentionally uses GET with base64 query payloads for command/write operations.
- `script/bootstrap-proxy-server.remote.sh` is generated and ignored. Regenerate it after changing `script/proxy_env_server.py` or `script/remote-proxy-server.sh`.
- `.kaiwu-remote.env` is local/private. The checked example should contain course IDs, proxy port, and remote paths only.
