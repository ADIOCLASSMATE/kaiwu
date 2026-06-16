#!/usr/bin/env python3
"""Kaiwu remote workflow helper.

This project cannot run kaiwudrl locally. Kaiwu WebIDE containers in this
course block general outbound network access, so remote control uses the
platform WebIDE port proxy:

- kaiwu-cli/ for platform API operations (login, log fetching).
- script/proxy_env_server.py, which runs inside WebIDE on 127.0.0.1:8765.
- https://tencentarena.com/p5/ide/<experiment_id>/proxy/8765/ for local access.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAIWU_CLI = (ROOT / "kaiwu-cli/bin/kaiwu.js").resolve()
ENV_FILE = ROOT / ".kaiwu-remote.env"
PASSWORD_FILE = ROOT / "password.txt"
PROXY_SERVER_PORT = 8765

DEFAULT_EXCLUDES = {
    ".DS_Store",
    ".git",
    ".pipeline",
    ".kaiwu-remote.env",
    "password.txt",
    "__pycache__",
    ".pytest_cache",
    "real_game_dataset",
    "real_game_raw_frames",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "kaiwu-sync-*.tgz",
    "script/bootstrap-proxy-server.remote.sh",
}

SYNC_INCLUDE = {
    "agent_diy",
    "agent_ppo",
    "conf",
    "kaiwu.json",
    "train_test.py",
}


def load_env_file() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def cfg(name: str, default: str = "") -> str:
    return os.environ.get(name) or load_env_file().get(name, default)


def domain_type() -> str:
    return cfg("KAIWU_DOMAIN_TYPE", "course")


def domain_id() -> int:
    legacy = cfg("KAIWU_STAGE_ID", "")
    return int(cfg("KAIWU_DOMAIN_ID", legacy or "2383"))


def experiment_id() -> int:
    return int(cfg("KAIWU_EXPERIMENT_ID", "15823"))


def team_id() -> int:
    return int(cfg("KAIWU_TEAM_ID", "6059"))


def read_password_file() -> tuple[str, str]:
    if not PASSWORD_FILE.exists():
        raise SystemExit("password.txt not found")
    data: dict[str, str] = {}
    for raw_line in PASSWORD_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        data[key.strip()] = value.strip()
    phone = data.get("phone_number") or data.get("phone") or data.get("mobile")
    password = data.get("password")
    if not phone or not password:
        raise SystemExit("password.txt must contain phone_number=... and password=...")
    return phone, password


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        text=True,
        capture_output=False,
        check=check,
    )


def node_cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    if not KAIWU_CLI.exists():
        raise SystemExit(f"kaiwu-cli not found: {KAIWU_CLI}")
    return run(["node", str(KAIWU_CLI), *args], check=check)


def kaiwu_sign(timestamp: int, token: str, url_path: str) -> int:
    last = url_path.rstrip("/").split("/")[-1]
    payload = str(timestamp) + token[-32:] + last
    value = 5381
    for char in payload:
        value = (value + ((value << 5) + ord(char))) & 0xFFFFFFFF
    return value & 0x7FFFFFFF


def load_kaiwu_session() -> dict:
    path = Path.home() / ".kaiwu/session.json"
    if not path.exists():
        raise SystemExit("Kaiwu session not found. Run: python3 script/kaiwu_remote.py login")
    return json.loads(path.read_text(encoding="utf-8"))


def course_body(extra: dict | None = None) -> dict:
    body = {
        "domain": {"type": domain_type(), "id": domain_id()},
        "experiment_id": experiment_id(),
    }
    if extra:
        body.update(extra)
    return body


def kaiwu_api(url_path: str, body: dict) -> dict:
    session = load_kaiwu_session()
    token = session.get("token") or ""
    timestamp = int(time.time())
    headers = {
        "Accept": "application/json",
        "Accept-Language": "zh",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "x-kaiwu-ts": str(timestamp),
        "x-kaiwu-auth": str(kaiwu_sign(timestamp, token, url_path)),
    }
    request = urllib.request.Request(
        "https://tencentarena.com" + url_path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("code") not in (None, 0):
        raise SystemExit(f"kaiwu api error {payload.get('code')}: {payload.get('msg') or payload.get('reason')}")
    return payload.get("data", payload)


def login(args: argparse.Namespace) -> None:
    phone, password = read_password_file()
    cli_args = [
        "login",
        "--phone",
        phone,
        "--password",
        password,
        "--stage-id",
        args.stage_id or str(domain_id()),
        "--team-id",
        args.team_id or cfg("KAIWU_TEAM_ID", "8365"),
        "--experiment-id",
        args.experiment_id or cfg("KAIWU_EXPERIMENT_ID", "11427"),
    ]
    if args.no_save_password:
        cli_args.append("--no-save-password")
    node_cli(cli_args)


def cli(args: argparse.Namespace) -> None:
    if not args.args:
        raise SystemExit("usage: script/kaiwu_remote.py cli -- <kaiwu-cli args>")
    cli_args = args.args[1:] if args.args and args.args[0] == "--" else args.args
    node_cli(cli_args)


def ide_status(_: argparse.Namespace) -> None:
    if domain_type() == "course":
        data = kaiwu_api("/api/v5/Course/GetWebIDE", course_body())
        rows = [
            ("id", data.get("id")),
            ("status", data.get("status")),
            ("domain", data.get("domain")),
            ("experiment_id", data.get("experiment_id")),
            ("cluster_config_id", data.get("cluster_config_id")),
            ("project", data.get("project")),
            ("deploy_at", data.get("deploy_at")),
            ("run_time", data.get("run_time")),
            ("host", data.get("host") or "(empty)"),
        ]
        for key, value in rows:
            print(f"{key}: {value}")
    else:
        node_cli(["ide-status"])


def start_ide(args: argparse.Namespace) -> None:
    if domain_type() == "course":
        cur = kaiwu_api("/api/v5/Course/GetWebIDE", course_body())
        body = course_body(
            {
                "competition_team_id": team_id(),
                "cluster_config_id": args.cluster_config_id or cur.get("cluster_config_id") or 0,
                "project": cur.get("project"),
            }
        )
        try:
            data = kaiwu_api("/api/v5/Course/StartWebIDE", body)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except SystemExit as exc:
            text = str(exc)
            if "1008" in text:
                print("IDE already running")
            else:
                raise
    else:
        cli_args = ["start-ide", "--yes"]
        if args.cluster_config_id:
            cli_args.extend(["--cluster-config-id", str(args.cluster_config_id)])
        node_cli(cli_args)


def course_status(_: argparse.Namespace) -> None:
    team = kaiwu_api("/api/v5/Course/GetExperimentTeam", course_body({"mask": ["user"]}))
    exp = kaiwu_api("/api/v5/Course/GetExperiment", {"domain": {"type": domain_type(), "id": domain_id()}, "id": experiment_id()})
    webide = kaiwu_api("/api/v5/Course/GetWebIDE", course_body())
    train_balance = kaiwu_api("/api/v5/Course/GetResourceBalance", course_body({"resource_module": "train"}))
    print("team:", json.dumps(team.get("team"), ensure_ascii=False))
    experiment = exp.get("experiment", {})
    print("experiment:", json.dumps({k: experiment.get(k) for k in ["id", "name", "name_en", "project_code"]}, ensure_ascii=False))
    print("webide:", json.dumps({k: webide.get(k) for k in ["id", "status", "cluster_config_id", "project", "run_time"]}, ensure_ascii=False))
    print("train_quota_type:", train_balance.get("quota_type"))


def bootstrap_proxy_server_command(args: argparse.Namespace) -> None:
    workspace = cfg("REMOTE_WORKSPACE", "/data/projects/hok1v1")
    port = int(cfg("KAIWU_PROXY_PORT", str(PROXY_SERVER_PORT)))
    server_b64 = base64.b64encode((ROOT / "script/proxy_env_server.py").read_bytes()).decode("ascii")
    shell_b64 = base64.b64encode((ROOT / "script/remote-proxy-server.sh").read_bytes()).decode("ascii")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {workspace}
mkdir -p script
python3 - <<'PY'
import base64
files = {{
    "script/proxy_env_server.py": "{server_b64}",
    "script/remote-proxy-server.sh": "{shell_b64}",
}}
for path, data in files.items():
    open(path, "wb").write(base64.b64decode(data))
PY
chmod +x script/proxy_env_server.py script/remote-proxy-server.sh
PROXY_PORT='{port}' REMOTE_WORKSPACE='{workspace}' bash script/remote-proxy-server.sh
"""
    out = Path(args.output) if args.output else ROOT / "script/bootstrap-proxy-server.remote.sh"
    out.write_text(script, encoding="utf-8")
    os.chmod(out, 0o700)
    print(f"wrote {out}")
    print("Copy the contents of this file and paste them into the Kaiwu WebIDE terminal.")
    print("It starts a local WebIDE command server reachable through VSCODE_PROXY_URI.")


def proxy_base_url() -> str:
    configured = cfg("KAIWU_PROXY_URL", "")
    if configured:
        return configured.rstrip("/")
    port = int(cfg("KAIWU_PROXY_PORT", str(PROXY_SERVER_PORT)))
    return f"https://tencentarena.com/p5/ide/{experiment_id()}/proxy/{port}"


def proxy_request(method: str, path: str, body: dict | None = None, timeout: int = 30) -> dict:
    session = load_kaiwu_session()
    url = proxy_base_url().rstrip("/") + "/" + path.lstrip("/")
    data = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {session.get('token') or ''}",
    }
    proxy_token = cfg("KAIWU_PROXY_TOKEN", "")
    if proxy_token:
        headers["X-Kaiwu-Proxy-Token"] = proxy_token
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"proxy api error HTTP {exc.code}: {detail}") from exc
    payload = json.loads(raw.decode("utf-8")) if raw else {}
    if isinstance(payload, dict) and payload.get("error"):
        raise SystemExit(f"proxy api error: {payload.get('error')}")
    return payload


def b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def proxy_health(_: argparse.Namespace) -> None:
    print(json.dumps(proxy_request("GET", "/health"), ensure_ascii=False, indent=2))


def proxy_command(args: argparse.Namespace) -> None:
    query = urllib.parse.urlencode({"cmd_b64": b64url(args.cmd), "timeout": str(args.timeout)})
    result = proxy_request("GET", f"/command?{query}", timeout=args.wait)
    print("returncode:", result.get("returncode"))
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)


def should_exclude(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = relative.parts
    if any(part in DEFAULT_EXCLUDES for part in parts):
        return True
    return any(fnmatch.fnmatch(str(relative), pattern) for pattern in DEFAULT_EXCLUDE_PATTERNS)


def make_payload() -> tuple[Path, int, int]:
    fd, temp_name = tempfile.mkstemp(prefix="kaiwu-sync-", suffix=".tgz", dir=ROOT)
    os.close(fd)
    archive = Path(temp_name)
    count = 0
    with tarfile.open(archive, "w:gz") as tar:
        for include_prefix in sorted(SYNC_INCLUDE):
            path = ROOT / include_prefix
            if not path.exists():
                continue
            if path.is_file():
                tar.add(path, arcname=include_prefix)
                count += 1
            elif path.is_dir():
                for file in path.rglob("*"):
                    if should_exclude(file):
                        continue
                    if file.is_file():
                        tar.add(file, arcname=str(file.relative_to(ROOT)))
                        count += 1
    return archive, archive.stat().st_size, count


def sync(args: argparse.Namespace) -> None:
    archive, size, count = make_payload()
    try:
        raw = archive.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        print(f"payload: {count} files, {size} bytes")
        payload_path = ".kaiwu_sync_payload.tgz.b64"
        chunk_size = 4096
        for index in range(0, len(encoded), chunk_size):
            chunk = encoded[index : index + chunk_size]
            query = urllib.parse.urlencode(
                {
                    "path": payload_path,
                    "content_b64": b64url(chunk),
                    "overwrite": "1",
                    "append": "0" if index == 0 else "1",
                }
            )
            proxy_request("GET", f"/write?{query}", timeout=args.wait)
        remote_code_dir = cfg("REMOTE_CODE_DIR", "/workspace/code")
        rm_targets = " ".join(f"{remote_code_dir}/{p}" for p in sorted(SYNC_INCLUDE))
        cmd = (
            "set -e; "
            "python3 - <<'PY'\n"
            "import base64\n"
            "data = open('.kaiwu_sync_payload.tgz.b64', 'rb').read()\n"
            "open('/tmp/kaiwu_sync_payload.tgz', 'wb').write(base64.b64decode(data))\n"
            "PY\n"
            f"mkdir -p {remote_code_dir}; "
            f"rm -rf {rm_targets}; "
            f"tar -xzf /tmp/kaiwu_sync_payload.tgz -C {remote_code_dir}; "
            "rm -f .kaiwu_sync_payload.tgz.b64 /tmp/kaiwu_sync_payload.tgz; "
            f"cd {remote_code_dir}; "
            "python3 - <<'PY'\n"
            "import ast, pathlib\n"
            "for p in pathlib.Path('.').rglob('*.py'):\n"
            "    if '__pycache__' in p.parts:\n"
            "        continue\n"
            "    ast.parse(p.read_text(encoding='utf-8'))\n"
            "print('sync ok')\n"
            "PY"
        )
        query = urllib.parse.urlencode({"cmd_b64": b64url(cmd), "timeout": "120"})
        result = proxy_request("GET", f"/command?{query}", timeout=args.wait)
        if result.get("returncode") != 0:
            print(result.get("stdout") or "", end="")
            print(result.get("stderr") or "", end="", file=sys.stderr)
            raise SystemExit(result.get("returncode"))
        print(result.get("stdout") or "sync ok")
    finally:
        archive.unlink(missing_ok=True)


def train_test(args: argparse.Namespace) -> None:
    workspace = cfg("REMOTE_WORKSPACE", "/data/projects/hok1v1")
    if args.nohup:
        cmd = f"cd {workspace} && nohup python3 train_test.py > /tmp/kaiwu_train_test.log 2>&1 & echo pid=$!"
        query = urllib.parse.urlencode({"cmd_b64": b64url(cmd), "timeout": "10"})
        result = proxy_request("GET", f"/command?{query}", timeout=30)
    else:
        cmd = f"cd {workspace} && python3 train_test.py"
        query = urllib.parse.urlencode({"cmd_b64": b64url(cmd), "timeout": str(args.timeout)})
        result = proxy_request("GET", f"/command?{query}", timeout=args.wait)
    print("returncode:", result.get("returncode"))
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)


def remote_log(_: argparse.Namespace) -> None:
    query = urllib.parse.urlencode(
        {"cmd_b64": b64url("tail -80 /tmp/kaiwu_train_test.log 2>/dev/null || true"), "timeout": "10"}
    )
    result = proxy_request("GET", f"/command?{query}", timeout=30)
    print(result.get("stdout") or "")


def doctor(_: argparse.Namespace) -> None:
    print(f"root: {ROOT}")
    print(f"kaiwu-cli: {KAIWU_CLI} exists={KAIWU_CLI.exists()}")
    print(f"python: {sys.executable}")
    print(f"password.txt: exists={PASSWORD_FILE.exists()}")
    if PASSWORD_FILE.exists():
        phone, password = read_password_file()
        print(f"password.txt parsed: phone_len={len(phone)} password_len={len(password)}")
    print(f"env file: {ENV_FILE} exists={ENV_FILE.exists()}")
    for key in [
        "KAIWU_DOMAIN_TYPE",
        "KAIWU_DOMAIN_ID",
        "KAIWU_STAGE_ID",
        "KAIWU_TEAM_ID",
        "KAIWU_EXPERIMENT_ID",
        "KAIWU_PROXY_URL",
        "KAIWU_PROXY_PORT",
        "KAIWU_PROXY_TOKEN",
        "REMOTE_WORKSPACE",
        "REMOTE_CODE_DIR",
    ]:
        value = cfg(key, "")
        if key == "KAIWU_PROXY_TOKEN" and value:
            value = value[:3] + "***" + value[-3:]
        print(f"{key}={value or '(unset)'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaiwu remote development helper")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor")
    p.set_defaults(func=doctor)

    p = sub.add_parser("login")
    p.add_argument("--stage-id")
    p.add_argument("--team-id")
    p.add_argument("--experiment-id")
    p.add_argument("--no-save-password", action="store_true")
    p.set_defaults(func=login)

    p = sub.add_parser("cli")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cli)

    p = sub.add_parser("ide-status")
    p.set_defaults(func=ide_status)

    p = sub.add_parser("course-status")
    p.set_defaults(func=course_status)

    p = sub.add_parser("start-ide")
    p.add_argument("--cluster-config-id", type=int)
    p.set_defaults(func=start_ide)

    p = sub.add_parser("bootstrap-proxy-server-command")
    p.add_argument("--output", default="")
    p.set_defaults(func=bootstrap_proxy_server_command)

    p = sub.add_parser("proxy-health")
    p.set_defaults(func=proxy_health)

    p = sub.add_parser("proxy-command")
    p.add_argument("--cmd", required=True)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--wait", type=int, default=90)
    p.set_defaults(func=proxy_command)

    p = sub.add_parser("sync")
    p.add_argument("--wait", type=int, default=120)
    p.set_defaults(func=sync)

    p = sub.add_parser("train-test")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--wait", type=int, default=660)
    p.add_argument("--nohup", action="store_true")
    p.set_defaults(func=train_test)

    p = sub.add_parser("train-test-log")
    p.set_defaults(func=remote_log)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
