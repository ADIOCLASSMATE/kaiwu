#!/usr/bin/env python3
"""HTTP command/file server for Kaiwu WebIDE port proxy.

Kaiwu WebIDE containers have no general outbound network. This server listens
inside WebIDE and is reached from the local machine through VSCODE_PROXY_URI.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_READ_BYTES = 1024 * 1024
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_WRITE_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
DEFAULT_TIMEOUT = int(os.environ.get("MAX_COMMAND_TIMEOUT", "600"))


class ServerError(Exception):
    pass


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "KaiwuProxyEnv/0.1"

    @property
    def workspace_root(self) -> Path:
        return self.server.workspace_root

    @property
    def token(self) -> str:
        return self.server.token

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_json(200, {"ok": True, "workspace": str(self.workspace_root)})
            elif parsed.path == "/list":
                self.require_auth()
                params = parse_qs(parsed.query)
                self.send_json(200, self.op_list(params.get("path", ["."])[0]))
            elif parsed.path == "/read":
                self.require_auth()
                params = parse_qs(parsed.query)
                self.send_json(200, {"content": self.op_read(params.get("path", [""])[0])})
            elif parsed.path == "/download":
                self.require_auth()
                params = parse_qs(parsed.query)
                self.send_json(200, self.op_download(params.get("path", [""])[0]))
            elif parsed.path == "/command":
                self.require_auth()
                params = parse_qs(parsed.query)
                self.send_json(
                    200,
                    self.op_command(
                        {
                            "cmd": self.decode_query_value(params, "cmd_b64"),
                            "timeout": params.get("timeout", [DEFAULT_TIMEOUT])[0],
                        }
                    ),
                )
            elif parsed.path == "/write":
                self.require_auth()
                params = parse_qs(parsed.query)
                content = self.decode_query_value(params, "content_b64")
                self.send_json(
                    200,
                    self.op_write(
                        {
                            "path": params.get("path", [""])[0],
                            "content": content,
                            "overwrite": params.get("overwrite", ["1"])[0] != "0",
                            "append": params.get("append", ["0"])[0] == "1",
                        }
                    ),
                )
            else:
                self.send_json(404, {"error": "not found"})
        except ServerError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def require_auth(self) -> None:
        if not self.token:
            return
        given = self.headers.get("X-Kaiwu-Proxy-Token", "")
        if given != self.token:
            raise ServerError("bad proxy token")

    def send_json(self, status: int, payload: dict | list) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def decode_query_value(self, params: dict, name: str) -> str:
        value = params.get(name, [""])[0]
        if not value:
            return ""
        return base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8")

    def resolve_path(self, user_path: str) -> Path:
        if not isinstance(user_path, str) or not user_path:
            raise ServerError("missing path")
        path = Path(user_path)
        if path.is_absolute():
            raise ServerError("absolute paths are not allowed")
        target = (self.workspace_root / path).resolve()
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ServerError("path escapes workspace root") from exc
        return target

    def op_list(self, path: str) -> list[dict]:
        target = self.resolve_path(path)
        if not target.exists():
            raise ServerError("path does not exist")
        if not target.is_dir():
            raise ServerError("path is not a directory")
        rows = []
        for child in sorted(target.iterdir(), key=lambda item: item.name):
            rows.append(
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return rows

    def op_read(self, path: str) -> str:
        target = self.resolve_path(path)
        if not target.is_file():
            raise ServerError("path is not a file")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ServerError("file too large")
        return target.read_text(encoding="utf-8", errors="replace")

    def op_download(self, path: str) -> dict:
        target = self.resolve_path(path)
        if not target.is_file():
            raise ServerError("path is not a file")
        if target.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise ServerError("file too large")
        data = target.read_bytes()
        return {
            "filename": target.name,
            "size": len(data),
            "encoding": "base64",
            "content_type": "application/octet-stream",
            "data": base64.b64encode(data).decode("ascii"),
        }

    def op_write(self, body: dict) -> dict:
        target = self.resolve_path(body.get("path"))
        content = body.get("content")
        if not isinstance(content, str):
            raise ServerError("content must be a string")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ServerError(f"content too large: {len(encoded)} bytes")
        overwrite = bool(body.get("overwrite", True))
        append = bool(body.get("append", False))
        if target.exists() and not overwrite:
            raise ServerError("file already exists")
        if not target.parent.exists():
            raise ServerError("parent directory does not exist")
        if target.exists() and not target.is_file():
            raise ServerError("target is not a file")
        if append:
            with target.open("a", encoding="utf-8") as file:
                file.write(content)
        else:
            target.write_text(content, encoding="utf-8")
        return {"path": str(target.relative_to(self.workspace_root)), "bytes": len(encoded), "append": append}

    def op_command(self, body: dict) -> dict:
        cmd = body.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ServerError("missing cmd")
        requested_timeout = int(body.get("timeout", DEFAULT_TIMEOUT))
        timeout = max(1, min(requested_timeout, DEFAULT_TIMEOUT))
        completed = subprocess.run(
            cmd,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=True,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-MAX_OUTPUT_BYTES:],
            "stderr": completed.stderr[-MAX_OUTPUT_BYTES:],
        }


class ProxyServer(ThreadingHTTPServer):
    def __init__(self, address, handler_class, workspace_root: Path, token: str):
        super().__init__(address, handler_class)
        self.workspace_root = workspace_root
        self.token = token


def parse_args():
    parser = argparse.ArgumentParser(description="Kaiwu WebIDE proxy command server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace-root", default=os.getcwd())
    parser.add_argument("--token", default=os.environ.get("KAIWU_PROXY_TOKEN", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    server = ProxyServer((args.host, args.port), ProxyHandler, workspace_root, args.token)
    print(f"proxy env server listening on http://{args.host}:{args.port} workspace={workspace_root}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
