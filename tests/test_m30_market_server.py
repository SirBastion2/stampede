"""M30 WebSocket server: import graph and one live frame (offline, CPU)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_market_server_imports_in_fresh_interpreter() -> None:
    code = textwrap.dedent(
        """
        import os
        os.environ["STAMPEDE_OFFLINE"] = "1"
        import ui.market.server as server
        assert server.app is not None
        """
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_market_ws_delivers_one_msgpack_frame() -> None:
    env = {
        **os.environ,
        "STAMPEDE_OFFLINE": "1",
        "STAMPEDE_PORT": "8000",
        "PYTHONPATH": str(ROOT),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "ui.server"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=0.25):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.1)
        else:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise AssertionError(f"market server did not become healthy: {stderr}")

        async def recv_once() -> bytes:
            import websockets

            async with websockets.connect("ws://127.0.0.1:8000/ws/market") as ws:
                payload = await asyncio.wait_for(ws.recv(), timeout=3.0)
                if isinstance(payload, str):
                    return payload.encode()
                return payload

        frame = asyncio.run(recv_once())
        assert len(frame) > 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
