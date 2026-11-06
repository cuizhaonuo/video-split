"""分析全链路后台化测试：npz 缺失 → 后台 姿态提取→分段 任务 → 状态轮询 → scenes 出现。

不碰真实 YOLO：monkeypatch app.main 的 subprocess.run，用假实现模拟两阶段
（extract 成功时落一个合成 npz，segment 成功时落一个最小 scenes.json）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import numpy as np
import pytest

from conftest import make_keypoints

NAME = "训练视频.mp4"


class _FakeProc:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


def _fake_run_factory(
    succeed_extract: bool = True,
    succeed_segment: bool = True,
    slow_extract_s: float = 0.0,
) -> Callable:
    """伪造 subprocess.run：按命令里的模块名分发到假 extract / 假 segment。"""

    def fake_run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if any("pipeline.extract_pose" in c for c in cmd):
            if slow_extract_s:
                time.sleep(slow_extract_s)
            if not succeed_extract:
                return _FakeProc(1, stderr="extract boom")
            video = cmd[-1]  # 命令形如 [python, -m, pipeline.extract_pose, <视频>]
            kps, bbox = make_keypoints(50, 5.0)
            np.savez(
                video + ".poses.npz",
                timestamps_ms=np.arange(50) * 200.0,
                keypoints=kps,
                bbox=bbox,
                video=video,
            )
            return _FakeProc(0)
        # segment 阶段：命令形如 [python, -m, pipeline.segment, <npz>, --min-segment-s, N]
        if not succeed_segment:
            return _FakeProc(1, stderr="segment boom")
        npz = cmd[cmd.index("--min-segment-s") - 1]
        scenes_json = Path(str(npz).replace(".poses.npz", ".scenes.json"))
        scenes_json.write_text(
            json.dumps(
                {
                    "video": str(npz)[: -len(".poses.npz")],
                    "scenes": [
                        {
                            "index": 1,
                            "name": "动作 1",
                            "start_ms": 0,
                            "end_ms": 10000,
                            "thumbnail": None,
                            "thumbnail_ms": 0,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return _FakeProc(0)

    return fake_run


def _wait_not_running(client, timeout: float = 10.0) -> dict:
    """轮询 status 直到任务结束，返回最终状态。"""
    url = f"/api/videos/{quote(NAME)}/analyze/status"
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(url)
        assert resp.status_code == 200
        last = resp.json()
        if not last["running"]:
            return last
        time.sleep(0.05)
    raise AssertionError(f"任务未在 {timeout}s 内结束: {last}")


@pytest.fixture
def patched(client, monkeypatch):
    """默认成功的假 subprocess；返回可改写的 holder。"""
    holder = {"fake": _fake_run_factory()}
    monkeypatch.setattr("app.main.subprocess.run", lambda cmd, **kw: holder["fake"](cmd, **kw))
    return holder


# ---------------------------------------------------------------- 全链路成功

def test_missing_npz_starts_background_task(client, patched):
    """npz 缺失 → 202 启动后台任务 → 轮询到完成 → scenes 出现。"""
    resp = client.post(f"/api/videos/{quote(NAME)}/analyze", json={"min_segment_s": 5})
    assert resp.status_code == 202
    assert resp.json()["started"] is True

    st = _wait_not_running(client)
    assert st["error"] is None
    assert st["stage"] is None

    scenes = client.get(f"/api/videos/{quote(NAME)}/scenes").json()
    assert scenes["analyzed"] is True
    assert scenes["analyzing"] is False
    assert len(scenes["scenes"]) == 1


def test_status_reports_running_before_completion(client, patched):
    """提取阶段故意放慢，观察 status 的 running/extract 中间态。"""
    patched["fake"] = _fake_run_factory(slow_extract_s=0.5)
    resp = client.post(f"/api/videos/{quote(NAME)}/analyze", json={"min_segment_s": 5})
    assert resp.status_code == 202

    st = client.get(f"/api/videos/{quote(NAME)}/analyze/status").json()
    assert st["running"] is True
    assert st["stage"] == "extract"

    # scenes 端点同时暴露 analyzing
    scenes = client.get(f"/api/videos/{quote(NAME)}/scenes").json()
    assert scenes["analyzing"] is True

    _wait_not_running(client)


# ---------------------------------------------------------------- 并发与失败

def test_duplicate_analyze_rejected_409(client, patched):
    """任务运行期间重复 POST 被拒绝（409），任务本身不受影响。"""
    patched["fake"] = _fake_run_factory(slow_extract_s=0.5)
    first = client.post(f"/api/videos/{quote(NAME)}/analyze", json={"min_segment_s": 5})
    assert first.status_code == 202

    second = client.post(f"/api/videos/{quote(NAME)}/analyze", json={"min_segment_s": 5})
    assert second.status_code == 409

    st = _wait_not_running(client)
    assert st["error"] is None


def test_extract_failure_propagates_error(client, patched):
    patched["fake"] = _fake_run_factory(succeed_extract=False)
    resp = client.post(f"/api/videos/{quote(NAME)}/analyze", json={"min_segment_s": 5})
    assert resp.status_code == 202

    st = _wait_not_running(client)
    assert st["running"] is False
    assert "extract boom" in st["error"]
    # 失败后不应有 scenes 缓存
    assert not client.get(f"/api/videos/{quote(NAME)}/scenes").json()["analyzed"]


def test_segment_failure_propagates_error(client, patched):
    patched["fake"] = _fake_run_factory(succeed_segment=False)
    resp = client.post(f"/api/videos/{quote(NAME)}/analyze", json={"min_segment_s": 5})
    assert resp.status_code == 202

    st = _wait_not_running(client)
    assert "segment boom" in st["error"]
    assert st["running"] is False


def test_status_unknown_video_404(client):
    assert client.get("/api/videos/no-such.mp4/analyze/status").status_code == 404
