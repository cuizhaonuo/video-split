"""票 04：scenes 缓存 / analyze / 缩略图 API 测试。

全程合成数据（手工构造 npz + pipeline.segment.analyze），不碰 YOLO。
"""

from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import write_poses_npz


# ---------------------------------------------------------------- GET scenes

def test_scenes_cached_returns_analyzed_true(analyzed_sample: Path, client):
    resp = client.get(f"/api/videos/{quote('sample.mp4')}/scenes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analyzed"] is True
    assert len(data["scenes"]) >= 1
    first = data["scenes"][0]
    assert first["index"] == 1
    assert first["start_ms"] == 0
    assert first["name"].startswith("动作")
    assert first["thumbnail"].endswith("scene-01.jpg")
    assert data["params"]["min_segment_s"] == 1.0


def test_scenes_uncached_returns_analyzed_false(client):
    resp = client.get(f"/api/videos/{quote('训练视频.mp4')}/scenes")
    assert resp.status_code == 200
    assert resp.json() == {"analyzed": False, "scenes": [], "analyzing": False}


def test_scenes_does_not_trigger_analysis(video_dir: Path, client):
    """缓存语义：GET scenes 绝不触发分析。"""
    resp = client.get(f"/api/videos/{quote('训练视频.mp4')}/scenes")
    assert resp.status_code == 200
    assert not (video_dir / "训练视频.mp4.scenes.json").exists()
    assert not (video_dir / "训练视频.mp4.poses.npz").exists()


def test_scenes_missing_video_404(client):
    assert client.get("/api/videos/no-such.mp4/scenes").status_code == 404


def test_scenes_rejects_path_traversal(client):
    assert client.get("/api/videos/..%2F..%2Ftests%2Fconftest.py/scenes").status_code == 404


# ---------------------------------------------------------------- POST analyze

def test_analyze_runs_subprocess_and_respects_min_segment(video_dir: Path, client):
    write_poses_npz(video_dir / "sample.mp4")
    resp = client.post(
        f"/api/videos/{quote('sample.mp4')}/analyze", json={"min_segment_s": 10}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analyzed"] is True
    # 10s 视频 + 最短段 10s → 必然合并为 1 段
    assert len(data["scenes"]) == 1
    assert data["params"]["min_segment_s"] == 10.0
    assert (video_dir / "sample.mp4.scenes.json").is_file()


def test_analyze_small_min_segment_allows_more_scenes(video_dir: Path, client):
    write_poses_npz(video_dir / "sample.mp4")
    resp = client.post(
        f"/api/videos/{quote('sample.mp4')}/analyze", json={"min_segment_s": 0.5}
    )
    assert resp.status_code == 200
    # 合成轨迹有强烈的前后运动变化，小粒度下应切出多段
    assert len(resp.json()["scenes"]) >= 2


def test_analyze_default_min_segment_is_5(video_dir: Path, client):
    write_poses_npz(video_dir / "sample.mp4")
    resp = client.post(f"/api/videos/{quote('sample.mp4')}/analyze", json={})
    assert resp.status_code == 200
    assert resp.json()["params"]["min_segment_s"] == 5.0


def test_analyze_missing_video_404(client):
    resp = client.post("/api/videos/no-such.mp4/analyze", json={})
    assert resp.status_code == 404


def test_analyze_invalid_min_segment_422(client):
    resp = client.post(
        f"/api/videos/{quote('sample.mp4')}/analyze", json={"min_segment_s": 0}
    )
    assert resp.status_code == 422


def test_analyze_chinese_name_full_chain(video_dir: Path, client):
    """中文名视频全链路：analyze 落盘 → 二次 GET 读缓存 → 缩略图路径正确。"""
    write_poses_npz(video_dir / "训练视频.mp4")
    resp = client.post(
        f"/api/videos/{quote('训练视频.mp4')}/analyze", json={"min_segment_s": 10}
    )
    assert resp.status_code == 200
    scenes = resp.json()["scenes"]
    assert len(scenes) == 1
    assert scenes[0]["thumbnail"] == "训练视频.mp4.thumbnails/scene-01.jpg"

    # 二次打开读缓存（不再触发分析）
    again = client.get(f"/api/videos/{quote('训练视频.mp4')}/scenes")
    assert again.status_code == 200
    assert again.json()["analyzed"] is True
    assert again.json()["scenes"] == scenes
    assert (video_dir / "训练视频.mp4.thumbnails" / "scene-01.jpg").is_file()
