"""ticket-03 单元测试：边界检测 / 短段合并 / 特征插值 / 落盘端到端。

单测全部用合成轨迹与合成视频；真实视频验收不在此（人工执行）。
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pipeline.segment import (
    build_features,
    detect_change_points,
    main,
    merge_short_segments,
    analyze,
)

SAMPLE_MS = 200.0  # 5fps


# ---------- fixtures ----------

def make_timestamps(n: int, sample_ms: float = SAMPLE_MS) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * sample_ms


def step_features(n: int, boundaries: list[int], levels: list[float],
                 noise: float = 0.15, seed: int = 42) -> np.ndarray:
    """构造已知变化点的一维特征序列（各段恒定均值 + 噪声），扩展为 (n,3)。"""
    rng = np.random.default_rng(seed)
    sig = np.zeros(n)
    edges = [0] + boundaries + [n]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sig[a:b] = levels[i]
    sig += rng.normal(0, noise, n)
    return np.stack([sig, sig * 0.5, sig * 0.2], axis=1)


def make_keypoints(n: int, speed_px_per_s: float = 40.0,
                   missing: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """匀速移动的关键点轨迹 (n,17,3)，missing 帧为全 0（未检出）。返回 (kps, bbox)。"""
    t = np.arange(n) / 5.0
    kps = np.zeros((n, 17, 3), dtype=np.float32)
    kps[:, :, 0] = 100.0 + speed_px_per_s * t[:, None]
    kps[:, :, 1] = 80.0 + speed_px_per_s * 0.6 * t[:, None]
    kps[:, :, 2] = 0.9
    bbox = np.stack([
        kps[:, 0, 0] - 50, kps[:, 0, 1] - 30,
        kps[:, 0, 0] + 50, kps[:, 0, 1] + 170,
    ], axis=1).astype(np.float32)
    if missing:
        kps[missing] = 0.0
        bbox[missing] = np.nan
    return kps, bbox


def write_synthetic_video(path, seconds: int = 20, fps: int = 30,
                          width: int = 320, height: int = 240) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for i in range(seconds * fps):
        frame = np.full((height, width, 3), (i * 3 % 256, 100, 150), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_fake_npz(path, n: int = 100, missing: list[int] | None = None,
                  video: str | None = None) -> str:
    kps, bbox = make_keypoints(n, missing=missing)
    np.savez_compressed(
        path,
        timestamps_ms=make_timestamps(n),
        keypoints=kps,
        bbox=bbox,
        conf=np.full(n, 0.9, dtype=np.float32),
        detected=np.asarray([i not in (missing or []) for i in range(n)]),
        frame_width=320,
        frame_height=240,
        sample_fps=5.0,
        video=video or "fake.mp4",
        model="fake",
        created="2026-11-06T00:00:00",
    )
    return str(path)


# ---------- 特征工程 ----------

def test_build_features_imputes_missing_frames():
    """未检出帧应被插值填充且被标记，特征全程有限值。"""
    n = 50
    kps, bbox = make_keypoints(n, missing=[10, 11, 12, 30])
    feats, imputed = build_features(kps, bbox, make_timestamps(n))
    assert feats.shape == (n, 3)
    assert np.isfinite(feats).all()
    assert set(np.flatnonzero(imputed).tolist()) == {10, 11, 12, 30}


def test_build_features_responds_to_motion_change():
    """静止→高速运动，速度特征应显著升高（尺度归一化后的量纲变化）。"""
    n = 60
    kps_slow, bbox_slow = make_keypoints(n, speed_px_per_s=5.0)
    kps_fast, bbox_fast = make_keypoints(n, speed_px_per_s=200.0)
    ts = make_timestamps(n)
    slow, _ = build_features(kps_slow, bbox_slow, ts)
    fast, _ = build_features(kps_fast, bbox_fast, ts)
    assert fast[10:, 0].mean() > 3 * slow[10:, 0].mean()


# ---------- 变化点检测 ----------

def test_detect_finds_two_changes_within_tolerance():
    n = 600  # 2 分钟 @5fps
    truth = [200, 400]
    feats = step_features(n, truth, levels=[0.0, 4.0, 0.0])
    ts = make_timestamps(n)
    cps = detect_change_points(feats, ts, min_segment_s=2.0, penalty=15.0)
    assert len(cps) == 2
    for c, t in zip(sorted(cps), truth):
        assert abs(c - t) <= 10  # ±2 秒


def test_detect_single_change():
    n = 300
    feats = step_features(n, [150], levels=[0.0, 5.0])
    ts = make_timestamps(n)
    cps = detect_change_points(feats, ts, min_segment_s=2.0, penalty=15.0)
    assert len(cps) == 1
    assert abs(cps[0] - 150) <= 10


def test_detect_no_change_returns_empty():
    n = 200
    rng = np.random.default_rng(0)
    feats = rng.normal(0, 0.1, (n, 3))
    ts = make_timestamps(n)
    cps = detect_change_points(feats, ts, min_segment_s=2.0, penalty=50.0)
    assert cps == []


# ---------- 短段合并 ----------

def test_merge_removes_short_segment():
    # 长 20s | 短 4s | 长 36s @5fps，min 5s → 短段并入邻段
    n = 300
    feats = step_features(n, [100, 120], levels=[0.0, 6.0, 0.0])
    ts = make_timestamps(n)
    merged = merge_short_segments(feats, ts, [100, 120], min_segment_s=5.0)
    assert 100 in merged and 120 not in merged


def test_merge_respects_min_segment_parameter():
    n = 300
    feats = step_features(n, [100, 120], levels=[0.0, 6.0, 0.0])
    ts = make_timestamps(n)
    kept = merge_short_segments(feats, ts, [100, 120], min_segment_s=1.0)
    assert 100 in kept and 120 in kept
    merged = merge_short_segments(feats, ts, [100, 120], min_segment_s=5.0)
    assert merged != kept


def test_merge_result_all_segments_meet_minimum():
    n = 400
    feats = step_features(n, [50, 80, 200, 260, 300],
                          levels=[0.0, 5.0, 0.0, 4.0, 1.0, 3.0])
    ts = make_timestamps(n)
    merged = merge_short_segments(feats, ts, [50, 80, 200, 260, 300],
                                  min_segment_s=5.0)
    bounds = [0] + sorted(merged)
    edges = bounds + [n]
    for a, b in zip(edges[:-1], edges[1:]):
        assert (b - a) * 0.2 >= 5.0 - 1e-9


# ---------- 端到端落盘（合成视频 + 假 npz） ----------

@pytest.fixture
def e2e_env(tmp_path):
    video = tmp_path / "demo.mp4"
    write_synthetic_video(video, seconds=20)
    npz = tmp_path / "demo.mp4.poses.npz"
    make_fake_npz(npz, n=100, missing=[40, 41], video=str(video))
    return video, npz, tmp_path


def test_analyze_end_to_end(e2e_env):
    video, npz, tmp_path = e2e_env
    out = tmp_path / "demo.mp4.scenes.json"
    result = analyze(npz, out=out, min_segment_s=1.0)
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["video"] == str(video)
    assert len(data["scenes"]) >= 1
    starts = [s["start_ms"] for s in data["scenes"]]
    assert starts == sorted(starts) and starts[0] == 0
    assert data["duration_ms"] >= starts[-1]
    for i, scene in enumerate(data["scenes"], start=1):
        assert scene["index"] == i
        assert scene["name"] == f"动作 {i}"
        thumb = tmp_path / scene["thumbnail"]
        assert thumb.is_file() and thumb.stat().st_size > 0


def test_analyze_default_out_is_sidecar(e2e_env):
    video, npz, tmp_path = e2e_env
    analyze(npz, min_segment_s=1.0)
    sidecar = tmp_path / "demo.mp4.scenes.json"
    assert sidecar.is_file()
    thumbs_dir = tmp_path / "demo.mp4.thumbnails"
    assert thumbs_dir.is_dir()
    assert list(thumbs_dir.glob("scene-*.jpg"))


def test_scene_thumbnail_ms_matches_start(e2e_env):
    """缩略图与时间戳严格一致：thumbnail_ms 存在且与 start_ms 差 ≤ 200ms。"""
    video, npz, tmp_path = e2e_env
    data = analyze(npz, min_segment_s=1.0)
    assert data["scenes"]
    for scene in data["scenes"]:
        assert "thumbnail_ms" in scene
        assert abs(scene["thumbnail_ms"] - scene["start_ms"]) <= 200


def test_cli_with_npz_input(e2e_env, capsys):
    video, npz, tmp_path = e2e_env
    rc = main([npz, "--min-segment-s", "1"])
    assert rc == 0
    assert (tmp_path / "demo.mp4.scenes.json").is_file()


def test_cli_with_video_input(e2e_env, tmp_path):
    video, _, tmp_path = e2e_env
    rc = main([str(video), "--min-segment-s", "1"])
    assert rc == 0
    assert (tmp_path / "demo.mp4.scenes.json").is_file()
