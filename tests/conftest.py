import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _write_video(path: Path, width: int = 320, height: int = 240, seconds: int = 2, fps: int = 30) -> None:
    """用 OpenCV 生成一段可解码的小 mp4（测试 fixture，不提交二进制）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter 打开失败: {path}"
    for i in range(seconds * fps):
        frame = np.full((height, width, 3), (i * 2 % 256, 100, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0


def make_keypoints(n_frames: int, fps: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """合成姿态轨迹：前半段缓慢漂移，后半段快速摆动（强烈的运动变化）。

    返回 (keypoints (N,17,3), bbox (N,4))，conf 恒 0.9（全程检出，无需 YOLO）。
    """
    t = np.arange(n_frames) / fps
    slow = np.arange(n_frames) < n_frames // 2
    x = np.where(slow, 100.0 + 2.0 * t, 160.0 + 40.0 * np.sin(8.0 * np.pi * t))
    y = 120.0 + 5.0 * np.sin(2.0 * np.pi * t)
    offsets = np.array([-30.0, -15.0, 0.0, 15.0, 30.0] * 4)[:17]
    xy = np.empty((n_frames, 17, 2))
    xy[..., 0] = x[:, None] + offsets[None, :]
    xy[..., 1] = y[:, None]
    conf = np.full((n_frames, 17), 0.9)
    keypoints = np.concatenate([xy, conf[..., None]], axis=-1)
    bbox = np.stack([x - 50, y - 100, x + 50, y + 100], axis=-1)
    return keypoints, bbox


def write_poses_npz(video_path: Path, n_frames: int = 50, fps: float = 5.0) -> Path:
    """为视频写一个手工构造的 <视频名>.poses.npz（10 秒 @5fps 合成轨迹）。"""
    keypoints, bbox = make_keypoints(n_frames, fps)
    timestamps = np.arange(n_frames) * (1000.0 / fps)
    npz_path = Path(str(video_path) + ".poses.npz")
    np.savez(
        npz_path,
        timestamps_ms=timestamps,
        keypoints=keypoints,
        bbox=bbox,
        video=str(video_path),
    )
    return npz_path


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    """包含一个英文命名和一个中文命名的 fixture 视频，模拟真实目录（每测试隔离）。"""
    d = tmp_path / "videos"
    d.mkdir()
    _write_video(d / "sample.mp4")
    _write_video(d / "训练视频.mp4")
    return d


@pytest.fixture
def analyzed_sample(video_dir: Path) -> Path:
    """sample.mp4 已用合成姿态跑过 pipeline.segment.analyze()（缓存 + 缩略图）。"""
    from pipeline.segment import analyze

    npz = write_poses_npz(video_dir / "sample.mp4")
    analyze(npz, min_segment_s=1.0)
    return video_dir / "sample.mp4"


@pytest.fixture
def client(video_dir: Path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(video_dir=video_dir)) as c:
        yield c
