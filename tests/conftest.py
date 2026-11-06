import sys
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _write_video(path: Path, width: int = 320, height: int = 240, seconds: int = 2, fps: int = 30) -> None:
    """用 OpenCV 生成一段可解码的小 mp4（测试 fixture，不提交二进制）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter 打开失败: {path}"
    for i in range(seconds * fps):
        import numpy as np

        frame = np.full((height, width, 3), (i * 2 % 256, 100, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory) -> Path:
    """包含一个英文命名和一个中文命名的 fixture 视频，模拟真实目录。"""
    d = tmp_path_factory.mktemp("videos")
    _write_video(d / "sample.mp4")
    _write_video(d / "训练视频.mp4")
    return d


@pytest.fixture
def client(video_dir: Path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(video_dir=video_dir)) as c:
        yield c
