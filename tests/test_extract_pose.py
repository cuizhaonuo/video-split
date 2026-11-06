"""ticket-02 单元测试：主角选择 / 重匹配 / 时间戳采样 / npz 持久化 / 叠加渲染。

不打真实 YOLO 模型；detector 用假实现注入。
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from pipeline.extract_pose import (
    Detection,
    choose_main_person,
    iter_frames_at_fps,
    iou,
    run_extraction,
)
from pipeline.render_overlay import _nearest_pose_index, render_overlay

W, H = 640, 480


def bbox(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def fake_kps(x=0.0, y=0.0, conf=0.9):
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 0] = x
    k[:, 1] = y
    k[:, 2] = conf
    return k


def make_det(b, score=0.9):
    return Detection(tuple(b), fake_kps((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), score)


# ---------- iou ----------

def test_iou_identical_and_disjoint():
    a = bbox(0, 0, 10, 10)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, bbox(20, 20, 30, 30)) == 0.0


# ---------- 主角选择：显著性 ----------

def test_choose_prefers_larger_bbox():
    small = bbox(0, 0, 60, 120)     # 靠边的小人
    big = bbox(260, 100, 380, 400)  # 居中的大人
    idx = choose_main_person([small, big], [0.9, 0.9], (W, H))
    assert idx == 1


def test_choose_center_weighting():
    # 两个面积相同的人，居中的胜出
    a = bbox(10, 100, 130, 400)      # 靠左
    b = bbox(260, 100, 380, 400)     # 居中
    idx = choose_main_person([a, b], [0.9, 0.9], (W, H))
    assert idx == 1


def test_choose_empty_returns_none():
    assert choose_main_person([], [], (W, H)) is None


# ---------- 重匹配：位置 + 体型 ----------

def test_prev_bbox_keeps_same_person_over_larger_newcomer():
    prev = bbox(100, 100, 220, 400)
    same = bbox(102, 104, 218, 398)    # 主角移动了一点点
    newcomer = bbox(400, 80, 620, 470) # 更大、更远
    idx = choose_main_person([newcomer, same], [0.9, 0.9], (W, H), prev_bbox=prev)
    assert idx == 1


def test_rematch_after_full_occlusion_picks_near_prev_position():
    prev = bbox(280, 120, 360, 380)
    # 遮挡若干帧后重新出现：一个在原位置附近（体型相近），一个在别处且更大
    near = bbox(300, 130, 372, 385)
    far_big = bbox(20, 40, 200, 470)
    idx = choose_main_person([far_big, near], [0.9, 0.9], (W, H), prev_bbox=prev)
    assert idx == 1


def test_long_gap_falls_back_to_prominence():
    prev = bbox(280, 120, 360, 380)  # 主角早已消失，bbox 全部离得很远
    a = bbox(10, 10, 30, 60)
    b = bbox(240, 100, 400, 460)     # 显著性更高的新人
    idx = choose_main_person([a, b], [0.9, 0.9], (W, H), prev_bbox=prev)
    assert idx == 1


# ---------- 时间戳采样（真实视频 pts，合成视频） ----------

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """白底 + 匀速下落矩形，30fps、3 秒。"""
    path = tmp_path_factory.mktemp("video") / "synth.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    assert vw.isOpened()
    for i in range(90):
        frame = np.full((H, W, 3), 255, dtype=np.uint8)
        y = int(i * 4) % (H - 40)
        cv2.rectangle(frame, (280, y), (360, y + 40), (0, 0, 0), -1)
        vw.write(frame)
    vw.release()
    return path


def test_iter_frames_timestamps_monotonic_and_rate(synthetic_video):
    path = synthetic_video
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    samples = list(iter_frames_at_fps(cap, 5.0))
    cap.release()
    ts = [t for t, _ in samples]
    assert len(samples) >= 13  # 3 秒 @5fps ≈ 15
    assert all(b > a for a, b in zip(ts, ts[1:])), "时间戳必须严格递增"
    # 实际采样速率接近 5fps
    span = (ts[-1] - ts[0]) / 1000.0
    rate = (len(ts) - 1) / span if span > 0 else 0
    assert 4.0 <= rate <= 6.0


# ---------- run_extraction 持久化（假 detector） ----------

class FakeDetector:
    """规则假检测：跟踪下落矩形；每隔 17 帧「丢失」一次，模拟遮挡 3 帧。"""

    def __init__(self, w=W, h=H, step=4):
        self.i = 0
        self.w, self.h, self.step = w, h, step

    def detect(self, frame):
        i = self.i
        self.i += 1
        # i 为采样调用序号（每样本 +1）；i%7==3 模拟主角短暂丢失
        if i % 7 == 3:
            return []
        y = int((i * self.step) % (self.h - 40))
        return [make_det(bbox(280, y, 360, y + 40), score=0.95)]


def test_run_extraction_writes_npz(tmp_path, synthetic_video):
    video = synthetic_video
    det = FakeDetector()
    out = tmp_path / "synth.mp4.poses.npz"
    info = run_extraction(video, det, out, fps=5.0)

    assert out.exists()
    data = np.load(out)
    for key in ("timestamps_ms", "keypoints", "bbox", "conf", "detected"):
        assert key in data.files

    ts = data["timestamps_ms"]
    n = len(ts)
    assert n == info["frames"] > 0
    assert np.all(np.diff(ts) > 0), "时间戳严格单调递增"
    assert data["keypoints"].shape == (n, 17, 3)
    assert data["bbox"].shape == (n, 4)
    assert data["detected"].dtype == bool

    det_mask = data["detected"]
    assert det_mask.any() and not det_mask.all(), "假 detector 应有丢失帧"
    # 丢失帧：bbox 全 NaN、conf 0、keypoints 全 0
    assert np.all(np.isnan(data["bbox"][~det_mask]))
    assert np.all(data["conf"][~det_mask] == 0)
    assert np.all(data["keypoints"][~det_mask] == 0)
    # 检出帧：bbox 有效且置信度 > 0
    assert np.all(~np.isnan(data["bbox"][det_mask]))
    assert np.all(data["conf"][det_mask] > 0)


def test_nearest_pose_index_tolerance():
    ts = np.array([0.0, 200.0, 400.0, 600.0])
    assert _nearest_pose_index(ts, 195.0) == 1
    assert _nearest_pose_index(ts, 405.0) == 2
    assert _nearest_pose_index(ts, 1000.0) is None  # 超出 150ms 容忍
    assert _nearest_pose_index(np.array([]), 10.0) is None


def test_render_overlay_smoke(synthetic_video, tmp_path):
    video = synthetic_video
    out_npz = tmp_path / "synth.mp4.poses.npz"
    run_extraction(video, FakeDetector(), out_npz, fps=5.0)
    out_mp4 = tmp_path / "overlay.mp4"
    path = render_overlay(video, out_npz, out_mp4, start=0.0, duration=3.0)
    assert path.exists() and path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (H, W)
