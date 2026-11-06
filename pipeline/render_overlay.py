"""骨架叠加渲染：把 .poses.npz 轨迹画回原视频，输出 mp4。

CLI:
    python -m pipeline.render_overlay <video> <poses.npz> [--start 秒] [--duration 秒] [-o out.mp4]

默认输出到 artifacts/ 下。逐帧顺序解码，按时间戳就近匹配姿态记录
（不假设帧序号对齐），超过容忍窗口的帧不画骨架。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# COCO 17 点骨架连接
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
    (5, 11), (6, 12), (11, 12),          # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
]

DEFAULT_KEYPOINT_CONF = 0.3
MATCH_TOLERANCE_MS = 150.0  # 与 5fps 采样间隔(200ms)同量级


def _nearest_pose_index(ts_ms_arr: np.ndarray, ts: float) -> int | None:
    """二分找时间戳最近的姿态记录，超出容忍窗口返回 None。"""
    if len(ts_ms_arr) == 0:
        return None
    i = int(np.searchsorted(ts_ms_arr, ts))
    cand = [j for j in (i - 1, i) if 0 <= j < len(ts_ms_arr)]
    best = min(cand, key=lambda j: abs(ts_ms_arr[j] - ts))
    if abs(ts_ms_arr[best] - ts) > MATCH_TOLERANCE_MS:
        return None
    return best


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, bbox: np.ndarray,
              keypoint_conf: float = DEFAULT_KEYPOINT_CONF) -> None:
    """在帧上原地画主角 bbox + 骨架。keypoints: (17,3) x,y,conf。"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)

    def pt(i: int):
        x, y, c = keypoints[i]
        return (int(round(float(x))), int(round(float(y)))), float(c)

    for a, b in EDGES:
        pa, ca = pt(a)
        pb, cb = pt(b)
        if ca >= keypoint_conf and cb >= keypoint_conf:
            cv2.line(frame, pa, pb, (0, 255, 255), 2)
    for i in range(17):
        p, c = pt(i)
        if c >= keypoint_conf:
            cv2.circle(frame, p, 3, (0, 165, 255), -1)


def render_overlay(
    video: str | Path,
    poses: str | Path,
    out_path: str | Path,
    start: float | None = None,
    duration: float | None = None,
    keypoint_conf: float = DEFAULT_KEYPOINT_CONF,
) -> Path:
    data = np.load(poses)
    ts_ms = data["timestamps_ms"].astype(np.float64)
    kps = data["keypoints"]
    bboxes = data["bbox"]
    detected = data["detected"]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    drawn = 0
    end = None if duration is None else (start if start is not None else 0.0) + duration

    while True:
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = ts / 1000.0
        if start is not None and t < start:
            continue
        if end is not None and t > end:
            break
        idx = _nearest_pose_index(ts_ms, ts)
        if idx is not None and detected[idx]:
            draw_pose(frame, kps[idx], bboxes[idx], keypoint_conf)
            drawn += 1
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建输出视频: {out_path}")
        writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    if writer is None:
        raise RuntimeError("指定范围内没有输出任何帧（检查 --start/--duration）")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="渲染骨架叠加视频")
    ap.add_argument("video")
    ap.add_argument("poses", help="extract_pose 输出的 .npz")
    ap.add_argument("--start", type=float, default=None, help="起始秒")
    ap.add_argument("--duration", type=float, default=None, help="时长秒（默认到结尾）")
    ap.add_argument("-o", "--out", default=None, help="输出 mp4 路径（默认 artifacts/<视频名>.overlay.mp4）")
    ap.add_argument("--keypoint-conf", type=float, default=DEFAULT_KEYPOINT_CONF)
    args = ap.parse_args(argv)

    if args.out:
        out = Path(args.out)
    else:
        stem = Path(args.video).name
        out = Path("artifacts") / f"{stem}.overlay.mp4"
    path = render_overlay(
        args.video, args.poses, out,
        start=args.start, duration=args.duration, keypoint_conf=args.keypoint_conf,
    )
    print(f"完成 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
