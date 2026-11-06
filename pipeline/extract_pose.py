"""姿态提取：5fps 抽帧 + YOLO11-pose + 单人主角选择（ADR-0002）。

CLI:
    python -m pipeline.extract_pose <video> [-o out.npz] [--fps 5] [--model path]

输出 .npz（默认 sidecar ``<视频名>.poses.npz``）字段:
    timestamps_ms : (N,) float64  抽帧真实时间戳（CAP_PROP_POS_MSEC，单调递增）
    keypoints     : (N,17,3) float32  COCO 17 关键点 x,y,conf；未检出帧全 0
    bbox          : (N,4) float32  主角 bbox xyxy；未检出帧为 NaN
    conf          : (N,) float32   主角检测置信度；未检出帧为 0
    detected      : (N,) bool      该帧是否检出主角
    frame_width / frame_height / sample_fps / video / model / created
"""

from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.paths import poses_npz_path

DEFAULT_FPS = 5.0
DEFAULT_MODEL = Path("D:/vscode-project/video-split/models/yolo11n-pose.pt")
DEFAULT_MIN_CONF = 0.25

# 权重： prominence 占比（bbox 大小 + 居中程度）
_W_AREA = 0.65
_W_CENTER = 0.35
# continuity 权重（仅当存在上一帧主角 bbox 时参与打分）
_W_PROM = 0.5
_W_IOU = 0.3
_W_SIZE = 0.2


@dataclass
class Detection:
    """单个人物检测结果。"""

    bbox: tuple[float, float, float, float]  # xyxy
    keypoints: np.ndarray  # (17,3) x,y,conf
    score: float


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个 xyxy bbox 的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def prominence_score(
    bbox: np.ndarray, frame_size: tuple[int, int], max_area: float
) -> float:
    """显著性分：bbox 面积（相对当前帧最大面积）+ 画面居中程度，均归一化到 [0,1]。"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    area = max(0.0, float(w) * float(h))
    area_norm = area / max_area if max_area > 0 else 0.0

    fw, fh = frame_size
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    # 归一化到画面中心的距离（半对角线为 1）
    dx = (cx - fw / 2.0) / (fw / 2.0)
    dy = (cy - fh / 2.0) / (fh / 2.0)
    dist = float(np.hypot(dx, dy))
    centrality = max(0.0, 1.0 - dist)

    return _W_AREA * area_norm + _W_CENTER * centrality


def choose_main_person(
    boxes: list[np.ndarray],
    scores: list[float],
    frame_size: tuple[int, int],
    prev_bbox: np.ndarray | None = None,
) -> int | None:
    """按 ADR-0002 选主角。

    - 无上一帧信息：显著性（bbox 最大 + 居中加权）直接决策。
    - 有上一帧：显著性 + 与上一帧位置 IoU + bbox 高度（体型）相似度加权，
      遮挡/镜头切换后优先找回「原位置附近、体型接近」的人。
    返回选中下标；无可用人选（boxes 为空）返回 None。
    """
    if not boxes:
        return None

    areas = np.array(
        [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes], dtype=np.float64
    )
    max_area = float(areas.max()) if len(areas) else 0.0
    proms = np.array([prominence_score(b, frame_size, max_area) for b in boxes])

    if prev_bbox is None:
        total = proms
    else:
        h_prev = max(1e-6, float(prev_bbox[3] - prev_bbox[1]))
        total = np.empty(len(boxes), dtype=np.float64)
        for i, b in enumerate(boxes):
            h_i = max(1e-6, float(b[3] - b[1]))
            size_sim = float(np.exp(-abs(np.log(h_i / h_prev))))
            ov = iou(np.asarray(prev_bbox, dtype=np.float64), np.asarray(b, dtype=np.float64))
            total[i] = _W_PROM * proms[i] + _W_IOU * ov + _W_SIZE * size_sim

    return int(np.argmax(total))


def iter_frames_at_fps(cap: cv2.VideoCapture, fps: float):
    """顺序解码，按真实时间戳（CAP_PROP_POS_MSEC）每隔 1000/fps ms 产出 (ts_ms, frame)。

    不假设「第 N 个样本对应第 N 个原始帧」；时间戳直接取自容器 pts，天然单调递增。
    """
    interval = 1000.0 / float(fps)
    next_target = float("-inf")
    last_ts = -1.0
    while True:
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = float(ts)
        if ts < last_ts:  # 防御异常容器：强制单调
            ts = last_ts
        last_ts = ts
        if ts >= next_target:
            yield ts, frame
            next_target = ts + interval


class YoloPoseDetector:
    """YOLO11-pose 检测器（COCO 17 关键点）。加载本地权重，绝不触发自动下载。"""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        from ultralytics import YOLO  # 延迟导入，单测不需要 ultralytics

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO 权重不存在: {path}（请使用预下载的绝对路径）")
        self.model = YOLO(str(path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model(frame, verbose=False)[0]
        if res.boxes is None or res.keypoints is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy()
        kpts = res.keypoints.data.cpu().numpy()  # (N,17,3)
        return [
            Detection(tuple(boxes[i]), kpts[i], float(confs[i]))
            for i in range(len(boxes))
            if int(clss[i]) == 0  # 只保留 person
        ]


def run_extraction(
    video: str | Path,
    detector,
    out_path: str | Path,
    fps: float = DEFAULT_FPS,
    min_conf: float = DEFAULT_MIN_CONF,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict:
    """对视频做 5fps 姿态提取并持久化轨迹。返回结果元信息。

    start_s/end_s（秒，可选）限定处理区间；时间戳取自容器 pts，
    因此分段提取后可直接按 timestamps_ms 拼接。
    """
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_s is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)

    ts_list: list[float] = []
    kp_list: list[np.ndarray] = []
    bbox_list: list[np.ndarray] = []
    conf_list: list[float] = []
    det_list: list[bool] = []

    prev_bbox: np.ndarray | None = None
    for ts, frame in iter_frames_at_fps(cap, fps):
        if end_s is not None and ts / 1000.0 > float(end_s):
            break
        dets = [d for d in detector.detect(frame) if d.score >= min_conf]
        idx = choose_main_person(
            [d.bbox for d in dets], [d.score for d in dets], (fw, fh), prev_bbox
        )
        if idx is None:
            kp_list.append(np.zeros((17, 3), dtype=np.float32))
            bbox_list.append(np.full(4, np.nan, dtype=np.float32))
            conf_list.append(0.0)
            det_list.append(False)
        else:
            d = dets[idx]
            prev_bbox = np.asarray(d.bbox, dtype=np.float64)
            kp_list.append(d.keypoints.astype(np.float32))
            bbox_list.append(np.asarray(d.bbox, dtype=np.float32))
            conf_list.append(d.score)
            det_list.append(True)
        ts_list.append(ts)
    cap.release()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = np.asarray(ts_list, dtype=np.float64)
    np.savez_compressed(
        out_path,
        timestamps_ms=timestamps,
        keypoints=np.stack(kp_list) if kp_list else np.zeros((0, 17, 3), np.float32),
        bbox=np.stack(bbox_list) if bbox_list else np.zeros((0, 4), np.float32),
        conf=np.asarray(conf_list, dtype=np.float32),
        detected=np.asarray(det_list, dtype=bool),
        frame_width=fw,
        frame_height=fh,
        sample_fps=fps,
        video=str(video),
        model=str(getattr(detector, "model_path", "")),
        created=_dt.datetime.now().isoformat(timespec="seconds"),
    )
    return {
        "frames": len(ts_list),
        "detected": int(sum(det_list)),
        "duration_s": (timestamps[-1] / 1000.0) if len(timestamps) else 0.0,
        "out": str(out_path),
    }


def default_sidecar_path(video: str | Path) -> Path:
    """sidecar npz 路径（形状统一收口在 pipeline/paths.py）。"""
    return poses_npz_path(video)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="5fps 姿态提取 → 主角轨迹 .npz")
    ap.add_argument("video", help="输入视频路径")
    ap.add_argument("-o", "--out", default=None, help="输出 npz 路径（默认 sidecar <video>.poses.npz）")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)
    ap.add_argument("--start", type=float, default=None, help="起始秒（含），配合 --end 分块提取")
    ap.add_argument("--end", type=float, default=None, help="结束秒（含）；时间戳仍为全片绝对时间")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else default_sidecar_path(args.video)
    detector = YoloPoseDetector(args.model)
    info = run_extraction(
        args.video, detector, out, fps=args.fps, min_conf=args.min_conf,
        start_s=args.start, end_s=args.end,
    )
    print(
        f"完成: {info['frames']} 帧（检出 {info['detected']}，"
        f"覆盖率 {info['detected'] / max(1, info['frames']):.1%}），"
        f"时长 {info['duration_s']:.1f}s → {info['out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
