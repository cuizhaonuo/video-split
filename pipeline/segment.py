"""动作分段：姿态轨迹 → 运动特征 → 无监督变化点检测 → scenes.json + 缩略图。

CLI:
    python -m pipeline.segment <视频或 poses.npz> [--min-segment-s 5] [-o scenes.json]
                               [--penalty P] [--smooth-window W]

输入是 ticket-02 的 ``<视频名>.poses.npz``（见 pipeline/extract_pose.py 字段定义）。

处理流程（ADR-0001，无监督）:
    1. 特征工程: 关节速度（按躯干长归一化）+ 关节角度变化；未检出帧线性插值并标记；
       滑动窗口移动平均平滑。
    2. 变化点检测: ruptures Pelt + RBF kernel，penalty / min_segment_s 可调。
    3. 边界对齐: 变化点索引 → timestamps_ms → 原视频时间（ms）。
    4. 短段合并: 短于 min_segment_s 的段并入平均特征最相近的邻段。
    5. 每段起始帧抽缩略图 JPG，写 sidecar scenes.json。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MIN_SEGMENT_S = 5.0
DEFAULT_PENALTY = 15.0
DEFAULT_SMOOTH_WINDOW = 5  # 1s @5fps

# COCO 17 关键点角度三元组（肘/肩/髋/膝，左右）
ANGLE_TRIPLETS = [
    (5, 7, 9), (6, 8, 10),    # 肘
    (7, 5, 11), (8, 6, 12),   # 肩
    (5, 11, 13), (6, 12, 14), # 髋
    (11, 13, 15), (12, 14, 16),  # 膝
]

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


# ---------------------------------------------------------------- 特征工程

def _impute_positions(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """对无效关节（conf<=0 或未检出帧）做逐关节线性插值；全程无效则用 0。

    xy: (N,17,2) float64（可能是全 0），conf: (N,17)。
    """
    n = xy.shape[0]
    valid = conf > 0
    out = xy.copy()
    idx = np.arange(n)
    for j in range(xy.shape[1]):
        for c in range(2):
            v = valid[:, j]
            if v.all():
                continue
            if not v.any():
                out[:, j, c] = 0.0
                continue
            out[:, j, c] = np.interp(idx, idx[v], xy[v, j, c])
    return out


def _body_scale(xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """每帧身体尺度：躯干长（肩中点-髋中点），缺失时退回 bbox 高，再退回中位数。"""
    shoulder_c = (xy[:, L_SHOULDER, :2] + xy[:, R_SHOULDER, :2]) / 2.0
    hip_c = (xy[:, L_HIP, :2] + xy[:, R_HIP, :2]) / 2.0
    torso = np.linalg.norm(shoulder_c - hip_c, axis=1)
    bbox_h = bbox[:, 3] - bbox[:, 1]  # 未检出帧为 NaN
    scale = np.where(torso > 1.0, torso, bbox_h)
    med = float(np.nanmedian(scale)) if np.isfinite(scale).any() else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1.0), scale, med)
    return scale


def _joint_angles(xy: np.ndarray) -> np.ndarray:
    """各三元组的关节角（弧度，[0, π]），无效关节为 NaN。返回 (N, len(ANGLE_TRIPLETS))。"""
    n = xy.shape[0]
    angles = np.full((n, len(ANGLE_TRIPLETS)), np.nan)
    for k, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        v1 = xy[:, a, :2] - xy[:, b, :2]
        v2 = xy[:, c, :2] - xy[:, b, :2]
        denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        cosang = np.clip(
            np.sum(v1 * v2, axis=1) / np.where(denom > 1e-6, denom, 1.0), -1.0, 1.0
        )
        ok = denom > 1e-6
        angles[ok, k] = np.arccos(cosang[ok])
    return angles


def build_features(
    keypoints: np.ndarray, bbox: np.ndarray, timestamps_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """姿态轨迹 → 运动特征 (N,3) 与未检出帧标记 (N,) bool。

    特征: [平均归一化关节速度, 最大归一化关节速度, 平均关节角变化(弧度/s)]。
    """
    keypoints = keypoints.astype(np.float64)
    n = len(keypoints)
    conf = keypoints[:, :, 2]
    frame_detected = (conf > 0).any(axis=1)
    imputed = ~frame_detected

    xy = _impute_positions(keypoints[:, :, :2], conf)
    scale = _body_scale(xy, bbox.astype(np.float64))

    dt = np.diff(timestamps_ms) / 1000.0
    dt = np.where(dt > 1e-6, dt, 1e-6)
    disp = np.linalg.norm(np.diff(xy, axis=0), axis=2)  # (N-1,17) 像素
    speed = disp / dt[:, None] / scale[:-1, None]  # 归一化速度
    speed = np.vstack([np.zeros((1, 17)), speed])

    angles = _joint_angles(xy)
    d_angle = np.abs(np.diff(angles, axis=0))  # (N-1, K)
    mean_d_angle = np.zeros(n)
    if n > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 行（未检出段）
            mean_d_angle[1:] = np.nan_to_num(np.nanmean(d_angle, axis=1) / dt)

    feats = np.stack(
        [speed.mean(axis=1), speed.max(axis=1), mean_d_angle], axis=1
    )
    return feats, imputed


def smooth_features(features: np.ndarray, window: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """居中滑动窗口移动平均（边缘收缩窗口）。"""
    if window <= 1:
        return features.copy()
    n, d = features.shape
    out = np.empty_like(features)
    half = window // 2
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = features[a:b].mean(axis=0)
    return out


# ---------------------------------------------------------------- 变化点检测

def _effective_fps(timestamps_ms: np.ndarray) -> float:
    dt = np.diff(timestamps_ms)
    med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 200.0
    return 1000.0 / med


def detect_change_points(
    features: np.ndarray,
    timestamps_ms: np.ndarray,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
    penalty: float = DEFAULT_PENALTY,
) -> list[int]:
    """在平滑特征上跑 ruptures Pelt(rbf)，返回内部变化点帧索引（升序）。"""
    import ruptures as rpt

    n = len(features)
    if n < 4:
        return []
    sig = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-9)
    min_size = max(2, int(round(min_segment_s * _effective_fps(timestamps_ms))))
    algo = rpt.Pelt(model="rbf", min_size=min_size, jump=5).fit(sig)
    cps = algo.predict(pen=penalty)  # 末尾含 n
    return [int(c) for c in cps[:-1] if 0 < c < n]


def merge_short_segments(
    features: np.ndarray,
    timestamps_ms: np.ndarray,
    change_points: list[int],
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
) -> list[int]:
    """迭代移除边界以消灭短于 min_segment_s 的段；短段并入平均特征最相近的邻段。"""
    n = len(features)
    bounds = [0] + sorted(c for c in change_points if 0 < c < n)
    fps = _effective_fps(timestamps_ms)
    min_frames = max(1, int(round(min_segment_s * fps)))
    means_cache: dict[int, np.ndarray] = {}

    def seg_mean(a: int, b: int) -> np.ndarray:
        key = a * (n + 1) + b
        if key not in means_cache:
            means_cache[key] = features[a:b].mean(axis=0)
        return means_cache[key]

    while len(bounds) > 1:
        edges = bounds + [n]
        segs = list(zip(edges[:-1], edges[1:]))
        durations = [(edges[i + 1] - edges[i]) / fps for i in range(len(segs))]
        i_short = int(np.argmin(durations))
        if durations[i_short] >= min_segment_s:
            break
        # 选特征最相近的邻段；边界段只有一个邻段
        prev_i, next_i = i_short - 1, i_short + 1
        candidates = [i for i in (prev_i, next_i) if 0 <= i < len(segs)]
        if len(candidates) == 2:
            a, b = segs[i_short]
            dists = [
                float(np.linalg.norm(seg_mean(*segs[c]) - seg_mean(a, b)))
                for c in candidates
            ]
            target = candidates[int(np.argmin(dists))]
        else:
            target = candidates[0]
        # 移除短段与目标段之间的边界
        shared = max(segs[i_short][0], segs[target][0])
        bounds.remove(shared)
    return sorted(c for c in bounds if 0 < c < n)


# ---------------------------------------------------------------- 落盘

def extract_thumbnails(
    video: str | Path,
    scene_starts_ms: list[int],
    out_dir: str | Path,
) -> list[Path]:
    """对每段起始时间戳 seek 抽帧存 JPG（scene-NN.jpg，NN 从 01 起）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    paths: list[Path] = []
    try:
        for i, ts_ms in enumerate(scene_starts_ms, start=1):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = cv2.VideoCapture(str(video))  # 重新打开重试一次
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(ts_ms) - 100.0))
                ok, frame = cap.read()
            p = out_dir / f"scene-{i:02d}.jpg"
            if ok and frame is not None:
                # cv2.imwrite 在 Windows 上对非 ASCII 路径静默失败，改用 imencode
                ok_enc, buf = cv2.imencode(".jpg", frame)
                if ok_enc:
                    buf.tofile(str(p))
            paths.append(p)
    finally:
        cap.release()
    return paths


def _default_paths(npz_path: Path, video: Path) -> tuple[Path, Path]:
    return (
        Path(str(video) + ".scenes.json"),
        Path(str(video) + ".thumbnails"),
    )


def analyze(
    npz_path: str | Path,
    video_path: str | Path | None = None,
    out: str | Path | None = None,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
    penalty: float = DEFAULT_PENALTY,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    make_thumbnails: bool = True,
) -> dict:
    """完整分析：npz → 特征 → 变化点 → 短段合并 → 缩略图 + scenes.json。返回 scenes.json 内容。"""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        timestamps = z["timestamps_ms"].astype(np.float64)
        keypoints = z["keypoints"]
        bbox = z["bbox"]
        video_field = str(z["video"]) if "video" in z.files else ""

    if len(timestamps) < 2:
        raise ValueError(f"姿态轨迹为空或过短: {npz_path}")

    video = Path(video_path) if video_path else Path(video_field)
    if not video.is_file():
        raise FileNotFoundError(f"找不到视频文件（缩略图需要）: {video}")

    features, _ = build_features(keypoints, bbox, timestamps)
    smoothed = smooth_features(features, smooth_window)
    cps = detect_change_points(smoothed, timestamps, min_segment_s, penalty)
    cps = merge_short_segments(smoothed, timestamps, cps, min_segment_s)

    fps = _effective_fps(timestamps)
    bounds = [0] + cps
    edges = bounds + [len(timestamps)]
    scene_starts = [int(round(timestamps[b])) for b in bounds]
    scene_ends = [
        int(round(timestamps[e - 1] + 1000.0 / fps)) for e in edges[1:]
    ]

    out_json, thumbs_dir = _default_paths(npz_path, video)
    out_json = Path(out) if out else out_json

    thumb_rel: list[str | None] = []
    if make_thumbnails:
        extract_thumbnails(video, scene_starts, thumbs_dir)
        thumb_rel = [
            str((thumbs_dir / f"scene-{i:02d}.jpg").as_posix())
            for i in range(1, len(scene_starts) + 1)
        ]

    scenes = [
        {
            "index": i,
            "name": f"动作 {i}",
            "start_ms": s,
            "end_ms": e,
            "thumbnail": t,
        }
        for i, (s, e, t) in enumerate(
            zip(scene_starts, scene_ends, thumb_rel), start=1
        )
    ]
    data = {
        "video": str(video),
        "analyzed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "params": {
            "sample_fps": round(fps, 3),
            "min_segment_s": min_segment_s,
            "penalty": penalty,
            "smooth_window": smooth_window,
            "detector": "ruptures-pelt-rbf",
        },
        "scenes": scenes,
        "duration_ms": int(round(timestamps[-1] + 1000.0 / fps)),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        argv = [str(a) for a in argv]
    ap = argparse.ArgumentParser(
        description="无监督姿态边界检测 → scenes.json + 缩略图"
    )
    ap.add_argument("input", help="输入视频路径或 poses.npz 路径")
    ap.add_argument("--min-segment-s", type=float, default=DEFAULT_MIN_SEGMENT_S)
    ap.add_argument("--penalty", type=float, default=DEFAULT_PENALTY)
    ap.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    ap.add_argument("-o", "--out", default=None, help="输出 json 路径（默认 sidecar）")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if inp.suffix.lower() == ".npz":
        npz_path, video_path = inp, None
    else:
        npz_path = Path(str(inp) + ".poses.npz")
        video_path = inp
        if not npz_path.is_file():
            print(f"找不到姿态轨迹 {npz_path}，请先运行 python -m pipeline.extract_pose")
            return 1

    data = analyze(
        npz_path,
        video_path=video_path,
        out=args.out,
        min_segment_s=args.min_segment_s,
        penalty=args.penalty,
        smooth_window=args.smooth_window,
    )
    out_path = args.out or str(_default_paths(npz_path, Path(data["video"]))[0])
    print(
        f"完成: {len(data['scenes'])} 段, 时长 {data['duration_ms'] / 1000:.1f}s → {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
