"""Sidecar 路径形状的唯一出处（app 与 pipeline 共用，Standards#2）。

约定（票 02/03/04 共同遵循）:
    <视频名>.poses.npz     姿态轨迹（pipeline/extract_pose.py 产出）
    <视频名>.scenes.json   分段结果 sidecar（pipeline/segment.py 产出）
    <视频名>.thumbnails/   缩略图目录，文件名 scene-{i:02d}.jpg（i 从 01 起）
"""

from __future__ import annotations

from pathlib import Path


def poses_npz_path(video: str | Path) -> Path:
    """视频 → sidecar 姿态轨迹 npz 路径。"""
    return Path(str(video) + ".poses.npz")


def scenes_json_path(video: str | Path) -> Path:
    """视频 → sidecar 分段结果 json 路径。"""
    return Path(str(video) + ".scenes.json")


def thumbnails_dir(video: str | Path) -> Path:
    """视频 → 缩略图目录路径。"""
    return Path(str(video) + ".thumbnails")


def thumbnail_filename(i: int) -> str:
    """第 i 个场景（1 起）的缩略图文件名。"""
    return f"scene-{i:02d}.jpg"
