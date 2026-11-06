"""FastAPI 应用：视频列表页、播放页、支持 HTTP Range 的流式端点。"""

import os
import re
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", str(_REPO_ROOT / "videos")))

_PAGE_STYLE = """
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem;
    background: #fafafa; color: #1a1a1a;
    font-family: system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
    line-height: 1.5;
  }
  main { max-width: 960px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin: 0 0 1.5rem; }
  .video-list { list-style: none; margin: 0; padding: 0; }
  .video-list li { margin: 0; }
  .video-list a {
    display: block; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    background: #ffffff; border: 1px solid #ddd; border-radius: 8px;
    color: #1a1a1a; text-decoration: none;
  }
  .video-list a:hover { border-color: #888; background: #f0f0f0; }
  .player video { width: 100%; max-height: 70vh; background: #000; border-radius: 8px; }
  .back { display: inline-block; margin-bottom: 1rem; color: #1a1a1a; }
"""


def _list_videos(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        return []
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file())


def _resolve_video(video_dir: Path, name: str) -> Path:
    """把 URL 中的视频名解析为目录内的文件，拒绝路径遍历。"""
    path = (video_dir / name).resolve()
    if not str(path).startswith(str(video_dir.resolve()) + os.sep) or not path.is_file():
        raise HTTPException(status_code=404, detail="video not found")
    return path


def create_app(video_dir: Path | None = None) -> FastAPI:
    video_dir = video_dir or _DEFAULT_VIDEO_DIR
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        items = "".join(
            f'<li><a href="/watch/{quote(p.name)}">{p.name}</a></li>'
            for p in _list_videos(video_dir)
        )
        return (
            "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>视频列表</title>"
            f"<style>{_PAGE_STYLE}</style></head><body><main>"
            "<h1>视频列表</h1>"
            f"<ul class='video-list'>{items}</ul>"
            "</main></body></html>"
        )

    @app.get("/watch/{name:path}", response_class=HTMLResponse)
    def watch(name: str) -> str:
        video = _resolve_video(video_dir, name)
        src = f"/video/{quote(video.name)}"
        return (
            "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{video.name}</title>"
            f"<style>{_PAGE_STYLE}</style></head><body><main class='player'>"
            f"<a class='back' href='/'>&#8592; 返回列表</a>"
            f"<h1>{video.name}</h1>"
            f"<video controls preload='metadata' src=\"{src}\"></video>"
            "</main></body></html>"
        )

    @app.get("/video/{name:path}")
    def stream(name: str, request: Request) -> StreamingResponse:
        return _stream_file(_resolve_video(video_dir, name), request)

    return app


_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _iter_file(path: Path, start: int, end: int):
    """从 start 到 end（含）逐块读取文件。"""
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _stream_file(path: Path, request: Request) -> StreamingResponse:
    file_size = path.stat().st_size
    base_headers = {"Accept-Ranges": "bytes"}

    range_header = request.headers.get("range")
    if not range_header:
        return StreamingResponse(
            _iter_file(path, 0, file_size - 1),
            media_type="video/mp4",
            headers={**base_headers, "Content-Length": str(file_size)},
        )

    match = _RANGE_RE.match(range_header.replace(" ", ""))
    start_str, end_str = match.groups() if match else ("", "")
    if not match or (start_str == "" and end_str == ""):
        return _range_not_satisfiable(file_size, base_headers)

    if start_str == "":
        # 后缀范围: bytes=-N 表示最后 N 字节
        length = min(int(end_str), file_size)
        start, end = file_size - length, file_size - 1
    else:
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)

    if start > end or start >= file_size:
        return _range_not_satisfiable(file_size, base_headers)

    return StreamingResponse(
        _iter_file(path, start, end),
        status_code=206,
        media_type="video/mp4",
        headers={
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(end - start + 1),
        },
    )


def _range_not_satisfiable(file_size: int, base_headers: dict) -> StreamingResponse:
    return StreamingResponse(
        iter(()),
        status_code=416,
        media_type="video/mp4",
        headers={**base_headers, "Content-Range": f"bytes */{file_size}"},
    )
