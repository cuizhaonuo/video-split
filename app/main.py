"""FastAPI 应用：视频列表页、播放页（场景导航 UI）、流式与分段 API。"""

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_ANALYZE_TIMEOUT_S = 600  # 全片 segment 约 7s，留足余量

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
  .watch-head {
    display: flex; justify-content: space-between; align-items: center;
    gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
  }
  .watch-head h1 { font-size: 1.2rem; margin: 0; }
  .analyze-controls { display: flex; align-items: center; gap: 0.5rem; }
  .analyze-controls label { font-size: 0.9rem; color: #444; }
  .analyze-controls input {
    width: 4.5rem; padding: 0.25rem 0.4rem;
    border: 1px solid #ccc; border-radius: 6px; font: inherit;
  }
  button.action {
    padding: 0.35rem 0.9rem; border: 1px solid #ccc; border-radius: 6px;
    background: #fff; color: #1a1a1a; font: inherit; cursor: pointer;
  }
  button.action:hover:not(:disabled) { border-color: #888; background: #f0f0f0; }
  button.action:disabled { opacity: 0.5; cursor: default; }
  .placeholder {
    display: none; align-items: center; justify-content: center; gap: 1rem;
    padding: 1.5rem; margin-top: 1rem;
    background: #fff; border: 1px dashed #bbb; border-radius: 8px; color: #555;
  }
  .scene-bar { display: flex; gap: 0.75rem; overflow-x: auto; padding: 0.75rem 0.25rem; }
  .scene-item {
    flex: 0 0 auto; width: 128px; padding: 0.35rem;
    border: 1px solid #ddd; border-radius: 8px; background: #fff;
    display: flex; flex-direction: column; gap: 0.3rem;
    font: inherit; color: inherit; text-align: left; cursor: pointer;
  }
  .scene-item img {
    width: 100%; height: 72px; object-fit: cover;
    border-radius: 4px; display: block; background: #eee;
  }
  .scene-item span { font-size: 0.8rem; color: #333; }
  .scene-item:hover { border-color: #888; }
  .scene-item.active { border-color: #2563eb; box-shadow: 0 0 0 2px #bfdbfe; }
"""


_SCENE_NAV_JS = """
(function () {
  'use strict';
  var NAME = __NAME__;
  var API = '/api/videos/' + encodeURIComponent(NAME);
  var videoEl = document.getElementById('video');
  var bar = document.getElementById('scene-bar');
  var ph = document.getElementById('placeholder');

  function fmt(ms) {
    var s = Math.floor(ms / 1000);
    return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
  }

  function thumbUrl(rel) {
    return '/thumbnails/' + rel.split('/').map(encodeURIComponent).join('/');
  }

  function render(data) {
    bar.innerHTML = '';
    if (!data.analyzed || !data.scenes.length) {
      ph.style.display = 'flex';
      bar.style.display = 'none';
      return;
    }
    ph.style.display = 'none';
    bar.style.display = 'flex';
    data.scenes.forEach(function (sc) {
      var item = document.createElement('button');
      item.type = 'button';
      item.className = 'scene-item';
      item.dataset.start = sc.start_ms;
      item.dataset.end = sc.end_ms;
      var img = document.createElement('img');
      img.src = thumbUrl(sc.thumbnail);
      img.alt = sc.name;
      img.loading = 'lazy';
      var cap = document.createElement('span');
      cap.textContent = sc.name + ' \\u00b7 ' + fmt(sc.start_ms);
      item.appendChild(img);
      item.appendChild(cap);
      item.addEventListener('click', function () {
        videoEl.currentTime = sc.start_ms / 1000;
        videoEl.play();
      });
      bar.appendChild(item);
    });
  }

  function loadScenes() {
    fetch(API + '/scenes').then(function (r) { return r.json(); }).then(render);
  }

  function doAnalyze() {
    var btn = document.getElementById('analyze-btn');
    var minInput = document.getElementById('min-segment');
    var orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = '\\u5206\\u6790\\u4e2d\\u2026';
    var minSeg = parseFloat(minInput.value);
    if (!isFinite(minSeg) || minSeg <= 0) { minSeg = 5; }
    fetch(API + '/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ min_segment_s: minSeg })
    }).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return null; }).then(function (err) {
          alert('\\u5206\\u6790\\u5931\\u8d25: ' + ((err && err.detail) || r.status));
        });
      }
      return r.json().then(render);
    }).finally(function () {
      btn.disabled = false;
      btn.textContent = orig;
    });
  }

  videoEl.addEventListener('timeupdate', function () {
    var ms = videoEl.currentTime * 1000;
    var items = document.querySelectorAll('.scene-item');
    var active = null;
    items.forEach(function (it) {
      var on = ms >= +it.dataset.start && ms < +it.dataset.end;
      it.classList.toggle('active', on);
      if (on) { active = it; }
    });
    if (active) {
      active.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  });

  document.getElementById('analyze-btn').addEventListener('click', doAnalyze);
  document.getElementById('start-analyze-btn').addEventListener('click', doAnalyze);
  loadScenes();
})();
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


class AnalyzeRequest(BaseModel):
    min_segment_s: float = Field(default=5.0, gt=0)


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
        name_js = json.dumps(video.name, ensure_ascii=False).replace("</", "<\\/")
        return (
            "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(video.name)}</title>"
            f"<style>{_PAGE_STYLE}</style></head><body><main class='player'>"
            '<a class="back" href="/">&#8592; 返回列表</a>'
            "<div class='watch-head'>"
            f"<h1>{html.escape(video.name)}</h1>"
            "<div class='analyze-controls'>"
            "<label>最短段长 <input id='min-segment' type='number' min='0.5' step='0.5' value='5'> 秒</label>"
            "<button id='analyze-btn' type='button' class='action'>重新分析</button>"
            "</div></div>"
            f"<video id='video' controls preload='metadata' src=\"{src}\"></video>"
            "<div id='placeholder' class='placeholder'>该视频尚未分析"
            "<button id='start-analyze-btn' type='button' class='action'>开始分析</button></div>"
            "<div id='scene-bar' class='scene-bar'></div>"
            f"<script>{_SCENE_NAV_JS.replace('__NAME__', name_js)}</script>"
            "</main></body></html>"
        )

    @app.get("/api/videos/{name:path}/scenes")
    def get_scenes(name: str):
        """读取 sidecar 缓存；不触发分析（缓存语义）。"""
        video = _resolve_video(video_dir, name)
        sidecar = Path(str(video) + ".scenes.json")
        if not sidecar.is_file():
            return {"analyzed": False, "scenes": []}
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["analyzed"] = True
        return data

    @app.post("/api/videos/{name:path}/analyze")
    def analyze_video(name: str, body: AnalyzeRequest | None = None):
        """subprocess 跑 pipeline.segment，同步等待（全片约 7s），完成后返回新 scenes。"""
        if body is None:
            body = AnalyzeRequest()
        video = _resolve_video(video_dir, name)
        npz = Path(str(video) + ".poses.npz")
        if not npz.is_file():
            raise HTTPException(
                status_code=409,
                detail=f"缺少姿态轨迹 {npz.name}，请先运行姿态提取（python -m pipeline.extract_pose）",
            )
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pipeline.segment",
                    str(npz), "--min-segment-s", str(body.min_segment_s),
                ],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=_ANALYZE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail=f"分析超时（>{_ANALYZE_TIMEOUT_S}s）")
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"分析失败: {proc.stderr[-2000:]}")
        data = json.loads(Path(str(video) + ".scenes.json").read_text(encoding="utf-8"))
        data["analyzed"] = True
        return data

    @app.get("/thumbnails/{path:path}")
    def thumbnail(path: str):
        """静态提供缩略图（scenes.json 里 thumbnail 字段的相对路径），防目录遍历。"""
        full = (video_dir / path).resolve()
        if (
            not str(full).startswith(str(video_dir.resolve()) + os.sep)
            or full.suffix.lower() != ".jpg"
            or not full.is_file()
        ):
            raise HTTPException(status_code=404, detail="thumbnail not found")
        return FileResponse(full, media_type="image/jpeg")

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
