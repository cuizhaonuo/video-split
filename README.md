# video-split 服务（票 01：服务与播放页骨架）

本地 FastAPI 服务：浏览器打开即见 `videos/` 目录下的视频列表，点开任一视频进入播放页，通过 HTTP Range（206 Partial Content）流式播放，支持拖动进度条。

## 运行

使用仓库共享 venv：

```bash
D:/vscode-project/video-split/venv/Scripts/python.exe -m uvicorn app.main:create_app --factory --port 8000
```

然后浏览器打开 <http://127.0.0.1:8000/>。

## 配置

视频目录默认为仓库根的 `videos/`（扫描 `.mp4/.mov/.avi/.mkv/.webm`），可用环境变量覆盖：

```bash
VIDEO_DIR=D:/some/other/videos D:/vscode-project/video-split/venv/Scripts/python.exe -m uvicorn app.main:create_app --factory --port 8000
```

## 路由

| 路由 | 说明 |
|------|------|
| `GET /` | 视频列表页 |
| `GET /watch/<视频名>` | 播放页（`<video>` 播放器） |
| `GET /video/<视频名>` | 视频流端点，支持 `Range` 请求（206/416），`Accept-Ranges: bytes` |

## 测试

```bash
cd <仓库根或 worktree 根>
D:/vscode-project/video-split/venv/Scripts/python.exe -m pytest -q
```

测试视频 fixture 由 OpenCV 在运行时生成（`tests/conftest.py`），不提交二进制文件。
