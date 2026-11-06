# video-split 服务（票 01-04：服务骨架 + 流式播放 + 场景导航界面）

本地 FastAPI 服务：浏览器打开即见 `videos/` 目录下的视频列表，点开任一视频进入播放页，通过 HTTP Range（206 Partial Content）流式播放，支持拖动进度条。播放页底部有场景缩略图条（票 04）：每段起始帧缩略图 + 「动作 N · mm:ss」，点击跳转播放，当前段高亮；支持一键（重新）分析。

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
| `GET /watch/<视频名>` | 播放页：播放器 + 场景缩略图条 + 「重新分析」控件 |
| `GET /video/<视频名>` | 视频流端点，支持 `Range` 请求（206/416），`Accept-Ranges: bytes` |
| `GET /api/videos/<视频名>/scenes` | 读取分段缓存 sidecar `<视频名>.scenes.json`；无缓存返回 `{"analyzed": false, "scenes": []}`，**绝不触发分析**（缓存语义） |
| `POST /api/videos/<视频名>/analyze` | 触发分析，body 可选 `{"min_segment_s": 5}`；用 subprocess 跑 `python -m pipeline.segment <视频名>.poses.npz --min-segment-s N`，同步等待（全片约 7s）后返回新 scenes；`poses.npz` 缺失返回 409 |
| `GET /thumbnails/<相对路径>` | 静态提供缩略图 JPG（路径即 scenes.json 的 `thumbnail` 字段，如 `<视频名>.mp4.thumbnails/scene-01.jpg`），带路径校验防目录遍历 |

## 场景导航界面（票 04）

播放页（light 主题，内联 CSS/JS，无外部依赖）行为：

- 加载时 `GET scenes`：有缓存直接渲染缩略图条（二次打开不重复分析）；无缓存显示「未分析」占位与「开始分析」按钮。
- 缩略图条横向滚动，每项 = 缩略图 + 「动作 N · mm:ss」；点击 seek 到该段 `start_ms/1000` 并播放。
- `timeupdate` 事件驱动当前段高亮（描边）并自动滚动进入视野。
- 右上角「最短段长（秒）」输入 + 「重新分析」按钮：分析中按钮禁用并显示「分析中…」，完成后按新粒度刷新缩略图条。
- 视频名（含中文/括号/空格）全程 URL 编码传递。

## 测试

```bash
cd <仓库根或 worktree 根>
D:/vscode-project/video-split/venv/Scripts/python.exe -m pytest -q
```

测试视频 fixture 由 OpenCV 在运行时生成（`tests/conftest.py`），不提交二进制文件。
