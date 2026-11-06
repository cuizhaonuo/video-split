from urllib.parse import quote


def test_watch_returns_200_html(client):
    resp = client.get(f"/watch/{quote('sample.mp4')}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_watch_has_video_element_pointing_to_stream(client):
    html = client.get(f"/watch/{quote('sample.mp4')}").text
    assert "<video" in html
    assert 'controls' in html
    assert 'src="/video/sample.mp4"' in html


def test_watch_uses_unencoded_name_as_page_title(client):
    html = client.get(f"/watch/{quote('sample.mp4')}").text
    assert "sample.mp4" in html


def test_watch_missing_video_returns_404(client):
    resp = client.get("/watch/no-such-file.mp4")
    assert resp.status_code == 404


def test_watch_rejects_path_traversal(client):
    resp = client.get("/watch/..%2F..%2Ftests%2Fconftest.py")
    assert resp.status_code == 404


# ---------------------------- 票 04：场景导航 UI ----------------------------

def test_watch_has_scene_navigation_ui(client):
    """播放页应包含缩略图条、分析控件与未分析占位。"""
    html = client.get(f"/watch/{quote('sample.mp4')}").text
    for frag in [
        "id='video'",
        "id='scene-bar'",
        "id='placeholder'",
        "id='analyze-btn'",
        "id='start-analyze-btn'",
        "id='min-segment'",
    ]:
        assert frag in html, f"缺少 {frag}"


def test_watch_js_targets_scenes_api(client):
    """内联 JS 应请求 scenes 缓存接口与 analyze 接口，并有当前段高亮逻辑。"""
    html = client.get(f"/watch/{quote('sample.mp4')}").text
    assert "'/api/videos/'" in html
    assert "'/scenes'" in html
    assert "'/analyze'" in html
    assert "encodeURIComponent" in html  # 中文名在运行时编码
    assert "timeupdate" in html  # 当前段高亮逻辑
    assert "currentTime" in html  # 点击缩略图跳转


def test_watch_chinese_name_passed_to_js(client):
    encoded = quote("训练视频.mp4")
    html = client.get(f"/watch/{quote('训练视频.mp4')}").text
    # 视频流地址 URL 编码
    assert f'src="/video/{encoded}"' in html
    # 视频名以 JSON 字符串传给内联 JS（运行时再 encodeURIComponent）
    assert 'var NAME = "训练视频.mp4";' in html
