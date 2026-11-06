from pathlib import Path


def test_root_returns_200_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_root_lists_fixture_videos(client):
    html = client.get("/").text
    assert "sample.mp4" in html
    assert "训练视频.mp4" in html


def test_root_links_to_watch_page(client, video_dir: Path):
    """每个条目都应有指向 /watch/<name> 的链接（URL 编码后）。"""
    from urllib.parse import quote

    html = client.get("/").text
    assert f'href="/watch/{quote("sample.mp4")}"' in html
    assert f'href="/watch/{quote("训练视频.mp4")}"' in html
