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
