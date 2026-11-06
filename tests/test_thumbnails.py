"""票 04：缩略图静态端点测试。"""

from urllib.parse import quote


def test_thumbnail_served_200_jpeg(analyzed_sample, client):
    resp = client.get(f"/thumbnails/{quote('sample.mp4.thumbnails')}/scene-01.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 100


def test_thumbnail_rejects_path_traversal(client):
    assert client.get("/thumbnails/..%2Fsecret.jpg").status_code == 404
    assert client.get(
        f"/thumbnails/{quote('..%2F..')}%2Ftests%2Fconftest.jpg"
    ).status_code == 404


def test_thumbnail_missing_404(analyzed_sample, client):
    assert client.get(
        f"/thumbnails/{quote('sample.mp4.thumbnails')}/scene-99.jpg"
    ).status_code == 404
    assert client.get("/thumbnails/no-such.mp4.thumbnails/scene-01.jpg").status_code == 404


def test_thumbnail_only_serves_jpg(analyzed_sample, client):
    """缩略图端点不得顺带暴露 sidecar json 等其他文件。"""
    resp = client.get(f"/thumbnails/{quote('sample.mp4.scenes.json')}")
    assert resp.status_code == 404
