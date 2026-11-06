from urllib.parse import quote

import pytest


@pytest.fixture
def video_size(video_dir):
    return (video_dir / "sample.mp4").stat().st_size


def stream(client, path="sample.mp4", **headers):
    return client.get(f"/video/{quote(path)}", headers=headers)


def test_stream_without_range_returns_200_full(client, video_size):
    resp = stream(client)
    assert resp.status_code == 200
    assert resp.headers["accept-ranges"] == "bytes"
    assert int(resp.headers["content-length"]) == video_size
    assert len(resp.content) == video_size


def test_stream_with_range_returns_206_and_content_range(client, video_size):
    resp = stream(client, range="bytes=100-199")
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 100-199/{video_size}"
    assert int(resp.headers["content-length"]) == 100
    assert len(resp.content) == 100


def test_stream_range_open_ended_goes_to_eof(client, video_size):
    resp = stream(client, range=f"bytes={video_size - 50}-")
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes {video_size - 50}-{video_size - 1}/{video_size}"
    assert len(resp.content) == 50


def test_stream_range_suffix_returns_last_n_bytes(client, video_size):
    resp = stream(client, range="bytes=-50")
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes {video_size - 50}-{video_size - 1}/{video_size}"
    assert len(resp.content) == 50


def test_stream_range_end_clamped_to_eof(client, video_size):
    resp = stream(client, range=f"bytes=0-{video_size + 1000}")
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 0-{video_size - 1}/{video_size}"
    assert len(resp.content) == video_size


def test_stream_invalid_range_start_beyond_eof_returns_416(client, video_size):
    resp = stream(client, range=f"bytes={video_size + 10}-")
    assert resp.status_code == 416
    assert resp.headers["content-range"] == f"bytes */{video_size}"


def test_stream_malformed_range_returns_416(client, video_size):
    resp = stream(client, range="bytes=abc-def")
    assert resp.status_code == 416


def test_stream_range_content_matches_full_body_slice(client, video_size):
    full = stream(client).content
    ranged = stream(client, range="bytes=200-299").content
    assert ranged == full[200:300]


def test_stream_missing_video_returns_404(client):
    assert stream(client, "no-such.mp4").status_code == 404


def test_stream_rejects_path_traversal(client):
    resp = client.get("/video/..%2F..%2Ftests%2Fconftest.py")
    assert resp.status_code == 404
