from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rl_episode_inspector.fake_data import generate_fake_cartpole_episodes
from rl_episode_inspector.server import create_app


@pytest.fixture
def client(tmp_path):
    episodes_dir = tmp_path / "episodes"
    generate_fake_cartpole_episodes(episodes_dir, num_episodes=6, seed=21, max_frames=120)
    return TestClient(create_app(episodes_dir))


@pytest.fixture
def empty_client(tmp_path):
    return TestClient(create_app(tmp_path / "empty"))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_list_episodes(client):
    data = client.get("/api/episodes").json()
    assert len(data["episodes"]) == 6
    # sorted by return descending
    returns = [e["episode_return"] for e in data["episodes"]]
    assert returns == sorted(returns, reverse=True)


def test_metadata_endpoint(client):
    episodes = client.get("/api/episodes").json()["episodes"]
    episode_id = episodes[0]["episode_id"]
    meta = client.get(f"/api/episodes/{episode_id}/metadata").json()
    assert meta["episode_id"] == episode_id
    assert meta["viewer"]["type"] == "cartpole"
    assert any(s["name"] == "reward_step_total" for s in meta["signals"])


def test_frames_endpoint_and_slicing(client):
    episode_id = client.get("/api/episodes").json()["episodes"][0]["episode_id"]
    full = client.get(f"/api/episodes/{episode_id}/frames").json()
    assert full["count"] > 0
    assert "cart_position" in full["columns"]

    sliced = client.get(f"/api/episodes/{episode_id}/frames?start=0&end=10").json()
    assert sliced["count"] == min(10, full["count"])


def test_frames_column_filter(client):
    episode_id = client.get("/api/episodes").json()["episodes"][0]["episode_id"]
    resp = client.get(f"/api/episodes/{episode_id}/frames?names=pole_angle").json()
    cols = set(resp["columns"].keys())
    assert "pole_angle" in cols
    assert "frame_index" in cols  # required cols always included


def test_signals_endpoint(client):
    episode_id = client.get("/api/episodes").json()["episodes"][0]["episode_id"]
    all_signals = client.get(f"/api/episodes/{episode_id}/signals").json()
    assert len(all_signals["signals"]) > 5

    filtered = client.get(
        f"/api/episodes/{episode_id}/signals?names=reward_step_total,reward_alive_weighted"
    ).json()
    names = {s["name"] for s in filtered["signals"]}
    assert names == {"reward_step_total", "reward_alive_weighted"}
    assert set(filtered["series"].keys()) == names
    assert len(filtered["frame_index"]) == len(filtered["series"]["reward_step_total"])


def test_ranking_endpoint(client):
    for mode in ("best", "worst", "median"):
        resp = client.get(f"/api/ranking?mode={mode}")
        assert resp.status_code == 200
        assert "episode_return" in resp.json()


def test_ranking_invalid_mode(client):
    assert client.get("/api/ranking?mode=banana").status_code == 422


def test_unknown_episode_404(client):
    assert client.get("/api/episodes/cartpole_999999/metadata").status_code == 404


def test_path_traversal_rejected(client):
    # encoded traversal should never escape; expect 400 (invalid id) or 404
    for bad in ["..%2F..%2Fetc%2Fpasswd", "..", "%2e%2e"]:
        resp = client.get(f"/api/episodes/{bad}/metadata")
        assert resp.status_code in (400, 404), (bad, resp.status_code)


def test_empty_directory(empty_client):
    assert empty_client.get("/api/episodes").json()["episodes"] == []
    assert empty_client.get("/api/ranking?mode=best").status_code == 404
