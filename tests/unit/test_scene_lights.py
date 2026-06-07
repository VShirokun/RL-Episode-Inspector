"""Unit tests for the pure light-normalization logic (no USD/pxr needed).

``extract_stage_lights`` itself needs a composed USD stage (Isaac only), but its
intensity model lives in ``normalize_lights``, which is pure and is where the
real subtlety — and a past bug — lives: USD photometric units aren't comparable
across light *types*, so intensities must be normalized within each kind, never
across kinds.
"""

from __future__ import annotations

from rl_episode_inspector.examples.scene_lights import (
    _AMBIENT_INTENSITY,
    _KEY_INTENSITY,
    _POINT_INTENSITY,
    normalize_lights,
)


def test_empty_in_empty_out():
    assert normalize_lights([]) == []


def test_per_kind_targets_for_single_lights():
    found = [
        {"name": "sun", "kind": "directional", "color": [1, 1, 1], "raw": 123.0,
         "direction": [0, 0, -1]},
        {"name": "bulb", "kind": "point", "color": [1, 1, 1], "raw": 9.0,
         "position": [0, 0, 2]},
        {"name": "dome", "kind": "hemisphere", "color": [0.7, 0.7, 0.7], "raw": 4000.0},
    ]
    out = {light["kind"]: light for light in normalize_lights(found)}
    # a single light of a kind maps to that kind's target regardless of raw value
    assert out["directional"]["intensity"] == _KEY_INTENSITY
    assert out["point"]["intensity"] == _POINT_INTENSITY
    assert out["hemisphere"]["intensity"] == _AMBIENT_INTENSITY


def test_huge_point_does_not_crush_dome():
    """The original bug: a SphereLight's huge raw value normalized the DomeLight
    fill to near-zero. Per-kind normalization must keep the dome at its target."""
    found = [
        {"name": "sphere", "kind": "point", "color": [1, 1, 1], "raw": 100_000.0,
         "position": [0, 0, 2.5]},
        {"name": "dome", "kind": "hemisphere", "color": [0.75, 0.75, 0.75], "raw": 2000.0},
    ]
    out = {light["kind"]: light for light in normalize_lights(found)}
    assert out["point"]["intensity"] == _POINT_INTENSITY
    assert out["hemisphere"]["intensity"] == _AMBIENT_INTENSITY  # NOT ~0


def test_within_kind_scaling_is_relative_to_brightest():
    found = [
        {"name": "bright", "kind": "directional", "color": [1, 1, 1], "raw": 100.0,
         "direction": [0, 0, -1]},
        {"name": "dim", "kind": "directional", "color": [1, 1, 1], "raw": 25.0,
         "direction": [1, 0, 0]},
    ]
    out = {light["name"]: light for light in normalize_lights(found)}
    assert out["bright"]["intensity"] == _KEY_INTENSITY          # brightest -> target
    assert out["dim"]["intensity"] == round(_KEY_INTENSITY * 0.25, 4)  # 25/100 of it


def test_geometry_and_order_preserved():
    found = [
        {"name": "p", "kind": "point", "color": [1, 1, 1], "raw": 1.0, "position": [1, 2, 3]},
        {"name": "d", "kind": "directional", "color": [1, 1, 1], "raw": 1.0,
         "direction": [0, -1, 0]},
    ]
    out = normalize_lights(found)
    assert [light["name"] for light in out] == ["p", "d"]  # input order kept
    assert out[0]["position"] == [1, 2, 3] and "direction" not in out[0]
    assert out[1]["direction"] == [0, -1, 0] and "position" not in out[1]
