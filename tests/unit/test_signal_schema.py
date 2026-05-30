from __future__ import annotations

from rl_episode_inspector.storage import SignalKind, SignalSpec, ViewerSpec


def test_signal_defaults():
    s = SignalSpec(name="x", kind=SignalKind.state)
    assert s.dtype == "float32"
    assert s.shape == []
    assert s.unit is None
    # use_enum_values -> stored as the string value
    assert s.kind == "state"


def test_signal_serialization_roundtrip():
    s = SignalSpec(name="reward_alive_raw", kind=SignalKind.reward_raw, unit=None,
                   description="Raw alive reward")
    restored = SignalSpec.model_validate(s.model_dump())
    assert restored == s


def test_viewer_spec_default():
    v = ViewerSpec()
    assert v.type == "generic"
    assert v.state_mapping == {}
