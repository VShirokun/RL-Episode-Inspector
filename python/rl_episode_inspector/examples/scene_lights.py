"""Extract the scene lights of an Isaac Lab task from its composed USD stage.

The viewer can then light the replayed robot the same way the source sim does
(same light kinds, directions and colors), instead of a generic default rig.
Photometric units don't map 1:1 to a real-time rasterizer, so intensities are
*normalized* (relative brightness preserved, absolute values not): the brightest
key light is scaled to a sensible directional intensity and the rest follow.

Returns plain dicts (``LightSpec``-shaped) so the caller can JSON-dump them next
to the exported meshes; the recorder turns them into ``LightSpec`` objects.

Requires pxr (USD); meant to run inside the Isaac env. See docs/overview_ru.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# UsdLux light prim types -> our generic kind.
_DISTANT = ("DistantLight",)
_DOME = ("DomeLight",)
_POINT = ("SphereLight", "DiskLight", "RectLight", "CylinderLight")

# Target real-time intensities (the brightest key light maps here).
_KEY_INTENSITY = 1.1
_AMBIENT_INTENSITY = 0.55


def _color(light: Any) -> list[float]:
    from pxr import UsdLux

    attr = UsdLux.LightAPI(light).GetColorAttr() if hasattr(UsdLux, "LightAPI") else None
    if attr is None:
        attr = light.GetAttribute("inputs:color")
    c = attr.Get() if attr and attr.IsValid() else None
    if c is None:
        return [1.0, 1.0, 1.0]
    return [max(0.0, float(c[0])), max(0.0, float(c[1])), max(0.0, float(c[2]))]


def _raw_intensity(light: Any) -> float:
    """``intensity * 2**exposure`` — the photometric strength before normalizing."""
    def get(name: str, default: float) -> float:
        a = light.GetAttribute(name)
        v = a.Get() if a and a.IsValid() else None
        return float(v) if v is not None else default

    return get("inputs:intensity", 1.0) * (2.0 ** get("inputs:exposure", 0.0))


def _world_dir(light: Any, cache: Any) -> list[float]:
    """World-space direction the light travels (UsdLux lights emit along -Z)."""
    from pxr import Gf

    m = cache.GetLocalToWorldTransform(light)
    rot = Gf.Matrix4d(m).GetOrthonormalized().ExtractRotationMatrix()
    d = Gf.Vec3d(0, 0, -1) * rot  # local -Z into world
    v = np.array([d[0], d[1], d[2]], dtype=float)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 1e-9 else [0.0, 0.0, -1.0]


def _world_pos(light: Any, cache: Any) -> list[float]:
    from pxr import Gf

    m = cache.GetLocalToWorldTransform(light)
    t = Gf.Matrix4d(m).ExtractTranslation()
    return [float(t[0]), float(t[1]), float(t[2])]


def extract_stage_lights(stage: Any, diag: list[str] | None = None) -> list[dict]:
    """Return the stage's lights as ``LightSpec``-shaped dicts (normalized).

    Skips light prims under ``/Render`` settings and zero-intensity lights. If
    the stage has no usable lights, returns ``[]`` (the viewer then falls back to
    its default rig).
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    found: list[tuple[str, str, list[float], float, Any]] = []  # kind,name,color,raw,prim
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        t = prim.GetTypeName()
        if t in _DISTANT:
            kind = "directional"
        elif t in _DOME:
            kind = "hemisphere"
        elif t in _POINT:
            kind = "point"
        else:
            continue
        raw = _raw_intensity(prim)
        if raw <= 0.0:
            continue
        found.append((kind, prim.GetName(), _color(prim), raw, prim))

    if not found:
        if diag is not None:
            diag.append("no UsdLux lights found")
        return []

    # Normalize: brightest non-dome light -> _KEY_INTENSITY; dome -> ambient fill.
    key_raw = max((r for k, _, _, r, _ in found if k != "hemisphere"), default=0.0)
    dome_raw = max((r for k, _, _, r, _ in found if k == "hemisphere"), default=0.0)
    norm = max(key_raw, dome_raw, 1e-9)

    lights: list[dict] = []
    for kind, name, color, raw, prim in found:
        if kind == "hemisphere":
            intensity = _AMBIENT_INTENSITY * (raw / norm)
            lights.append({"name": name, "kind": "hemisphere", "color": color,
                           "intensity": round(intensity, 4)})
        elif kind == "directional":
            intensity = _KEY_INTENSITY * (raw / norm)
            lights.append({"name": name, "kind": "directional", "color": color,
                           "intensity": round(intensity, 4),
                           "direction": [round(v, 4) for v in _world_dir(prim, cache)]})
        else:  # point
            intensity = _KEY_INTENSITY * (raw / norm)
            lights.append({"name": name, "kind": "point", "color": color,
                           "intensity": round(intensity, 4),
                           "position": [round(v, 4) for v in _world_pos(prim, cache)]})
        if diag is not None:
            diag.append(f"light {name} type={prim.GetTypeName()} -> {kind} "
                        f"raw={round(raw, 2)} norm_intensity={round(intensity, 3)}")
    return lights
