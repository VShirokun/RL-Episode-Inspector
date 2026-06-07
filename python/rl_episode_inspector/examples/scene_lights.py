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

# Target real-time intensities, per light kind. USD photometric units are NOT
# comparable across light types (a SphereLight's candela vs a DomeLight's nits vs
# a DistantLight's lux), so we never normalize across kinds — each kind gets a
# perceptually sensible target and raw values only scale lights *within* a kind.
# This preserves the lighting's character (where light comes from, its color)
# without one type's huge raw number drowning another to near-black.
_KEY_INTENSITY = 1.0  # directional (sun-like)
_POINT_INTENSITY = 1.0  # local point light
# Dome/hemisphere fill is the main thing that lights camera-facing surfaces in a
# simple (no-IBL) renderer, so give it enough weight that a scene lit ONLY by a
# dome + an overhead key still shows the robot clearly (matching how Isaac's dome
# fills the scene), not as a near-black silhouette.
_AMBIENT_INTENSITY = 0.9  # dome / hemisphere fill


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
    # Per-light "raw" entries (geometry already resolved). normalize_lights() —
    # which is pure / USD-free and unit-tested — turns these into final specs.
    found: list[dict] = []
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
        entry = {"name": prim.GetName(), "kind": kind, "color": _color(prim),
                 "raw": raw, "type": str(t)}
        if kind == "directional":
            entry["direction"] = [round(v, 4) for v in _world_dir(prim, cache)]
        elif kind == "point":
            entry["position"] = [round(v, 4) for v in _world_pos(prim, cache)]
        found.append(entry)

    lights = normalize_lights(found)
    if diag is not None:
        if not lights:
            diag.append("no UsdLux lights found")
        for raw_e, spec in zip(found, lights):
            diag.append(f"light {spec['name']} type={raw_e['type']} -> {spec['kind']} "
                        f"raw={round(raw_e['raw'], 2)} norm_intensity={spec['intensity']}")
    return lights


def normalize_lights(found: list[dict]) -> list[dict]:
    """Normalize raw per-light entries into ``LightSpec``-shaped dicts (pure).

    Each ``found`` entry has ``name``/``kind``/``color``/``raw`` plus ``direction``
    (directional) or ``position`` (point). Intensities are normalized **within
    each kind** to that kind's real-time target — never across kinds, because USD
    photometric units aren't comparable between light types (so a huge SphereLight
    ``raw`` can't crush a DomeLight's fill toward zero). Output order matches input.
    """
    if not found:
        return []
    target = {"directional": _KEY_INTENSITY, "point": _POINT_INTENSITY,
              "hemisphere": _AMBIENT_INTENSITY}
    kind_ref = {k: max((e["raw"] for e in found if e["kind"] == k), default=1.0)
                for k in target}
    out: list[dict] = []
    for e in found:
        kind = e["kind"]
        intensity = round(target[kind] * (e["raw"] / max(kind_ref[kind], 1e-9)), 4)
        spec = {"name": e["name"], "kind": kind, "color": e["color"], "intensity": intensity}
        if "direction" in e:
            spec["direction"] = e["direction"]
        if "position" in e:
            spec["position"] = e["position"]
        out.append(spec)
    return out
