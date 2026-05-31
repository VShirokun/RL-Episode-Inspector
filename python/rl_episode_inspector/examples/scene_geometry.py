"""Automatic, exact geometry export from an Isaac Lab articulation to GLB.

Given a composed USD stage and a robot's body prims, this reproduces *every*
visual geometry of *every* body — triangle meshes AND USD primitive shapes
(Capsule, Sphere, Cylinder, Cube, Cone) — baked into each body's local frame and
written as one GLB per body. The viewer then places each body at its recorded
pose, reconstructing the robot exactly as it is in sim. No manual configuration:
the caller passes the robot's prim path + body names and gets back a
``{body_name: "<key>/<body>.glb"}`` mapping to hand to the recorder.

Requires pxr (USD) + trimesh; meant to run inside the Isaac env (where the stage
is composed and assets are cached). Materials are not exported (uniform color);
geometry is exact. See docs/isaac_lab_integration.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _triangulate(counts, indices) -> list[list[int]]:
    """Fan-triangulate USD faceVertexCounts/Indices."""
    tris: list[list[int]] = []
    pos = 0
    for c in counts:
        face = indices[pos : pos + c]
        for k in range(1, c - 1):
            tris.append([face[0], face[k], face[k + 1]])
        pos += c
    return tris


# Rotate a Z-axis-aligned primitive to the USD prim's `axis` attribute.
def _axis_rotation(axis: str) -> np.ndarray:
    if axis == "X":  # Z -> X  (rotate +90° about Y)
        return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    if axis == "Y":  # Z -> Y  (rotate -90° about X)
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    return np.eye(3)


def _primitive_mesh(prim: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Tessellate a USD primitive Gprim into (vertices, faces). None if unknown."""
    import trimesh
    from pxr import UsdGeom

    t = prim.GetTypeName()
    tm = None
    axis = "Z"
    if t == "Capsule":
        g = UsdGeom.Capsule(prim)
        r = float(g.GetRadiusAttr().Get())
        h = float(g.GetHeightAttr().Get())
        axis = g.GetAxisAttr().Get() or "Z"
        tm = trimesh.creation.capsule(height=h, radius=r)  # along Z, centered
    elif t == "Sphere":
        r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        tm = trimesh.creation.icosphere(subdivisions=2, radius=r)
    elif t == "Cylinder":
        g = UsdGeom.Cylinder(prim)
        r = float(g.GetRadiusAttr().Get())
        h = float(g.GetHeightAttr().Get())
        axis = g.GetAxisAttr().Get() or "Z"
        tm = trimesh.creation.cylinder(radius=r, height=h)  # along Z, centered
    elif t == "Cube":
        s = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        tm = trimesh.creation.box(extents=(s, s, s))
    elif t == "Cone":
        g = UsdGeom.Cone(prim)
        r = float(g.GetRadiusAttr().Get())
        h = float(g.GetHeightAttr().Get())
        axis = g.GetAxisAttr().Get() or "Z"
        tm = trimesh.creation.cone(radius=r, height=h)
        tm.apply_translation((0, 0, -h / 2.0))  # trimesh cone base at z=0 -> center
    else:
        return None
    verts = np.asarray(tm.vertices, dtype=float) @ _axis_rotation(axis).T
    return verts, np.asarray(tm.faces, dtype=np.int64)


def _mesh_geometry(prim: Any) -> tuple[np.ndarray, np.ndarray] | None:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not idx:
        return None
    faces = np.array(_triangulate(list(counts), list(idx)), dtype=np.int64)
    return np.array(pts, dtype=float), faces


def _is_collision(prim: Any) -> bool:
    p = str(prim.GetPath()).lower()
    return "collision" in p


def export_articulation_meshes(
    stage: Any,
    root_prim_path: str,
    body_names: list[str],
    out_dir: str | Path,
    robot_key: str,
    diag: list[str] | None = None,
) -> dict[str, str]:
    """Export one GLB per body of the articulation; return {body: rel_glb_path}.

    Walks the composed stage under ``root_prim_path`` (descending into instance
    proxies), assigns each visual geom to its owning rigid body, bakes it into
    the body-local frame, and writes ``<out_dir>/<robot_key>/<body>.glb``.
    """
    import trimesh
    from pxr import Gf, Usd, UsdGeom

    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.XformCache()
    body_set = set(body_names)
    out_root = Path(out_dir) / robot_key
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise ValueError(f"Invalid root prim path: {root_prim_path}")

    def owning_body(prim: Any) -> str | None:
        cur = prim
        while cur and cur.IsValid():
            if cur.GetName() in body_set:
                return cur.GetName()
            cur = cur.GetParent()
        return None

    # body name -> world transform (rest pose), and accumulated geometry
    body_world: dict[str, Gf.Matrix4d] = {}
    geo: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {b: [] for b in body_names}

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        t = prim.GetTypeName()
        if t not in ("Mesh", "Capsule", "Sphere", "Cylinder", "Cube", "Cone"):
            continue
        if _is_collision(prim):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        body = owning_body(prim)
        if body is None:
            continue
        result = _mesh_geometry(prim) if t == "Mesh" else _primitive_mesh(prim)
        if result is None:
            continue
        verts, faces = result

        if body not in body_world:
            body_prim = prim
            while body_prim.GetName() != body:
                body_prim = body_prim.GetParent()
            body_world[body] = cache.GetLocalToWorldTransform(body_prim)
        gpw = cache.GetLocalToWorldTransform(prim)
        # body_pos_w/quat_w (what the recorder stores) is a RIGID, scale-free
        # frame. If the body/link prim carries a scale in its xform (Franka links
        # have a cm->m scale; the humanoid feet scale a unit Cube), dividing by
        # the body's *full* transform would cancel that scale and bloat the mesh
        # (Franka 100x; feet -> 2 m). Orthonormalize to drop the body scale so the
        # visual scale stays baked into the geometry.
        body_inv = Gf.Matrix4d(body_world[body]).GetOrthonormalized().GetInverse()
        rel = gpw * body_inv
        m = np.array(rel, dtype=float)  # row-vector convention: v' = [v,1] @ m
        raw_lo, raw_hi = verts.min(0), verts.max(0)
        verts = (verts @ m[:3, :3] + m[3, :3]) * stage_mpu
        if diag is not None and len(diag) < 12:
            gw = np.array(gpw, dtype=float)
            bw = np.array(Gf.Matrix4d(body_world[body]), dtype=float)
            diag.append(
                f"{body}/{prim.GetName()} type={t} mpu={stage_mpu} "
                f"raw_bbox=[{np.round(raw_lo,3)}..{np.round(raw_hi,3)}] "
                f"baked=[{np.round(verts.min(0),3)}..{np.round(verts.max(0),3)}] "
                f"rel_diag={np.round(np.diag(m)[:3],4)} "
                f"gpw_diag={np.round(np.diag(gw)[:3],4)} bw_diag={np.round(np.diag(bw)[:3],4)}"
            )
        geo[body].append((verts, faces))

    out_root.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, str] = {}
    for body, parts in geo.items():
        if not parts:
            continue
        offset = 0
        V, F = [], []
        for v, f in parts:
            V.append(v)
            F.append(f + offset)
            offset += len(v)
        vertices = np.vstack(V)
        faces = np.vstack(F)
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        tm.visual = trimesh.visual.ColorVisuals(
            mesh=tm, vertex_colors=np.tile([180, 195, 215, 255], (len(vertices), 1))
        )
        tm.export(str(out_root / f"{body}.glb"), file_type="glb")
        meshes[body] = f"{robot_key}/{body}.glb"
    return meshes
