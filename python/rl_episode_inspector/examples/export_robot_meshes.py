"""Export per-body robot meshes from USD to GLB for the browser viewer.

Standalone (pxr + trimesh) — does NOT need the Isaac Sim app, so it's fast to
run. For each body it bakes the visual mesh into the body-local frame and writes
one ``<body>.glb``. The articulation viewer loads these and drives each body's
transform from the recorded pose, reconstructing the real robot (with the cube
proxies as a fallback when meshes aren't available or "cubes" mode is selected).

Example (Franka, from its cached Isaac Lab asset)::

    isaaclab-env/bin/python -m rl_episode_inspector.examples.export_robot_meshes \
        --usd-dir <ISAAC_ASSETS>/Robots/FrankaEmika/Props \
        --bodies panda_link0,panda_link1,...,panda_hand,panda_leftfinger,panda_rightfinger \
        --out-dir sample_data/reach/assets/franka

Each body ``<name>`` is read from ``<usd-dir>/<name>.usd``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _triangulate(counts, indices):
    """Fan-triangulate USD faceVertexCounts/Indices into a list of [i,j,k]."""
    tris = []
    pos = 0
    for c in counts:
        face = indices[pos : pos + c]
        for k in range(1, c - 1):
            tris.append([face[0], face[k], face[k + 1]])
        pos += c
    return tris


def export_body(usd_path: Path, out_path: Path) -> tuple[int, int]:
    """Convert one link USD to a GLB (body-local frame). Returns (verts, tris)."""
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.XformCache()

    all_v: list = []
    all_f: list = []
    offset = 0
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        m = cache.GetLocalToWorldTransform(prim)
        verts = np.array([[*m.Transform(p)] for p in pts], dtype=np.float64) * scale
        tris = np.array(_triangulate(list(counts), list(idx)), dtype=np.int64) + offset
        all_v.append(verts)
        all_f.append(tris)
        offset += len(verts)

    if not all_v:
        raise ValueError(f"No mesh geometry found in {usd_path}")

    vertices = np.vstack(all_v)
    faces = np.vstack(all_f)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Uniform light-steel material so the whole robot reads as one object.
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.tile([180, 195, 215, 255], (len(vertices), 1))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path), file_type="glb")
    return len(vertices), len(faces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-body robot meshes (USD -> GLB).")
    parser.add_argument("--usd-dir", required=True, help="dir containing <body>.usd files")
    parser.add_argument("--bodies", required=True, help="comma-separated body names")
    parser.add_argument("--out-dir", required=True, help="output dir for <body>.glb files")
    args = parser.parse_args()

    usd_dir = Path(args.usd_dir)
    out_dir = Path(args.out_dir)
    bodies = [b.strip() for b in args.bodies.split(",") if b.strip()]

    total = 0
    for body in bodies:
        usd_path = usd_dir / f"{body}.usd"
        if not usd_path.exists():
            print(f"  SKIP {body}: {usd_path} not found")
            continue
        v, f = export_body(usd_path, out_dir / f"{body}.glb")
        total += 1
        print(f"  {body}.glb: {v} verts, {f} tris")
    print(f"Exported {total} body meshes to {out_dir}")


if __name__ == "__main__":
    main()
