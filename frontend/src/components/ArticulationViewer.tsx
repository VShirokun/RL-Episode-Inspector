// Generic articulation viewer (viewer.type === "articulation3d").
//
// Renders an entire robot from recorded per-body poses. Each rigid body is an
// Object3D whose transform is driven by the pose columns; its child is either
// the real mesh (GLB loaded from /api/assets, "models" mode — the default) or a
// proxy cube ("cubes" mode, or when a body has no mesh / the GLB fails to load).
// Bones connect each body to its parent; point markers (reach target) are drawn
// too. Driven entirely by metadata.viewer — robot-agnostic.
//
// Recorded poses are sim-frame (z-up, Isaac); a group rotated -90° about X maps
// that to Three.js y-up without per-body axis math.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clampFrame } from "../playback/frameSync";
import { usePlaybackStore } from "../playback/playbackStore";
import type { BodySpec, MarkerSpec } from "../types/signal";

export function ArticulationViewer() {
  const mountRef = useRef<HTMLDivElement>(null);
  // Re-create the scene when the render mode flips (models <-> cubes).
  const renderMode = usePlaybackStore((s) => s.renderMode);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const loaded0 = usePlaybackStore.getState().loaded;
    const bodies: BodySpec[] = loaded0?.metadata.viewer.bodies ?? [];
    const markers: MarkerSpec[] = loaded0?.metadata.viewer.markers ?? [];
    const upAxis = loaded0?.metadata.viewer.up_axis ?? "z";
    // "quaternion": apply the recorded body quaternion (sim-frame poses, e.g.
    // Franka). "bone": orient each body's mesh long axis along its bone toward
    // its child joint — used when recorded rotations are in a different frame
    // than the geometry (retargeted humanoid mocap: PhysX body frame is rotated
    // ~90° from the authored mesh frame).
    const orientMode = loaded0?.metadata.viewer.orient_mode ?? "quaternion";
    // First child of each body (for bone orientation); -1 if it's a leaf.
    const childIdx: number[] = bodies.map(() => -1);
    bodies.forEach((b, i) => {
      if (b.parent >= 0 && childIdx[b.parent] < 0) childIdx[b.parent] = i;
    });

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x10131a);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    camera.position.set(1.4, 1.0, 1.4);
    // antialias off + capped pixel ratio: MSAA and extra pixels are very
    // expensive under software WebGL (llvmpipe/SwiftShader), which is what
    // rasterizes the detailed robot meshes on the CPU main thread. This keeps
    // input responsive even with the full meshes.
    const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0.4, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(2, 4, 3);
    scene.add(dir);
    const grid = new THREE.GridHelper(2, 20, 0x335, 0x223);
    scene.add(grid);

    let needsRender = true;
    let lastFrame = -1;

    const root = new THREE.Group();
    if (upAxis === "z") root.rotation.x = -Math.PI / 2;
    scene.add(root);

    // "models" => solid geometry (real meshes where available, else solid
    // capsule limbs + joint spheres so a meshless robot like the MuJoCo humanoid
    // — which is built from capsules in the sim — still looks like a real figure).
    // "cubes" => lightweight boxes + thin bone lines.
    const solid = renderMode === "models";
    const loader = new GLTFLoader();
    const jointMat = new THREE.MeshStandardMaterial({ color: 0x9fb3d1, metalness: 0.2, roughness: 0.6 });
    const rootMat = new THREE.MeshStandardMaterial({ color: 0x8899aa, metalness: 0.2, roughness: 0.6 });
    const limbMat = new THREE.MeshStandardMaterial({ color: 0x6f86b8, metalness: 0.15, roughness: 0.75 });

    // Per-body unit vector of the mesh's longest local axis, filled when its GLB
    // loads; used only in "bone" mode to align the geometry to the bone.
    const longAxis: (THREE.Vector3 | null)[] = bodies.map(() => null);
    const computeLongAxis = (obj: THREE.Object3D): THREE.Vector3 => {
      const size = new THREE.Box3().setFromObject(obj).getSize(new THREE.Vector3());
      if (size.x >= size.y && size.x >= size.z) return new THREE.Vector3(1, 0, 0);
      if (size.y >= size.z) return new THREE.Vector3(0, 1, 0);
      return new THREE.Vector3(0, 0, 1);
    };

    // One group per body; child is a real mesh, a solid joint sphere, or a cube.
    const bodyGroups = bodies.map((b, i) => {
      const g = new THREE.Group();
      root.add(g);
      const addBox = () => {
        g.add(new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.05),
          new THREE.MeshStandardMaterial({ color: i === 0 ? 0x8899aa : 0x4f9dff, metalness: 0.2, roughness: 0.6 })));
        needsRender = true;
      };
      const addJoint = () => {
        g.add(new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 16), b.parent < 0 ? rootMat : jointMat));
        needsRender = true;
      };
      if (solid && b.mesh) {
        loader.load(`/api/assets/${b.mesh}`, (gltf) => {
          g.add(gltf.scene);
          longAxis[i] = computeLongAxis(gltf.scene);
          needsRender = true;
        }, undefined, () => addJoint()); // GLB missing/failed -> solid joint
      } else if (solid) {
        addJoint(); // meshless body in models mode -> solid joint sphere
      } else {
        addBox(); // cubes mode
      }
      return g;
    });

    // Bones: solid capsule/cylinder limbs (models mode, meshless bodies) or thin
    // lines (cubes mode). Meshed bodies (e.g. Franka) get no bone.
    const boneMat = new THREE.LineBasicMaterial({ color: 0x9fb3d1 });
    type Bone = { kind: "cyl"; obj: THREE.Mesh } | { kind: "line"; obj: THREE.Line } | null;
    const bones: Bone[] = bodies.map((b) => {
      if (b.parent < 0) return null;
      if (solid && !b.mesh) {
        const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1, 12), limbMat);
        root.add(cyl);
        return { kind: "cyl", obj: cyl };
      }
      if (!solid) {
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geom, boneMat);
        root.add(line);
        return { kind: "line", obj: line };
      }
      return null;
    });

    const markerMeshes = markers.map((mk) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.03, 16, 16),
        new THREE.MeshBasicMaterial({ color: mk.color ?? "#4ecb71" }),
      );
      root.add(m);
      return m;
    });

    // Auto-frame from the robot's extent at FRAME 0 (its actual size, not the
    // whole trajectory — a walking/dancing figure translates a lot, which would
    // frame it tiny). The follow-cam then keeps the moving root centered.
    // Positions are sim z-up; map to three (x, z, -y) like `root`.
    if (loaded0 && bodies.length > 0) {
      const lo = [Infinity, Infinity, Infinity];
      const hi = [-Infinity, -Infinity, -Infinity];
      for (const bd of bodies) {
        bd.pos.forEach((col, axis) => {
          const v = (loaded0.columns[col]?.[0] as number) ?? NaN;
          if (!Number.isFinite(v)) return;
          if (v < lo[axis]) lo[axis] = v;
          if (v > hi[axis]) hi[axis] = v;
        });
      }
      if (Number.isFinite(lo[0])) {
        const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2;
        const radius = Math.max(0.4, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
        const tx = cx, ty = cz, tz = -cy; // sim (z-up) -> three (y-up)
        controls.target.set(tx, ty, tz);
        const d = radius * 3.5; // pull back enough to clear the robot's own meshes
        camera.position.set(tx + d * 0.7, ty + d * 0.6, tz + d * 0.7);
        camera.near = Math.max(0.01, d * 0.01);
        camera.far = d * 30;
        camera.updateProjectionMatrix();
        // size the floor grid to the scene and place it under the motion (y=0 = ground)
        const span = Math.max(2, Math.ceil(radius * 3));
        scene.remove(grid);
        const sized = new THREE.GridHelper(span, span * 2, 0x335, 0x223);
        sized.position.set(tx, 0, tz);
        scene.add(sized);
      }
    }

    const resize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      needsRender = true;
    };
    resize();

    const tmp = new THREE.Vector3();
    const boneDir = new THREE.Vector3();
    const vSelf = new THREE.Vector3();
    const vOther = new THREE.Vector3();
    const followTarget = new THREE.Vector3();
    const followDelta = new THREE.Vector3();
    const UP = new THREE.Vector3(0, 1, 0); // CylinderGeometry axis
    const rootIdx = Math.max(0, bodies.findIndex((b) => b.parent < 0)); // follow this body
    const MIN_DRAW_MS = 1000 / 40; // cap GL draws at ~40 fps (eases software GL)
    let lastDraw = 0;
    let running = false;
    let raf = 0;

    const loop = (now: number) => {
      const { loaded, currentFrame, isPlaying } = usePlaybackStore.getState();
      const moved = controls.update();
      const idx = loaded ? clampFrame(currentFrame, loaded.metadata.num_frames) : -1;
      const dirty = isPlaying || moved || needsRender || idx !== lastFrame;
      if (loaded && dirty && now - lastDraw >= MIN_DRAW_MS) {
        lastDraw = now;
        lastFrame = idx;
        needsRender = false;
        const num = (c: string) => (loaded.columns[c]?.[idx] as number) ?? 0;
        const bodyPos = (j: number, out: THREE.Vector3) =>
          out.set(num(bodies[j].pos[0]), num(bodies[j].pos[1]), num(bodies[j].pos[2]));
        bodies.forEach((b, i) => {
          const g = bodyGroups[i];
          g.position.set(num(b.pos[0]), num(b.pos[1]), num(b.pos[2]));
          const la = longAxis[i];
          if (orientMode === "bone" && la) {
            // Orient the mesh's long axis along the bone: toward the child joint
            // if this body has one, else away from the parent (leaf bodies).
            vSelf.copy(g.position);
            const other = childIdx[i] >= 0 ? childIdx[i] : b.parent;
            if (other >= 0) {
              bodyPos(other, vOther);
              if (childIdx[i] >= 0) boneDir.subVectors(vOther, vSelf);
              else boneDir.subVectors(vSelf, vOther);
              if (boneDir.lengthSq() > 1e-9) g.quaternion.setFromUnitVectors(la, boneDir.normalize());
            }
          } else {
            // recorded quaternion is (w,x,y,z); Three.js wants (x,y,z,w)
            g.quaternion.set(num(b.quat[1]), num(b.quat[2]), num(b.quat[3]), num(b.quat[0]));
          }
        });
        bodies.forEach((b, i) => {
          const bone = bones[i];
          if (!bone || b.parent < 0) return;
          const c = bodyGroups[i].position;
          const p = bodyGroups[b.parent].position;
          if (bone.kind === "line") {
            const pos = bone.obj.geometry.attributes.position as THREE.BufferAttribute;
            pos.setXYZ(0, c.x, c.y, c.z);
            pos.setXYZ(1, p.x, p.y, p.z);
            pos.needsUpdate = true;
          } else {
            // orient/scale a unit (Y-axis) cylinder to span from body to parent
            boneDir.subVectors(p, c);
            const len = boneDir.length() || 1e-6;
            bone.obj.position.copy(c).addScaledVector(boneDir, 0.5);
            bone.obj.quaternion.setFromUnitVectors(UP, boneDir.normalize());
            bone.obj.scale.set(1, len, 1);
          }
        });
        markers.forEach((mk, i) => {
          tmp.set(num(mk.pos[0]), num(mk.pos[1]), num(mk.pos[2]));
          markerMeshes[i].position.copy(tmp);
        });
        // Follow-cam: keep the root body centered (so a walking/dancing figure
        // stays in view). Translate camera + target by the root's movement; user
        // orbit still works. Static-root robots (e.g. the reach arm) are a no-op.
        followTarget.copy(bodyGroups[rootIdx].position);
        root.localToWorld(followTarget);
        followDelta.subVectors(followTarget, controls.target);
        camera.position.add(followDelta);
        controls.target.copy(followTarget);
        renderer.render(scene, camera);
      }
      // Keep the rAF loop alive only while there's something to animate; once
      // idle (paused, settled, nothing pending) STOP it so the tab does zero
      // work — no 60 fps wakeups that would keep the GPU/compositor busy.
      if (isPlaying || moved || needsRender || idx !== lastFrame) {
        raf = requestAnimationFrame(loop);
      } else {
        running = false;
      }
    };
    const ensureRunning = () => {
      if (!running) {
        running = true;
        raf = requestAnimationFrame(loop);
      }
    };
    const requestRender = () => {
      needsRender = true;
      ensureRunning();
    };

    // Wake the loop on camera interaction, playback ticks, seeks, and resizes.
    const ro = new ResizeObserver(() => {
      resize();
      ensureRunning();
    });
    ro.observe(mount);
    controls.addEventListener("start", requestRender);
    controls.addEventListener("change", requestRender);
    const unsubscribe = usePlaybackStore.subscribe(requestRender);
    ensureRunning();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.removeEventListener("start", requestRender);
      controls.removeEventListener("change", requestRender);
      unsubscribe();
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [renderMode]);

  return <div className="viewer3d" ref={mountRef} data-testid="viewer3d" />;
}
