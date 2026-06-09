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

    // Default fallback light rig. Always built; its *visibility* is driven by
    // the store's `defaultLights` flag (toggled by a checkbox), so turning it on
    // or off needs no scene rebuild / mesh reload. When the task carries its own
    // lights (below), the user can switch this off to see only the sim lighting.
    const defaultLights = new THREE.Group();
    defaultLights.add(new THREE.AmbientLight(0xffffff, 0.55));
    defaultLights.add(new THREE.HemisphereLight(0xffffff, 0x40464f, 0.6));
    const defDir = new THREE.DirectionalLight(0xffffff, 0.9);
    defDir.position.set(3, 6, 4);
    defaultLights.add(defDir);
    defaultLights.visible = usePlaybackStore.getState().defaultLights;
    scene.add(defaultLights);
    const grid = new THREE.GridHelper(2, 20, 0x335, 0x223);
    scene.add(grid);

    let needsRender = true;
    let lastFrame = -1;
    // Async GLB loads finish *after* the render loop may have gone idle; `wake`
    // (wired to requestRender once it exists) both marks a redraw AND restarts
    // the stopped rAF loop, so a late-loading mesh actually gets drawn.
    let wake: () => void = () => {
      needsRender = true;
    };

    const root = new THREE.Group();
    if (upAxis === "z") root.rotation.x = -Math.PI / 2;
    scene.add(root);

    // Lights captured from the source sim (metadata.viewer.lights): light the
    // replay the same way the task does. Directional/point lights live under
    // `root` so the sim(z-up)->three(y-up) mapping applies to their direction/
    // position; ambient/hemisphere are orientation-free. Empty list => the
    // default rig above carries the scene.
    const simLights = loaded0?.metadata.viewer.lights ?? [];
    for (const L of simLights) {
      const c = new THREE.Color(L.color?.[0] ?? 1, L.color?.[1] ?? 1, L.color?.[2] ?? 1);
      if (L.kind === "directional") {
        const dl = new THREE.DirectionalLight(c, L.intensity);
        const d = L.direction ?? [0, 0, -1];
        // a DirectionalLight shines from its position toward the origin (its
        // default target), so place it opposite the travel direction `d`
        dl.position.set(-d[0], -d[1], -d[2]);
        root.add(dl);
      } else if (L.kind === "point") {
        const p = L.position ?? [0, 0, 0];
        const pl = new THREE.PointLight(c, L.intensity);
        // decay 0: the captured intensity is already normalized for inspection,
        // so keep it uniform — the position still sets where the light comes from
        // without the physical 1/d^2 falloff blacking out the far side of a robot.
        pl.decay = 0;
        pl.position.set(p[0], p[1], p[2]);
        root.add(pl);
      } else if (L.kind === "hemisphere") {
        // ground term kept fairly light so downward / camera-facing normals still
        // read (a near-black ground would silhouette the robot from the side)
        scene.add(new THREE.HemisphereLight(c, 0x44505f, L.intensity));
      } else {
        scene.add(new THREE.AmbientLight(c, L.intensity));
      }
    }

    // "models" => solid geometry (real meshes where available, else solid
    // capsule limbs + joint spheres so a meshless robot like the MuJoCo humanoid
    // — which is built from capsules in the sim — still looks like a real figure).
    // "cubes" => lightweight boxes + thin bone lines.
    const solid = renderMode === "models";
    const loader = new GLTFLoader();
    // Bright, distinct per-body colors — like viewers that flat-color untextured
    // meshes. Tinting each body by index makes objects clearly visible and tells
    // them apart (the exported GLBs carry no textures, and some come out dark).
    const BODY_COLORS = [0xff5d5d, 0x4f9dff, 0xffd23b, 0x4ecb71, 0xb78bff, 0xff9d3b,
                         0x3fd0c9, 0xff6fcf, 0x9ad24e, 0xffa3d1];
    const colorFor = (i: number) => BODY_COLORS[i % BODY_COLORS.length];
    // Bright flat-colored material: an emissive base so the color is always
    // visible (like viewers that flat-shade untextured meshes), plus `flatShading`
    // so lights still produce real shading even on exported meshes that ship
    // WITHOUT vertex normals (which otherwise render unlit/black).
    const bodyMat = (i: number) => {
      const c = new THREE.Color(colorFor(i));
      return new THREE.MeshStandardMaterial({
        color: c, emissive: c, emissiveIntensity: 0.35,
        metalness: 0.0, roughness: 0.75, flatShading: true,
      });
    };
    const limbMat = new THREE.MeshStandardMaterial({ color: 0x6f86b8, metalness: 0.0, roughness: 0.8 });

    // Proxy size scales with the scene extent so meshless bodies stay visible on
    // a large scene (e.g. a tennis court spanning ~20 m) without being oversized
    // on a small robot. Meshed bodies are unaffected (they use their real GLB).
    let proxySize = 0.05;
    if (loaded0 && bodies.length > 0) {
      const lo = [Infinity, Infinity, Infinity];
      const hi = [-Infinity, -Infinity, -Infinity];
      for (const b of bodies) {
        b.pos.forEach((col, ax) => {
          const v = loaded0.columns[col]?.[0] as number;
          if (Number.isFinite(v)) { lo[ax] = Math.min(lo[ax], v); hi[ax] = Math.max(hi[ax], v); }
        });
      }
      if (Number.isFinite(lo[0])) {
        const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
        proxySize = Math.min(0.5, Math.max(0.05, diag * 0.03));
      }
    }

    // One group per body; child is a real mesh, a solid joint sphere, or a cube.
    const bodyGroups = bodies.map((b, i) => {
      const g = new THREE.Group();
      root.add(g);
      const addBox = () => {
        g.add(new THREE.Mesh(new THREE.BoxGeometry(proxySize, proxySize, proxySize), bodyMat(i)));
        wake();
      };
      const addJoint = () => {
        g.add(new THREE.Mesh(new THREE.SphereGeometry(proxySize, 16, 16), bodyMat(i)));
        wake();
      };
      if (solid && b.mesh) {
        loader.load(`/api/assets/${b.mesh}`, (gltf) => {
          // Replace the textureless (sometimes dark/metallic) exported material
          // with a bright, fully lit per-body color so lighting reads clearly.
          const mat = bodyMat(i);
          gltf.scene.traverse((o) => {
            const mesh = o as THREE.Mesh;
            if (mesh.isMesh) mesh.material = mat;
          });
          g.add(gltf.scene);
          wake();
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
        const cam = loaded0.metadata.viewer.camera;
        if (cam) {
          // Park at the task's captured play viewpoint (sim z-up -> three y-up).
          camera.position.set(cam.eye[0], cam.eye[2], -cam.eye[1]);
          controls.target.set(cam.lookat[0], cam.lookat[2], -cam.lookat[1]);
          const cd = Math.hypot(cam.eye[0] - cam.lookat[0], cam.eye[1] - cam.lookat[1], cam.eye[2] - cam.lookat[2]);
          camera.near = Math.max(0.01, cd * 0.004);
          camera.far = Math.max(cd, radius * 2) * 20;
        } else {
          controls.target.set(tx, ty, tz);
          const d = radius * 3.5; // pull back enough to clear the robot's own meshes
          camera.position.set(tx + d * 0.7, ty + d * 0.6, tz + d * 0.7);
          camera.near = Math.max(0.01, d * 0.01);
          camera.far = d * 30;
        }
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
    const followTarget = new THREE.Vector3();
    const followDelta = new THREE.Vector3();
    const UP = new THREE.Vector3(0, 1, 0); // CylinderGeometry axis
    const rootIdx = Math.max(0, bodies.findIndex((b) => b.parent < 0)); // follow this body
    const followCam = !loaded0?.metadata.viewer.camera; // fixed task camera => don't follow
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
        bodies.forEach((b, i) => {
          const g = bodyGroups[i];
          g.position.set(num(b.pos[0]), num(b.pos[1]), num(b.pos[2]));
          // recorded quaternion is (w,x,y,z); Three.js wants (x,y,z,w)
          g.quaternion.set(num(b.quat[1]), num(b.quat[2]), num(b.quat[3]), num(b.quat[0]));
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
        // orbit still works. Disabled when the task provides a fixed camera (so
        // the view matches the task's play viewpoint and doesn't drift).
        if (followCam) {
          followTarget.copy(bodyGroups[rootIdx].position);
          root.localToWorld(followTarget);
          followDelta.subVectors(followTarget, controls.target);
          camera.position.add(followDelta);
          controls.target.copy(followTarget);
        }
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
    // Now that the loop can be (re)started, route late mesh loads through it.
    wake = requestRender;

    // Wake the loop on camera interaction, playback ticks, seeks, and resizes.
    const ro = new ResizeObserver(() => {
      resize();
      ensureRunning();
    });
    ro.observe(mount);
    controls.addEventListener("start", requestRender);
    controls.addEventListener("change", requestRender);
    // Keep the default-rig visibility in sync with the store (checkbox) and
    // wake the render loop on any store change (playback ticks, seeks, toggles).
    const onStore = () => {
      defaultLights.visible = usePlaybackStore.getState().defaultLights;
      requestRender();
    };
    const unsubscribe = usePlaybackStore.subscribe(onStore);
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
