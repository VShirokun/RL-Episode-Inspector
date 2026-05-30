// 3D viewer for the Franka Reach task (viewer.type === "reach3d").
//
// Renders a deterministic visual reconstruction from recorded state: the
// end-effector as a sphere and the current target as a wireframe sphere (radius
// = reach threshold) on a ground plane, with a connector line. Like the Cartpole
// viewer it reads the columns named by metadata.viewer.state_mapping each frame.
//
// Frame convention: recorded positions are in the Franka base frame (x forward,
// y left, z up). Three.js is y-up, so we map base (x, y, z) -> three (x, z, y).

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clampFrame } from "../playback/frameSync";
import { usePlaybackStore } from "../playback/playbackStore";

const REACH_THRESHOLD = 0.07; // meters; matches examples/reach/targets.py

function roleColumns() {
  const m = usePlaybackStore.getState().loaded?.metadata.viewer.state_mapping ?? {};
  return {
    ex: m["ee_x"] ?? "ee_x",
    ey: m["ee_y"] ?? "ee_y",
    ez: m["ee_z"] ?? "ee_z",
    tx: m["target_x"] ?? "target_x",
    ty: m["target_y"] ?? "target_y",
    tz: m["target_z"] ?? "target_z",
  };
}

// base (x fwd, y left, z up) -> three (x, y=up, z)
function toThree(x: number, y: number, z: number): [number, number, number] {
  return [x, z, y];
}

export function ReachViewer() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x10131a);

    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    camera.position.set(1.2, 0.9, 1.2);

    // antialias off keeps software WebGL (llvmpipe/SwiftShader) cheap.
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0.5, 0.3, 0); // workspace center (three coords)

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(2, 4, 3);
    scene.add(dir);

    const grid = new THREE.GridHelper(2, 20, 0x335, 0x223);
    scene.add(grid);

    // robot base pedestal at the origin (base frame origin)
    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 0.08, 0.1, 24),
      new THREE.MeshStandardMaterial({ color: 0x445566 }),
    );
    base.position.set(0, 0.05, 0);
    scene.add(base);

    // end-effector (solid sphere)
    const ee = new THREE.Mesh(
      new THREE.SphereGeometry(0.03, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0x4f9dff, metalness: 0.2, roughness: 0.5 }),
    );
    scene.add(ee);

    // current target (wireframe sphere at reach threshold radius) + center dot
    const target = new THREE.Mesh(
      new THREE.SphereGeometry(REACH_THRESHOLD, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x4ecb71, wireframe: true }),
    );
    scene.add(target);
    const targetDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.012, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x4ecb71 }),
    );
    scene.add(targetDot);

    // connector line EE -> target
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]);
    const line = new THREE.Line(
      lineGeom,
      new THREE.LineBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.6 }),
    );
    scene.add(line);

    const resize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      needsRender = true;
    };
    let needsRender = true;
    let lastFrame = -1;
    let running = false;
    resize();

    const cols = roleColumns();
    let raf = 0;
    const loop = () => {
      // On-demand rendering that stops when idle (see ArticulationViewer).
      const { loaded, currentFrame, isPlaying } = usePlaybackStore.getState();
      const moved = controls.update();
      const idx = loaded ? clampFrame(currentFrame, loaded.metadata.num_frames) : -1;
      const dirty = isPlaying || moved || needsRender || idx !== lastFrame;
      if (loaded && dirty) {
        lastFrame = idx;
        needsRender = false;
        const get = (c: string) => (loaded.columns[c]?.[idx] as number) ?? 0;
        const [ex, ey, ez] = toThree(get(cols.ex), get(cols.ey), get(cols.ez));
        const [tx, ty, tz] = toThree(get(cols.tx), get(cols.ty), get(cols.tz));
        ee.position.set(ex, ey, ez);
        target.position.set(tx, ty, tz);
        targetDot.position.set(tx, ty, tz);
        // turn the target green->cyan when the EE is inside the reach zone
        const inside =
          ee.position.distanceTo(target.position) <= REACH_THRESHOLD * 1.5;
        (target.material as THREE.MeshBasicMaterial).color.set(inside ? 0x3fd0c9 : 0x4ecb71);
        const pos = line.geometry.attributes.position as THREE.BufferAttribute;
        pos.setXYZ(0, ex, ey, ez);
        pos.setXYZ(1, tx, ty, tz);
        pos.needsUpdate = true;
        renderer.render(scene, camera);
      }
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
  }, []);

  return <div className="viewer3d" ref={mountRef} data-testid="viewer3d" />;
}
