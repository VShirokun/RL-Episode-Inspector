// Generic articulation viewer (viewer.type === "articulation3d").
//
// Renders an entire robot from recorded per-body poses: one oriented box per
// rigid body (showing position + orientation) connected by "bones" to each
// body's parent, plus any point markers (e.g. a reach target). Driven entirely
// by metadata.viewer.bodies/markers and the pose_<body>_* columns — nothing here
// is robot-specific, so it works for Franka now and a humanoid later.
//
// Recorded poses are in the sim frame (z-up, Isaac). We parent everything to a
// group rotated -90° about X so z-up data displays correctly in Three.js (y-up),
// without per-body axis math.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clampFrame } from "../playback/frameSync";
import { usePlaybackStore } from "../playback/playbackStore";
import type { BodySpec, MarkerSpec } from "../types/signal";

export function ArticulationViewer() {
  const mountRef = useRef<HTMLDivElement>(null);

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

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0.4, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(2, 4, 3);
    scene.add(dir);
    scene.add(new THREE.GridHelper(2, 20, 0x335, 0x223));

    // Group that maps the recorded up-axis to Three.js's y-up.
    const root = new THREE.Group();
    if (upAxis === "z") root.rotation.x = -Math.PI / 2;
    scene.add(root);

    // one oriented box per body
    const bodyMeshes = bodies.map((_, i) => {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(0.05, 0.05, 0.05),
        new THREE.MeshStandardMaterial({
          color: i === 0 ? 0x8899aa : 0x4f9dff,
          metalness: 0.2,
          roughness: 0.6,
        }),
      );
      root.add(m);
      return m;
    });

    // one bone (line) per non-root body, connecting it to its parent
    const boneMat = new THREE.LineBasicMaterial({ color: 0x9fb3d1 });
    const bones = bodies.map((b) => {
      if (b.parent < 0) return null;
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const line = new THREE.Line(g, boneMat);
      root.add(line);
      return line;
    });

    // markers (e.g. reach target)
    const markerMeshes = markers.map((mk) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.03, 16, 16),
        new THREE.MeshBasicMaterial({ color: mk.color ?? "#4ecb71" }),
      );
      root.add(m);
      return m;
    });

    const resize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    const tmp = new THREE.Vector3();
    let raf = 0;
    const render = () => {
      const { loaded, currentFrame } = usePlaybackStore.getState();
      if (loaded) {
        const idx = clampFrame(currentFrame, loaded.metadata.num_frames);
        const num = (c: string) => (loaded.columns[c]?.[idx] as number) ?? 0;
        // position bodies
        bodies.forEach((b, i) => {
          const mesh = bodyMeshes[i];
          mesh.position.set(num(b.pos[0]), num(b.pos[1]), num(b.pos[2]));
          // recorded quaternion is (w,x,y,z); Three.js wants (x,y,z,w)
          mesh.quaternion.set(num(b.quat[1]), num(b.quat[2]), num(b.quat[3]), num(b.quat[0]));
        });
        // update bones to parent positions
        bodies.forEach((b, i) => {
          const line = bones[i];
          if (!line || b.parent < 0) return;
          const pos = line.geometry.attributes.position as THREE.BufferAttribute;
          const c = bodyMeshes[i].position;
          const p = bodyMeshes[b.parent].position;
          pos.setXYZ(0, c.x, c.y, c.z);
          pos.setXYZ(1, p.x, p.y, p.z);
          pos.needsUpdate = true;
        });
        // markers
        markers.forEach((mk, i) => {
          tmp.set(num(mk.pos[0]), num(mk.pos[1]), num(mk.pos[2]));
          markerMeshes[i].position.copy(tmp);
        });
      }
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(render);
    };
    render();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  return <div className="viewer3d" ref={mountRef} data-testid="viewer3d" />;
}
