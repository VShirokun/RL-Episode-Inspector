// Cartpole 3D viewer (Three.js). Renders simple geometry — a cart box and a
// pole — and drives their transforms from recorded state each animation frame.
// This is a deterministic *visual reconstruction* (not a physics replay): the
// cart's x and the pole's lean come straight from the frame columns named by
// metadata.viewer.state_mapping.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clampFrame } from "../playback/frameSync";
import { usePlaybackStore } from "../playback/playbackStore";

const POLE_LENGTH = 1.0;
const CART_Y = 0.25;

function readMapping(): { cartCol: string; angleCol: string } {
  const loaded = usePlaybackStore.getState().loaded;
  const m = loaded?.metadata.viewer.state_mapping ?? {};
  return {
    cartCol: m["cart_position"] ?? "cart_position",
    angleCol: m["pole_angle"] ?? "pole_angle",
  };
}

export function Viewer3D() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x10131a);

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(2.5, 2.0, 4.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0.6, 0);

    // lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(3, 6, 4);
    scene.add(dir);

    // ground + grid + track
    const grid = new THREE.GridHelper(12, 24, 0x335, 0x223);
    scene.add(grid);
    const track = new THREE.Mesh(
      new THREE.BoxGeometry(8, 0.02, 0.2),
      new THREE.MeshStandardMaterial({ color: 0x2a3550 }),
    );
    track.position.y = 0.01;
    scene.add(track);

    // cart group (moves along x); pole pivots at the top of the cart
    const cart = new THREE.Group();
    const cartBody = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.3, 0.4),
      new THREE.MeshStandardMaterial({ color: 0x4f9dff, metalness: 0.2, roughness: 0.6 }),
    );
    cartBody.position.y = CART_Y;
    cart.add(cartBody);

    const polePivot = new THREE.Group();
    polePivot.position.y = CART_Y + 0.15;
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.04, 0.04, POLE_LENGTH, 16),
      new THREE.MeshStandardMaterial({ color: 0xff6b6b, metalness: 0.1, roughness: 0.5 }),
    );
    pole.position.y = POLE_LENGTH / 2; // base at pivot
    polePivot.add(pole);
    const hub = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 16, 16),
      new THREE.MeshStandardMaterial({ color: 0xffd166 }),
    );
    polePivot.add(hub);
    cart.add(polePivot);
    scene.add(cart);

    const resize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      // updateStyle=true (default) so the canvas CSS size tracks the pane size
      // independent of devicePixelRatio; passing false makes the canvas overflow
      // on HiDPI displays and cover neighbouring panels.
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    let raf = 0;
    const render = () => {
      const { loaded, currentFrame } = usePlaybackStore.getState();
      if (loaded) {
        const { cartCol, angleCol } = readMapping();
        const n = loaded.metadata.num_frames;
        const idx = clampFrame(currentFrame, n);
        const x = (loaded.columns[cartCol]?.[idx] as number) ?? 0;
        const angle = (loaded.columns[angleCol]?.[idx] as number) ?? 0;
        cart.position.x = x;
        polePivot.rotation.z = -angle; // angle from vertical, tilt in x-y plane
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
