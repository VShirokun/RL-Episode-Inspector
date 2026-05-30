import { defineConfig, devices } from "@playwright/test";

// Boots the real backend (auto-generating a small fake dataset if needed) and
// the Vite dev server, then runs the E2E specs in ../e2e against them.
const VENV = "../.venv/bin/rl-episode-inspector";
const EP_DIR = "../sample_data/cartpole/episodes";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: { timeout: 7_000 },
  fullyParallel: false,
  retries: 0,
  reporter: process.env.CI ? "list" : "line",
  use: { baseURL: "http://localhost:3000", trace: "on-first-retry" },
  // Run at deviceScaleFactor 2 (HiDPI): regression guard for the WebGL canvas
  // overflowing its pane on retina-class displays (see Viewer3D setSize).
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1400, height: 850 }, deviceScaleFactor: 2 },
    },
  ],
  webServer: [
    {
      command: `sh -c '${VENV} generate-fake-cartpole -o ${EP_DIR} -n 8 --seed 1 >/dev/null 2>&1 || true; ${VENV} serve --episodes-dir ${EP_DIR} --port 8000'`,
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      command: "npm run dev",
      url: "http://localhost:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
});
