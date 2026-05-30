import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server runs on :3000 and proxies API calls to the Python backend on
// :8000, so the frontend can use same-origin "/api" paths and we avoid CORS in
// development. Override the backend with the RLEI_BACKEND env var.
const backend = process.env.RLEI_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": { target: backend, changeOrigin: true },
      "/health": { target: backend, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    // Only vitest unit tests under src/. The Playwright specs in e2e/ also end
    // in .spec.ts but must NOT be collected by vitest (they need @playwright/test).
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
