import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  // Episode list loads and an episode auto-selects (viewer canvas appears).
  await expect(page.getByTestId("episode-list")).toBeVisible();
  await expect(page.locator(".episode-row").first()).toBeVisible();
  await expect(page.getByTestId("viewer3d")).toBeVisible();
});

test("page opens and lists episodes", async ({ page }) => {
  const rows = page.locator(".episode-row");
  expect(await rows.count()).toBeGreaterThan(0);
});

test("viewer canvas does not overflow into the charts pane (HiDPI guard)", async ({ page }) => {
  const layout = await page.evaluate(() => {
    const canvas = document.querySelector(".viewer3d canvas") as HTMLCanvasElement;
    const charts = document.querySelector(".charts-pane") as HTMLElement;
    return {
      canvasRight: canvas.getBoundingClientRect().right,
      chartsLeft: charts.getBoundingClientRect().left,
    };
  });
  // The canvas must not extend past where the charts pane begins.
  expect(layout.canvasRight).toBeLessThanOrEqual(layout.chartsLeft + 1);
});

test("best episode can be selected", async ({ page }) => {
  await page.getByTestId("btn-best").click();
  await expect(page.locator(".episode-row.selected")).toBeVisible();
  await expect(page.getByTestId("viewer3d")).toBeVisible();
});

test("play and pause toggle", async ({ page }) => {
  const playBtn = page.getByTestId("play-pause");
  const counter = page.getByTestId("frame-counter");
  const before = await counter.textContent();
  await playBtn.click();
  await page.waitForTimeout(400);
  await playBtn.click(); // pause
  const after = await counter.textContent();
  expect(after).not.toBe(before);
});

test("Space toggles play/pause", async ({ page }) => {
  const counter = page.getByTestId("frame-counter");
  const before = await counter.textContent();
  await page.keyboard.press("Space");
  await page.waitForTimeout(400);
  await page.keyboard.press("Space");
  expect(await counter.textContent()).not.toBe(before);
});

test("ArrowRight steps a frame", async ({ page }) => {
  await page.getByTestId("frame-counter").click(); // ensure focus on body
  const counter = page.getByTestId("frame-counter");
  const start = await counter.textContent();
  await page.keyboard.press("ArrowRight");
  await expect(counter).not.toHaveText(start ?? "");
});

test("clicking a chart seeks", async ({ page }) => {
  const counter = page.getByTestId("frame-counter");
  const chart = page.getByTestId("timeseries-chart").first();
  const box = await chart.boundingBox();
  expect(box).not.toBeNull();
  if (!box) return;
  await page.mouse.click(box.x + box.width * 0.75, box.y + box.height / 2);
  // frame should now be in the latter portion of the episode
  const txt = (await counter.textContent()) ?? "0 / 0";
  const [cur, total] = txt.split("/").map((s) => Number(s.trim()));
  expect(cur).toBeGreaterThan(total * 0.4);
});

test("dragging a chart scrubs", async ({ page }) => {
  const counter = page.getByTestId("frame-counter");
  const chart = page.getByTestId("timeseries-chart").first();
  const box = await chart.boundingBox();
  if (!box) return;
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * 0.2, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.9, y, { steps: 8 });
  await page.mouse.up();
  const txt = (await counter.textContent()) ?? "0 / 0";
  const [cur, total] = txt.split("/").map((s) => Number(s.trim()));
  expect(cur).toBeGreaterThan(total * 0.6);
});

test("speed change works", async ({ page }) => {
  await page.getByTestId("speed-select").selectOption("2");
  await expect(page.getByTestId("speed-select")).toHaveValue("2");
});

test("backend error surfaces (bad episode)", async ({ page }) => {
  // Force a metadata fetch failure and ensure the UI shows an error, not a crash.
  await page.route("**/api/episodes/*/metadata", (route) =>
    route.fulfill({ status: 500, body: JSON.stringify({ detail: "boom" }) }),
  );
  await page.locator(".episode-row").first().click();
  await expect(page.getByTestId("error-state")).toBeVisible();
});
