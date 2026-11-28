import { expect, test } from "@playwright/test";

// One path through the whole running system: the dashboard starts a session, which goes
// API -> replayer -> ingest -> worker -> API -> WebSocket -> this page. Uses the synthetic
// signal so it needs no dataset (CI has none).
test("start a session from the dashboard and see it live", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("connection")).toHaveText("Live", { timeout: 20_000 });

  await page.getByRole("button", { name: "Synthetic" }).click();
  await page.getByLabel("Speed").selectOption("2");
  await page.getByLabel("Duration").selectOption("1");
  await page.getByRole("button", { name: "Start session" }).click();

  // The new session is watched automatically and listed as running.
  const running = page.locator(".session.active");
  await expect(running).toContainText("synth-", { timeout: 10_000 });

  // A prediction arrives for it, with its model version.
  await expect(page.getByTestId("predicted-class")).toHaveText(/rest|left|right/, {
    timeout: 45_000,
  });
  await expect(page.getByTestId("prediction-model")).toContainText(/\d+\.\d+\.\d+/);

  // At least one worker is up, and the system tiles are populated.
  await expect(page.locator('[data-testid="workers"] tr[data-status="healthy"]').first()).toBeVisible();
  await expect(page.getByTestId("tile-Workers").locator(".value")).not.toHaveText("0");

  // The EEG canvas is drawn at a real size.
  const box = await page.getByTestId("trace-canvas").boundingBox();
  expect(box && box.width > 300 && box.height > 100).toBeTruthy();

  // Clean up: stop it from the UI.
  await running.getByRole("button", { name: "Stop" }).click();
  await expect(running).toContainText("stopped", { timeout: 15_000 });
});
