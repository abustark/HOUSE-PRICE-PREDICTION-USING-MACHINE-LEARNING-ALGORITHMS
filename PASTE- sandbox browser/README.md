# PASTE- sandbox browser — self-contained kit for online-only AI agents

> **Purpose:** hand this folder (whole) to any online-only AI agent and it gains the ability to run a
> **real headless Chromium** inside a sandbox where the Playwright CDN is blocked, there is no system
> browser, and there is no root for `apt install`. The trick: Chromium comes from the **npm registry**
> (reachable), not a CDN.
>
> **Source of truth:** ABot repo — `docs/SANDBOX-BROWSER-VISUAL-TESTS.md` (guide) +
> `tools/visual-check.js` (working reference implementation). Copied here verbatim at ABot commit
> `f1e79bf` (v2.0.28). If the repo copy changes, re-copy it here.

## Files in this folder

| File | What it is |
|---|---|
| `SANDBOX-BROWSER-VISUAL-TESTS.md` | **Start here.** Full step-by-step setup guide: install, extract fonts + system libs, launch pattern, troubleshooting table |
| `visual-check.js` | A complete, working reference implementation (ABot's real-browser suite): cross-platform browser resolution, request interception for deterministic offline runs, Firebase/app mocking via `evaluateOnNewDocument`, pixel-measure assertions, screenshots, exit-code semantics |
| `package-scripts-snippet.json` | The `npm` scripts to merge into your `package.json` (`visual` = Linux sandbox, `visual:local` = dev machine with installed Chrome/Edge) |

## Quick start for an agent (3 steps)

1. **Read `SANDBOX-BROWSER-VISUAL-TESTS.md`** and run its Setup steps 1–4 (≈30 s per session;
   `/tmp` and unsaved `node_modules` are wiped between sessions — redo after every sandbox reset).
2. **Adapt `visual-check.js`**: keep the `launchBrowser()` + request-interception + mock-injection
   skeleton, replace the ABot-specific selectors/assertions (`#app-container`, `#composerDock`,
   `#chatHistoryContainer`, …) with your own app's.
3. **Run it**: start your app locally (`node server.js` on `:3000` or your equivalent), then
   `npm run visual` (Linux sandbox) or `npm run visual:local` (Windows/macOS dev machine).

## Notes that save hours (learned in production)

- The sandbox is **offline by default**, which is *why* the mock-based tests work. On a machine
  **with** internet, real third-party scripts (Firebase SDK from gstatic, Google Identity, CDNs) load
  and overwrite your injected mocks — `visual-check.js` therefore **blocks every non-localhost
  request** via `page.setRequestInterception`. Keep that.
- Browser resolution order in the reference implementation:
  `ABOT_CHROME` env override → locally installed Chrome/Edge (Windows paths + macOS) →
  `@sparticuz/chromium` (Linux sandbox binary).
- Screenshots are your evidence: save them on every run and assert geometry with
  `getBoundingClientRect()`, not vibes.
