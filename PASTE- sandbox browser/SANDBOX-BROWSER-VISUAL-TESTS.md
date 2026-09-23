# Sandbox Real-Browser Setup — Chromium Visual Tests (No CDN Required)

> **What this is:** step-by-step instructions + the exact code for running a **real Chromium browser inside the Arena sandbox** — the thing that looked impossible because the Playwright CDN is blocked.
>
> **The code that uses this:** `tools/visual-check.js` (run via `npm run visual`)
> **Proven working:** 2026-09-21 · commit `3293941` (v2.0.22) · first run 15/15 PASS
>
> **Why not Playwright:** `npx playwright install chromium` downloads the browser from `cdn.playwright.dev` — the sandbox's network proxy kills that TLS connection every time, and there is no system browser and no root for `apt install`. The trick below gets Chromium **from the npm registry instead** (which *is* reachable — we install packages all session).

---

## The idea in one line

The npm package **`@sparticuz/chromium`** ships a complete Chromium binary **inside the package itself** (built for AWS Lambda, where there's also no CDN/root) — plus tarballs of the system libraries and fonts Chromium needs. `puppeteer-core` drives it. Nothing to download from blocked hosts.

## What's inside the package

```
node_modules/@sparticuz/chromium/bin/
├── chromium.br          ← the Chromium binary (Brotli-compressed)
├── al2023.tar.br        ← system shared libs (libnss3, libnspr4, libexpat…) ★ the missing piece
├── fonts.tar.br         ← fonts (Chromium refuses to render text without them)
└── swiftshader.tar.br   ← software GL
```

`@sparticuz/chromium` auto-extracts `chromium.br` + `swiftshader` to `/tmp` when you call `executablePath()` — but **does NOT auto-extract `al2023.tar.br` (the libs) or `fonts.tar.br` in a plain Linux container**, which is why the first launch failed with `libnspr4.so: cannot open shared object file`. Extracting those two manually is the whole secret.

---

## Setup steps (run once per sandbox session)

> `/tmp` and un-saved `node_modules` are wiped between sessions — redo this whenever the sandbox resets. Takes ~30 seconds.

### 1. Install from npm (reachable)

```bash
npm install --no-save --no-audit --no-fund @sparticuz/chromium puppeteer-core
```

(`--no-save` keeps them out of `package.json` — they're sandbox tooling, not app deps. Note: a later plain `npm install <anything>` may prune them; if `require` fails, reinstall.)

### 2. Extract fonts (plain brotli→tar works)

```bash
node -e "
const { brotliDecompressSync } = require('zlib');
const { execSync } = require('child_process');
const fs = require('fs');
const pkg = 'node_modules/@sparticuz/chromium/bin';
const out = '/tmp/fonts.tar';
fs.copyFileSync(pkg + '/fonts.tar.br', out);
fs.writeFileSync(out, brotliDecompressSync(fs.readFileSync(out)));
execSync('tar -xf /tmp/fonts.tar -C /tmp');
console.log('fonts extracted to /tmp/fonts');
"
```

### 3. Extract the system libs — ⚠️ use Python's tarfile, not GNU tar

The decompressed `al2023.tar` confuses GNU tar ("This does not look like a tar archive") even though it IS a tar (`lib/libexpat.so.1`…). Python's `tarfile` handles it fine:

```bash
node -e "
const { brotliDecompressSync } = require('zlib');
const fs = require('fs');
fs.writeFileSync('/tmp/al2023.tar',
  brotliDecompressSync(fs.readFileSync('node_modules/@sparticuz/chromium/bin/al2023.tar.br')));
"
python3 -c "
import tarfile
t = tarfile.open('/tmp/al2023.tar')
t.extractall('/tmp/al2023-libs')   # creates /tmp/al2023-libs/lib/*.so
t.close()
print('libs extracted')
"
ls /tmp/al2023-libs/lib/    # should list libnss3.so, libnspr4.so, libexpat.so.1 …
```

### 4. Launch (the launcher pattern used everywhere after this)

```javascript
const chromium = require('@sparticuz/chromium').default;   // note: .default (ESM interop)
const puppeteer = require('puppeteer-core');

const browser = await puppeteer.launch({
  args: [...chromium.args, '--no-sandbox', '--disable-setuid-sandbox'],
  executablePath: await chromium.executablePath(),  // → /tmp/chromium
  headless: 'shell',
});
const page = await browser.newPage();
await page.setViewport({ width: 375, height: 812, isMobile: true, hasTouch: true });
// …drive the page…
await browser.close();
```

**Run it with the library path + font config:**

```bash
LD_LIBRARY_PATH=/tmp/al2023-libs/lib FONTCONFIG_PATH=/tmp/fonts node your-script.js
```

---

## Running ABot's visual checks

```bash
# terminal 1 — the app (placeholder .env is fine; visuals mock Firebase client-side)
node server.js

# terminal 2 — the real-browser suite (mobile 375×812 + desktop 1280×800)
npm run visual         # Linux sandbox: @sparticuz/chromium + its extracted libs
npm run visual:local   # dev machine (Windows/macOS): uses installed Chrome/Edge
# or point it at a specific binary:
ABOT_CHROME="C:\Program Files\Google\Chrome\Application\chrome.exe" node tools/visual-check.js
```

**Browser policy — LOCAL FIRST:** the tool resolves a browser in this order: `ABOT_CHROME` → a
**locally installed Chrome/Edge** (Windows/macOS/Linux paths) → `@sparticuz/chromium`
**only as a fallback when no local browser exists** (Safari can't be automated — use installed
Chrome/Edge on macOS). It also **blocks every non-localhost request**:
on a machine with internet the real Firebase SDK loads from gstatic and overwrites the injected mock,
so the fake login would never happen (the sandbox is offline, which is why this only bites on dev machines).

What it does (see `tools/visual-check.js`):
- Injects a Firebase/marked/hljs mock via `evaluateOnNewDocument` so the app logs in without real creds
- Pixel-measures the composer: mic absence, **stop button overlays send button exactly** (same x/y/size), send flush right, scroll padding ≥ dock height, duplicate ids
- Sends a real message to the real local server and asserts the error/success card renders visibly
- Saves screenshots to `docs/audits/visual-{mobile,desktop}-{full,composer,error}.png`
- Exits non-zero on any failure → safe to add to CI later

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libnspr4.so: cannot open shared object file` | al2023 libs not extracted / `LD_LIBRARY_PATH` unset | Redo step 3; run with `LD_LIBRARY_PATH=/tmp/al2023-libs/lib` |
| `chromium.executablePath is not a function` | Missing `.default` on the require | `require('@sparticuz/chromium').default` |
| `Cannot find module '@sparticuz/chromium'` | A later `npm install` pruned the `--no-save` packages | Re-run step 1 |
| Garbled/missing text in screenshots | Fonts not extracted / `FONTCONFIG_PATH` unset | Redo step 2; run with `FONTCONFIG_PATH=/tmp/fonts` |
| `tar: This does not look like a tar archive` | GNU tar rejects the decompressed al2023 archive | Use `python3 -c "import tarfile; tarfile.open(...).extractall(...)"` |
| Page shows old build | Not this tool — that's the service worker; hit the site with `#reset` first |

---

## History / credits

- Idea + insistence: **the owner** ("can't you write a script to run a browser inbuilt for you?") after the sandbox's Playwright CDN block was twice declared unfixable. They were right that a path existed.
- First working launch + ABot integration (`tools/visual-check.js`, 15/15 PASS at mobile+desktop): v2.0.22, commit `3293941`.
- Evidence screenshots: `docs/audits/visual-mobile-error.png` (the money shot — single send button, error card rendering), `visual-desktop-full.png` (full shell), and 4 more.
