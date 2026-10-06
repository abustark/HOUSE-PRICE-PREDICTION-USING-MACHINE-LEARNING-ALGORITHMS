'use strict';
/**
 * REAL-browser visual verification (v2.0.22).
 * Drives a real Chromium with puppeteer-core against the local server.
 *
 * Run (Linux sandbox): LD_LIBRARY_PATH=/tmp/al2023-libs/lib FONTCONFIG_PATH=/tmp/fonts node tools/visual-check.js
 * Run (dev machine):   npm run visual:local   (uses an installed Chrome/Edge)
 * Override:            ABOT_CHROME=<path-to-chrome> node tools/visual-check.js
 * Policy:              LOCAL BROWSER FIRST — a locally installed Chrome/Edge is always
 *                      preferred; @sparticuz/chromium (sandbox) only if nothing is installed.
 * Requires: node server.js running on :3000.
 */
const path = require('path');
const fs = require('fs');
const puppeteer = require('puppeteer-core');

// Browser policy — LOCAL FIRST, SANDBOX LAST: an agent must look for a locally
// installed browser (Chrome or Edge; Safari cannot be driven by puppeteer) and
// actually run against it. @sparticuz/chromium — the sandbox's npm-shipped Linux
// Chromium — is only a fallback for remote machines with no browser installed.
let chromium = null;
try { chromium = require('@sparticuz/chromium').default; } catch { chromium = null; }

const LOCAL_BROWSERS = [
    process.env.ABOT_CHROME,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/microsoft-edge',
    '/usr/bin/microsoft-edge-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
];

async function launchBrowser() {
    // 1) A locally installed browser always wins (ABOT_CHROME is checked first).
    const localExe = LOCAL_BROWSERS.find((p) => p && fs.existsSync(p));
    if (localExe) {
        console.log(`[visual] using local browser: ${localExe}`);
        return puppeteer.launch({
            executablePath: localExe,
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
            headless: true,
        });
    }
    // 2) Nothing installed locally → only now fall back to the sandbox Chromium.
    if (chromium) {
        console.log('[visual] no local browser found - falling back to @sparticuz/chromium (sandbox)');
        return puppeteer.launch({
            args: [...chromium.args, '--no-sandbox', '--disable-setuid-sandbox'],
            executablePath: await chromium.executablePath(),
            headless: 'shell',
        });
    }
    throw new Error('No browser available: install Chrome/Edge, set ABOT_CHROME=<path-to-browser>, or install @sparticuz/chromium (sandbox)');
}

const ROOT = path.join(__dirname, '..');
const OUT = path.join(ROOT, 'docs', 'audits');
fs.mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, ok, detail = '') => {
    results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
};

async function runViewport(browser, { w, h, label, mobile }) {
    const page = await browser.newPage();
    await page.setViewport({ width: w, height: h, isMobile: mobile, hasTouch: mobile });
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 160)); });
    page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + String(e).slice(0, 160)));

    // Deterministic run: block all external requests. On a machine WITH internet
    // the real Firebase/GSI SDKs load from gstatic and overwrite the mocked
    // window.firebase below, so the fake login never happens (the sandbox is
    // offline, which is why this only shows up on dev machines).
    await page.setRequestInterception(true);
    page.on('request', (req) => {
        const u = req.url();
        const sameOrigin = u.startsWith('http://localhost:3000/') || u.startsWith('http://127.0.0.1:3000/') || u.startsWith('data:');
        return sameOrigin ? req.continue() : req.abort();
    });

    // Mock Firebase BEFORE any app script runs (placeholder creds can't really auth)
    await page.evaluateOnNewDocument(() => {
        const USER = { uid: 'visual-test', displayName: 'Visual Tester', email: 'visual@test.dev', isAnonymous: false, emailVerified: true, photoURL: null };
        const doc = () => ({
            set: () => Promise.resolve(), update: () => Promise.resolve(), delete: () => Promise.resolve(),
            get: () => Promise.resolve({ exists: false, empty: true, forEach: () => { } }),
            collection: () => ({ doc, add: () => Promise.resolve() }),
        });
        const coll = () => ({ doc, add: () => Promise.resolve(), get: () => Promise.resolve({ empty: true, forEach: () => { } }), where: () => ({ get: () => Promise.resolve({ empty: true, forEach: () => { } }) }), orderBy: () => ({ limit: () => ({ get: () => Promise.resolve({ empty: true, forEach: () => { } }) }) }) });
        const authObj = {
            onAuthStateChanged: (cb) => { setTimeout(() => cb(USER), 80); return () => { }; },
            get currentUser() { return USER; },
            signInAnonymously: () => Promise.resolve(USER),
            signInWithEmailAndPassword: () => Promise.resolve({ user: USER }),
            createUserWithEmailAndPassword: () => Promise.resolve({ user: USER }),
            sendPasswordResetEmail: () => Promise.resolve(),
            signOut: () => Promise.resolve(),
        };
        window.firebase = {
            initializeApp: () => ({}),
            auth: () => authObj,
            firestore: () => ({ collection: coll, doc }),
            GoogleAuthProvider: function () { },
        };
        window.marked = { parse: (s) => String(s) };
        window.hljs = { highlightElement: () => { } };
    });

    await page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await new Promise((r) => setTimeout(r, 2500)); // init + fake login + chat render

    const rect = (sel) => page.$eval(sel, (el) => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });

    // app shell visible?
    const appVisible = await page.evaluate(() => {
        const app = document.getElementById('app-container');
        return app && app.style.display !== 'none' && app.offsetParent !== null;
    });
    check(`[${label}] app shell visible after mock login`, !!appVisible);

    // B-33: mic removed
    const micCount = await page.evaluate(() => document.querySelectorAll('#voiceBtn').length);
    check(`[${label}] mic button removed (B-33)`, micCount === 0, `count=${micCount}`);

    // B-33/B-35: send & stop share the SAME slot
    const sendR = await rect('#sendBtn');
    const stopR = await rect('#stopBtn');
    const sameSlot = Math.abs(sendR.x - stopR.x) < 2 && Math.abs(sendR.y - stopR.y) < 2 && Math.abs(sendR.w - stopR.w) < 2;
    check(`[${label}] stop overlays send slot exactly (B-35)`, sameSlot, `send(${Math.round(sendR.x)},${Math.round(sendR.y)},${Math.round(sendR.w)}) stop(${Math.round(stopR.x)},${Math.round(stopR.y)},${Math.round(stopR.w)})`);

    // send flush right inside the composer box?
    const boxR = await page.$eval('#messageInput', (el) => { const box = el.closest('div[class*="rounded-2xl"], div[class*="bg-raised"]') || el.parentElement.parentElement; const r = box.getBoundingClientRect(); return { right: r.right, left: r.left }; });
    const flushRight = Math.abs((sendR.x + sendR.w) - (boxR.right - 8)) < 24;
    check(`[${label}] send button flush right (B-33)`, flushRight, `send right=${Math.round(sendR.x + sendR.w)} box right=${Math.round(boxR.right)}`);

    // B-36: bottom padding >= dock height (+headroom) on mobile
    const pad = await page.evaluate(() => {
        const c = document.getElementById('chatHistoryContainer');
        const d = document.getElementById('composerDock');
        return { pad: parseFloat(getComputedStyle(c).paddingBottom), dock: d.getBoundingClientRect().height, scrollable: c.scrollHeight >= c.clientHeight };
    });
    if (mobile) check(`[${label}] scroll padding >= dock height (B-36)`, pad.pad >= pad.dock, `pad=${Math.round(pad.pad)} dock=${Math.round(pad.dock)}`);

    // duplicate ids
    const dups = await page.evaluate(() => {
        const seen = {};
        document.querySelectorAll('[id]').forEach((el) => { seen[el.id] = (seen[el.id] || 0) + 1; });
        return Object.entries(seen).filter(([, n]) => n > 1).map(([id]) => id);
    });
    check(`[${label}] no duplicate ids`, dups.length === 0, dups.join(','));

    // screenshots
    await page.screenshot({ path: path.join(OUT, `visual-${label}-full.png`) });
    await page.screenshot({ path: path.join(OUT, `visual-${label}-composer.png`), clip: { x: 0, y: Math.max(0, h - 340), width: w, height: Math.min(340, h) } });

    // send a message → real server → 401 JSON → in-place error card should render
    await page.type('#messageInput', 'visual probe');
    await page.click('#sendBtn');
    await new Promise((r) => setTimeout(r, 1200));
    const errCard = await page.evaluate(() => {
        const card = document.querySelector('[role="alert"]');
        return card ? card.textContent.slice(0, 120) : null;
    });
    check(`[${label}] error surfaces as visible card (B-20/B-34)`, !!errCard, errCard || 'none');

    // stop button becomes visible during stream? (fast 401 — check class swap happened)
    const stopCleared = await page.evaluate(() => {
        const sb = document.getElementById('stopBtn');
        return sb.classList.contains('stop-hidden');
    });
    check(`[${label}] stop button re-hidden after stream ends`, stopCleared);

    // final screenshot with the error card
    await page.screenshot({ path: path.join(OUT, `visual-${label}-error.png`), fullPage: false });

    // ---- v2.0.26: the owner's two streaming scenarios (real streamed fetch) ----
    // Mock /api/chat with a genuine SSE stream so the app's real streaming
    // path (placeholders, chunk innerHTML updates, finalize) is exercised.
    await page.evaluate(() => {
        window.fetch = (url, _opts) => {
            if (!String(url).includes('/api/chat')) {
                return Promise.reject(new TypeError('only /api/chat mocked'));
            }
            const { ReadableStream } = window;
            const enc = new TextEncoder();
            const longText = Array.from({ length: 40 }, (_, i) =>
                `Paragraph ${i + 1}: lorem ipsum dolor sit amet, streaming layout probe line.`).join('\n\n');
            const body = { model: 'probe/model', choices: [{ delta: { content: longText } }] };
            const rs = new ReadableStream({
                start(c) {
                    // drip the response in chunks over ~1.2s like a real stream
                    const parts = 8;
                    for (let i = 0; i < parts; i++) {
                        setTimeout(() => c.enqueue(enc.encode(`data: ${JSON.stringify(body)}\n\n`)), i * 150);
                    }
                    setTimeout(() => c.enqueue(enc.encode('data: [DONE]\n\n')), parts * 150 + 60);
                    setTimeout(() => c.close(), parts * 150 + 100);
                },
            });
            return Promise.resolve({ ok: true, body: rs });
        };
        // long-answer ask
        const input = document.getElementById('messageInput');
        input.value = 'long reply probe';
        document.getElementById('sendBtn').click();
    });
    await new Promise((r) => setTimeout(r, 600)); // mid-stream
    {
        const mid = await page.evaluate(() => {
            const el = document.getElementById('chatHistoryContainer');
            return el.scrollHeight - el.scrollTop - el.clientHeight;
        });
        check(`[${label}] mid-stream: container pinned to bottom (follow live)`, mid < 140, `distance=${Math.round(mid)}px`);
    }
    await new Promise((r) => setTimeout(r, 1800)); // stream done + settle
    {
        const st = await page.evaluate(() => {
            const cont = document.getElementById('chatHistoryContainer');
            const dock = document.getElementById('composerDock').getBoundingClientRect();
            const msgs = document.querySelectorAll('#chatHistory .message-container, #chatHistory > div');
            const last = msgs[msgs.length - 1];
            const lastRect = last ? last.getBoundingClientRect() : null;
            return {
                dist: cont.scrollHeight - cont.scrollTop - cont.clientHeight,
                lastBottom: lastRect ? Math.round(lastRect.bottom) : null,
                dockTop: Math.round(dock.top),
            };
        });
        check(`[${label}] LONG reply: pinned to bottom after stream`, st.dist < 140, `distance=${Math.round(st.dist)}px`);
        check(`[${label}] LONG reply: last content above composer top`, st.lastBottom === null || st.lastBottom <= st.dockTop + 24,
            `lastBottom=${st.lastBottom} dockTop=${st.dockTop}`);
        await page.screenshot({ path: path.join(OUT, `visual-${label}-long.png`), fullPage: false });
    }
    // ---- v2.0.27 (B-43): the CSS half of the scroll fix must actually be LIVE ----
    // v2.0.26 changed src/input.css but shipped a stale dist/output.css, so
    // `overflow-anchor:none` + the keyboard-aware dock never reached clients.
    {
        const cssProbe = await page.evaluate(() => {
            const cont = document.getElementById('chatHistoryContainer');
            const dockStyle = getComputedStyle(document.getElementById('composerDock'));
            return {
                anchor: getComputedStyle(cont).overflowAnchor,
                dockPos: dockStyle.position,
                dockBottom: dockStyle.bottom,
            };
        });
        check(`[${label}] B-43: overflow-anchor:none live (CSS not stale)`,
            cssProbe.anchor === 'none', `overflow-anchor=${cssProbe.anchor}`);
        const kbAware = cssProbe.dockPos !== 'fixed' || cssProbe.dockBottom !== 'auto';
        check(`[${label}] B-42: dock is keyboard-aware (bottom:var(--kb-h))`,
            kbAware, `position=${cssProbe.dockPos} bottom=${cssProbe.dockBottom}`);
    }

    // SHORT reply: response smaller than the viewport
    await page.evaluate(() => {
        const input = document.getElementById('messageInput');
        input.value = 'short reply probe';
        document.getElementById('sendBtn').click();
    });
    await new Promise((r) => setTimeout(r, 1600));
    {
        const st = await page.evaluate(() => {
            const dock = document.getElementById('composerDock').getBoundingClientRect();
            const msgs = document.querySelectorAll('#chatHistory .message-container, #chatHistory > div');
            const last = msgs[msgs.length - 1];
            const lastRect = last ? last.getBoundingClientRect() : null;
            return { lastBottom: lastRect ? Math.round(lastRect.bottom) : null, dockTop: Math.round(dock.top) };
        });
        check(`[${label}] SHORT reply: fully visible above composer`,
            st.lastBottom === null || st.lastBottom <= st.dockTop + 24,
            `lastBottom=${st.lastBottom} dockTop=${st.dockTop}`);
        await page.screenshot({ path: path.join(OUT, `visual-${label}-short.png`), fullPage: false });
    }

    if (consoleErrors.length) results.push(`--- console errors (${label}) ---\n` + consoleErrors.slice(0, 5).join('\n'));
    await page.close();
    return;
}

(async () => {
    const browser = await launchBrowser();

    try {
        await runViewport(browser, { w: 375, h: 812, label: 'mobile', mobile: true });
        await runViewport(browser, { w: 1280, h: 800, label: 'desktop', mobile: false });
    } finally {
        await browser.close();
    }

    const fails = results.filter((r) => r.startsWith('FAIL')).length;
    fs.writeFileSync(path.join(OUT, 'visual-check-results.txt'), results.join('\n'));
    console.log(`\n${fails} FAILURE(S) — screenshots in docs/audits/`);
    process.exit(fails ? 1 : 0);
})().catch((e) => { console.error('VISUAL CHECK CRASHED:', e); process.exit(1); });
