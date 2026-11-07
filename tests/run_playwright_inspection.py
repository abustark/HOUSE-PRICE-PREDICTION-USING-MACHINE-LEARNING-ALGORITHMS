"""
Playwright automated inspection script for PC and Mobile UI verification.
Runs FastAPI server on a dynamically allocated free port with HTTP healthcheck.
Performs end-to-end verification without saving persistent screenshots to the repository.
"""

import os
import sys
import time
import socket
import urllib.request
import threading
import asyncio
import uvicorn
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.server import app

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

PORT = get_free_port()
BASE_URL = f"http://127.0.0.1:{PORT}"

async def run_playwright():
    async with async_playwright() as p:
        # Try system browser channels (msedge or chrome)
        try:
            browser = await p.chromium.launch(headless=True, channel="msedge")
            print("Browser launched via msedge channel.")
        except Exception:
            browser = await p.chromium.launch(headless=True, channel="chrome")
            print("Browser launched via chrome channel.")

        # --- 1. DESKTOP TEST (1440x900) ---
        print("\nRunning Desktop (1440x900) test assertions...")
        desktop_context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await desktop_context.new_page()
        page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

        response = await page.goto(BASE_URL, wait_until="domcontentloaded")
        print(f"Desktop page status: {response.status}")
        await page.wait_for_selector("#area_type", state="attached", timeout=10000)
        await asyncio.sleep(0.5)

        # Desktop 1: Check form elements exist
        assert await page.is_visible("#area_type")
        assert await page.is_visible("#location")
        assert await page.is_visible("#predict-btn")
        print("[PASS] Desktop Predictor form rendered.")

        # Desktop 2: Click Predict button -> Live Prediction Result
        await page.click("#predict-btn")
        await page.wait_for_selector("#result-body:not([hidden])", timeout=10000)
        await asyncio.sleep(0.5)
        
        price_text = await page.text_content("#price-num")
        unit_text = await page.text_content("#price-unit")
        print(f"[PASS] Desktop Prediction result: INR {price_text} {unit_text}")

        # Desktop 3: Click Instant API Console Tab -> Live Explorer
        await page.click("#tab-api")
        await page.wait_for_selector("#view-api.active", timeout=5000)
        # Click "Send Request" to test live endpoint in console
        await page.click("#btn-send-api")
        await page.wait_for_selector("#api-status-pill:not([hidden])", timeout=10000)
        status_text = await page.text_content("#api-status-pill")
        print(f"[PASS] Desktop API Console response: {status_text}")
        await asyncio.sleep(0.5)

        # Desktop 4: Click Model Performance Tab
        await page.click("#tab-metrics")
        await page.wait_for_selector("#view-metrics.active", timeout=5000)
        await asyncio.sleep(0.5)
        print("[PASS] Desktop Model Performance tab rendered.")

        # Check footer
        assert await page.is_visible(".site-footer")
        footer_text = await page.text_content(".site-footer")
        assert "Basith AbuSyed" in footer_text
        print("[PASS] Desktop footer verified with Basith AbuSyed copyright.")

        await desktop_context.close()

        # --- 2. MOBILE TEST (390x844 - iPhone / Pixel) ---
        print("\nRunning Mobile (390x844) test assertions...")
        mobile_context = await browser.new_context(
            viewport={"width": 390, "height": 844},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            is_mobile=True,
            has_touch=True,
        )
        mpage = await mobile_context.new_page()

        m_response = await mpage.goto(BASE_URL, wait_until="domcontentloaded")
        print(f"Mobile page status: {m_response.status}")
        await mpage.wait_for_selector("#area_type", state="attached", timeout=10000)
        await asyncio.sleep(0.5)
        print("[PASS] Mobile Predictor form rendered.")

        # Mobile 2: Prediction Result
        await mpage.click("#predict-btn")
        await mpage.wait_for_selector("#result-body:not([hidden])", timeout=10000)
        await asyncio.sleep(0.5)
        m_price = await mpage.text_content("#price-num")
        m_unit = await mpage.text_content("#price-unit")
        print(f"[PASS] Mobile Prediction result: INR {m_price} {m_unit}")

        # Mobile 3: Instant API Console on Mobile
        await mpage.click("#tab-api")
        await mpage.wait_for_selector("#view-api.active", timeout=5000)
        await asyncio.sleep(0.5)
        print("[PASS] Mobile Instant API Console rendered.")

        # Check footer on mobile
        assert await mpage.is_visible(".site-footer")
        print("[PASS] Mobile footer verified.")

        await mobile_context.close()
        await browser.close()
        print("\nAll Desktop and Mobile Playwright verifications passed successfully without persisting screenshots!")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

    print(f"Starting background FastAPI uvicorn server on {BASE_URL} (Port {PORT})...")
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Healthcheck polling with HTTP client
    ready = False
    for i in range(40):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/meta", timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)

    if not ready:
        raise RuntimeError(f"FastAPI server failed to become healthy on {BASE_URL}")

    print("FastAPI server started and verified healthy via HTTP probe.")

    try:
        asyncio.run(run_playwright())
    finally:
        server.should_exit = True
        server_thread.join(timeout=3.0)

if __name__ == "__main__":
    main()
