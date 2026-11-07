"""
Playwright automated inspection script for PC and Mobile UI verification.
Runs in-process FastAPI server with uvicorn.
Saves screenshots to artifacts/ and conversation brain artifacts/ per AGENTS.md.
"""

import os
import sys
import asyncio
import uvicorn
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.server import app

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PORT = 8018
BASE_URL = f"http://127.0.0.1:{PORT}"

async def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Start in-process uvicorn server
    print(f"Starting in-process FastAPI uvicorn server on {BASE_URL}...")
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    
    # Wait for server to bind
    for _ in range(15):
        if server.started:
            break
        await asyncio.sleep(0.2)
    print("FastAPI server started successfully.")

    try:
        async with async_playwright() as p:
            # Try system browser channels (msedge or chrome)
            try:
                browser = await p.chromium.launch(headless=True, channel="msedge")
                print("Browser launched via msedge channel.")
            except Exception:
                browser = await p.chromium.launch(headless=True, channel="chrome")
                print("Browser launched via chrome channel.")

            # --- 1. DESKTOP TEST (1440x900) ---
            print("\nCapturing Desktop (1440x900) viewports...")
            desktop_context = await browser.new_context(viewport={"width": 1440, "height": 900})
            page = await desktop_context.new_page()
            page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
            page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

            await page.goto(BASE_URL, wait_until="networkidle")
            await page.wait_for_selector("#area_type")
            await asyncio.sleep(0.5)

            # Desktop 1: Predictor initial state
            p1 = os.path.join(ARTIFACTS_DIR, "playwright_desktop_predictor.png")
            await page.screenshot(path=p1, full_page=True)
            print(f"Captured: {p1}")

            # Desktop 2: Click Predict button -> Live Prediction Result
            await page.click("#predict-btn")
            await page.wait_for_selector("#result-body:not([hidden])", timeout=5000)
            await asyncio.sleep(0.5)

            p2 = os.path.join(ARTIFACTS_DIR, "playwright_desktop_result.png")
            await page.screenshot(path=p2, full_page=True)
            print(f"Captured: {p2}")

            # Desktop 3: Click Instant API Console Tab -> Live Explorer
            await page.click("#tab-api")
            await page.wait_for_selector("#view-api.active", timeout=2000)
            # Click "Send Request" to test live endpoint in console
            await page.click("#btn-send-api")
            await page.wait_for_selector("#api-status-pill:not([hidden])", timeout=5000)
            await asyncio.sleep(0.5)

            p3 = os.path.join(ARTIFACTS_DIR, "playwright_desktop_api_console.png")
            await page.screenshot(path=p3, full_page=True)
            print(f"Captured: {p3}")

            # Desktop 4: Click Model Performance Tab
            await page.click("#tab-metrics")
            await page.wait_for_selector("#view-metrics.active", timeout=2000)
            await asyncio.sleep(0.5)

            p4 = os.path.join(ARTIFACTS_DIR, "playwright_desktop_metrics.png")
            await page.screenshot(path=p4, full_page=True)
            print(f"Captured: {p4}")

            await desktop_context.close()

            # --- 2. MOBILE TEST (390x844 - iPhone / Pixel) ---
            print("\nCapturing Mobile (390x844) viewports...")
            mobile_context = await browser.new_context(
                viewport={"width": 390, "height": 844},
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
                is_mobile=True,
                has_touch=True,
            )
            mpage = await mobile_context.new_page()

            await mpage.goto(BASE_URL, wait_until="networkidle")
            await mpage.wait_for_selector("#area_type")
            await asyncio.sleep(0.5)

            # Mobile 1: Initial Predictor
            m1 = os.path.join(ARTIFACTS_DIR, "playwright_mobile_predictor.png")
            await mpage.screenshot(path=m1, full_page=True)
            print(f"Captured: {m1}")

            # Mobile 2: Prediction Result
            await mpage.click("#predict-btn")
            await mpage.wait_for_selector("#result-body:not([hidden])", timeout=5000)
            await asyncio.sleep(0.5)

            m2 = os.path.join(ARTIFACTS_DIR, "playwright_mobile_result.png")
            await mpage.screenshot(path=m2, full_page=True)
            print(f"Captured: {m2}")

            # Mobile 3: Instant API Console on Mobile
            await mpage.click("#tab-api")
            await mpage.wait_for_selector("#view-api.active", timeout=2000)
            await asyncio.sleep(0.5)

            m3 = os.path.join(ARTIFACTS_DIR, "playwright_mobile_api_console.png")
            await mpage.screenshot(path=m3, full_page=True)
            print(f"Captured: {m3}")

            await mobile_context.close()
            await browser.close()
            print("\nAll Desktop and Mobile Playwright screenshots captured successfully!")

    finally:
        server.should_exit = True
        await server_task

if __name__ == "__main__":
    asyncio.run(main())
