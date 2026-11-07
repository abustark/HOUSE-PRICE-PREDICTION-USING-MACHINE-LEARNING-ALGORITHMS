"""
Playwright automated inspection script for PC and Mobile UI verification.
Runs in-process FastAPI server with uvicorn.
Performs end-to-end verification without saving persistent screenshots to the repository.
"""

import os
import sys
import asyncio
import tempfile
import uvicorn
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.server import app

PORT = 8018
BASE_URL = f"http://127.0.0.1:{PORT}"

async def main():
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

    temp_dir = tempfile.mkdtemp(prefix="playwright_transient_")

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
            print("\nRunning Desktop (1440x900) test assertions...")
            desktop_context = await browser.new_context(viewport={"width": 1440, "height": 900})
            page = await desktop_context.new_page()
            page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
            page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

            await page.goto(BASE_URL, wait_until="networkidle")
            await page.wait_for_selector("#area_type")
            await asyncio.sleep(0.5)

            # Desktop 1: Check form elements exist
            assert await page.is_visible("#area_type")
            assert await page.is_visible("#location")
            assert await page.is_visible("#predict-btn")
            print("✓ Desktop Predictor form rendered.")

            # Desktop 2: Click Predict button -> Live Prediction Result
            await page.click("#predict-btn")
            await page.wait_for_selector("#result-body:not([hidden])", timeout=5000)
            await asyncio.sleep(0.5)
            
            price_text = await page.text_content("#price-num")
            unit_text = await page.text_content("#price-unit")
            print(f"✓ Desktop Prediction result: ₹{price_text} {unit_text}")

            # Desktop 3: Click Instant API Console Tab -> Live Explorer
            await page.click("#tab-api")
            await page.wait_for_selector("#view-api.active", timeout=2000)
            # Click "Send Request" to test live endpoint in console
            await page.click("#btn-send-api")
            await page.wait_for_selector("#api-status-pill:not([hidden])", timeout=5000)
            status_text = await page.text_content("#api-status-pill")
            print(f"✓ Desktop API Console response: {status_text}")
            await asyncio.sleep(0.5)

            # Desktop 4: Click Model Performance Tab
            await page.click("#tab-metrics")
            await page.wait_for_selector("#view-metrics.active", timeout=2000)
            await asyncio.sleep(0.5)
            print("✓ Desktop Model Performance tab rendered.")

            # Check footer
            assert await page.is_visible(".site-footer")
            footer_text = await page.text_content(".site-footer")
            assert "Basith AbuSyed" in footer_text
            print("✓ Desktop footer verified with Basith AbuSyed copyright.")

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

            await mpage.goto(BASE_URL, wait_until="networkidle")
            await mpage.wait_for_selector("#area_type")
            await asyncio.sleep(0.5)
            print("✓ Mobile Predictor form rendered.")

            # Mobile 2: Prediction Result
            await mpage.click("#predict-btn")
            await mpage.wait_for_selector("#result-body:not([hidden])", timeout=5000)
            await asyncio.sleep(0.5)
            m_price = await mpage.text_content("#price-num")
            m_unit = await mpage.text_content("#price-unit")
            print(f"✓ Mobile Prediction result: ₹{m_price} {m_unit}")

            # Mobile 3: Instant API Console on Mobile
            await mpage.click("#tab-api")
            await mpage.wait_for_selector("#view-api.active", timeout=2000)
            await asyncio.sleep(0.5)
            print("✓ Mobile Instant API Console rendered.")

            # Check footer on mobile
            assert await mpage.is_visible(".site-footer")
            print("✓ Mobile footer verified.")

            await mobile_context.close()
            await browser.close()
            print("\nAll Desktop and Mobile Playwright verifications passed successfully without persisting screenshots!")

    finally:
        server.should_exit = True
        await server_task

if __name__ == "__main__":
    asyncio.run(main())
