"""Playwright layout test for the real Streamlit app.

Starts `streamlit run frontend/app.py`, opens it in headless Chromium and checks
the two tabs, the sidebar Folder ID field, the main controls, validation
messages, and that nothing overflows horizontally on desktop or phone widths.
Screenshots go to tests/ui/screenshots/.

Needs:  pip install streamlit playwright && playwright install chromium
Run:    python -m unittest tests.ui.test_layout_playwright -v
Skips itself (with the reason) when those are not available.
"""
import os
import shutil
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHOTS = Path(__file__).resolve().parent / "screenshots"
PORT = int(os.environ.get("UI_TEST_PORT", "8599"))
URL = f"http://localhost:{PORT}"


def _skip_reason():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        return "streamlit is not installed"
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "playwright is not installed"
    try:
        with sync_playwright() as p:
            p.chromium.launch().close()
    except Exception as exc:  # browser binary missing
        return f"Chromium is not available: {str(exc).splitlines()[0]}"
    return None


REASON = _skip_reason()


@unittest.skipIf(REASON, REASON or "")
class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright

        SHOTS.mkdir(exist_ok=True)
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "app.py"),
             "--server.headless", "true", "--server.port", str(PORT),
             "--browser.gatherUsageStats", "false"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{URL}/_stcore/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            cls.server.kill()
            raise RuntimeError("Streamlit server did not start")
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.terminate()

    def open(self, width=1280, height=800):
        page = self.browser.new_page(viewport={"width": width, "height": height})
        page.goto(URL)
        page.get_by_role("tab", name="Vugola AI Mode").wait_for(timeout=30000)
        return page

    def assert_no_horizontal_overflow(self, page):
        overflow = page.evaluate(
            "() => document.scrollingElement.scrollWidth - document.scrollingElement.clientWidth")
        self.assertLessEqual(overflow, 1, f"page scrolls horizontally by {overflow}px")

    def test_two_tabs_and_sidebar(self):
        page = self.open()
        self.assertTrue(page.get_by_role("tab", name="Vugola AI Mode").is_visible())
        self.assertTrue(page.get_by_role("tab", name="Viewmax Mode").is_visible())
        sidebar = page.locator('[data-testid="stSidebar"]')
        self.assertTrue(sidebar.get_by_label("Google Drive Folder ID").is_visible())
        page.screenshot(path=str(SHOTS / "desktop_vugola.png"), full_page=True)
        page.close()

    def test_vugola_controls(self):
        page = self.open()
        self.assertTrue(page.get_by_label("YouTube URL").is_visible())
        self.assertTrue(page.get_by_text("or upload a video file").is_visible())
        self.assertTrue(page.get_by_role("button", name="Generate viral clips").is_visible())
        page.close()

    def test_viewmax_controls(self):
        page = self.open()
        page.get_by_role("tab", name="Viewmax Mode").click()
        self.assertTrue(page.get_by_label("What should the video be about?").is_visible())
        self.assertTrue(page.get_by_role("button", name="Generate video").is_visible())
        page.screenshot(path=str(SHOTS / "desktop_viewmax.png"), full_page=True)
        page.close()

    def test_validation_messages(self):
        page = self.open()
        page.get_by_role("button", name="Generate viral clips").click()
        page.get_by_text("Paste a YouTube URL or upload a video first.").wait_for(timeout=10000)
        page.get_by_role("tab", name="Viewmax Mode").click()
        page.get_by_role("button", name="Generate video").click()
        page.get_by_text("Enter a prompt first.").wait_for(timeout=10000)
        page.close()

    def test_no_horizontal_overflow_desktop_and_phone(self):
        for w, h, name in ((1280, 800, "desktop"), (390, 844, "phone")):
            page = self.open(w, h)
            self.assert_no_horizontal_overflow(page)
            page.get_by_role("tab", name="Viewmax Mode").click()
            self.assert_no_horizontal_overflow(page)
            page.screenshot(path=str(SHOTS / f"{name}_viewmax.png"), full_page=True)
            page.close()


if __name__ == "__main__":
    unittest.main()
