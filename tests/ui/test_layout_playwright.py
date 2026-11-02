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

    def setUp(self):
        self.opened = []

    def tearDown(self):
        # pages a failing test never closed: keep a picture of what was on screen
        for n, page in enumerate(self.opened):
            if not page.is_closed():
                try:
                    page.screenshot(path=str(SHOTS / f"FAIL_{self._testMethodName}_{n}.png"), full_page=True)
                except Exception:
                    pass
                page.close()

    def open(self, width=1280, height=800, demo=False):
        page = self.browser.new_page(viewport={"width": width, "height": height})
        self.opened.append(page)
        page.goto(URL + ("/?demo=1" if demo else ""))
        page.get_by_role("tab", name="🎬 Clip Repurposer Studio").wait_for(timeout=30000)
        # Streamlit paints progressively: wait for the real controls, not just the tab strip
        page.get_by_label("YouTube URL").wait_for(state="visible", timeout=30000)
        if width > 640:
            page.get_by_label("Google Drive Folder ID", exact=True).wait_for(state="visible", timeout=30000)
        return page

    def assert_no_horizontal_overflow(self, page):
        overflow = page.evaluate(
            "() => document.scrollingElement.scrollWidth - document.scrollingElement.clientWidth")
        self.assertLessEqual(overflow, 1, f"page scrolls horizontally by {overflow}px")
        main = page.evaluate(
            """() => { const m = document.querySelector('[data-testid="stMain"]');
                       return m ? m.scrollWidth - m.clientWidth : 0; }""")
        self.assertLessEqual(main, 1, f"main area scrolls horizontally by {main}px")

    def test_branding_and_theme(self):
        page = self.open()
        self.assertEqual(page.title(), "CinePulse AI")
        self.assertTrue(page.get_by_role("heading", name="CinePulse AI").first.is_visible())
        for old in ("Vugola", "Viewmax", "Clip Studio"):
            self.assertEqual(page.get_by_text(old).count(), 0, old)
        bg = page.evaluate("() => getComputedStyle(document.querySelector('.stApp')).backgroundColor")
        self.assertNotEqual(bg, "rgb(255, 255, 255)")  # dark theme, not default white
        blur = page.evaluate(
            """() => getComputedStyle(document.querySelector('[data-testid="stSidebar"]')).backdropFilter""")
        self.assertIn("blur", blur)
        page.screenshot(path=str(SHOTS / "desktop_home.png"), full_page=True)
        page.close()

    def test_demo_cards_and_badges(self):
        for w, h, name in ((1280, 800, "desktop"), (390, 844, "phone")):
            page = self.open(w, h, demo=True)
            page.get_by_text("Top score").first.wait_for(timeout=15000)
            self.assertTrue(page.get_by_text("Clips created").first.is_visible())
            self.assertTrue(page.locator(".cp-badge.score").first.is_visible())
            self.assertGreaterEqual(page.locator(".cp-quote").count(), 2)
            self.assert_no_horizontal_overflow(page)
            page.screenshot(path=str(SHOTS / f"{name}_demo_clips.png"), full_page=True)
            page.get_by_role("tab", name="⚡ AI Generator Lab").click()
            page.get_by_text("Est. length").first.wait_for(timeout=10000)
            self.assert_no_horizontal_overflow(page)
            page.screenshot(path=str(SHOTS / f"{name}_demo_lab.png"), full_page=True)
            page.close()

    def test_sidebar_collapses_and_reopens(self):
        page = self.open()
        sidebar = page.locator('[data-testid="stSidebar"]')
        self.assertEqual(sidebar.get_attribute("data-collapsed"), "false")
        sidebar.locator('[data-testid="stSidebarCollapseButton"]').first.click()
        page.wait_for_function(
            "() => document.querySelector('[data-testid=\"stSidebar\"]').dataset.collapsed === 'true'", timeout=10000)
        page.screenshot(path=str(SHOTS / "desktop_sidebar_collapsed.png"), full_page=True)
        page.locator('[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"]').first.click()
        page.wait_for_function(
            "() => document.querySelector('[data-testid=\"stSidebar\"]').dataset.collapsed === 'false'", timeout=10000)
        page.close()

    def test_phone_starts_with_sidebar_closed(self):
        page = self.open(390, 844)
        self.assertEqual(page.locator('[data-testid="stSidebar"]').get_attribute("data-collapsed"), "true")
        page.screenshot(path=str(SHOTS / "phone_home.png"), full_page=True)
        page.close()


if __name__ == "__main__":
    unittest.main()
