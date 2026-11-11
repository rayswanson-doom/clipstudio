"""Playwright layout test for the real Streamlit app.

Starts `streamlit run frontend/app.py`, opens it in headless Chromium and checks
the four creation modes, the sidebar Folder ID field, the main controls, validation
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

    CLASSIC = "Narrated video with a 3D avatar (classic)"

    def choose(self, page, label, option):
        """Pick an option of a Streamlit selectbox the way a user does: open it and click the option; when the
        popover does not open (it can close while the page is still scrolling), type the option into the
        searchable select and press Enter, which is what a keyboard user would do."""
        box = page.get_by_label(label, exact=True)
        box.scroll_into_view_if_needed()
        box.click()
        try:
            page.get_by_role("option", name=option, exact=True).click(timeout=6000)
        except Exception:
            box.click()
            page.keyboard.type(option)
            page.get_by_role("option", name=option, exact=True).first.wait_for(state="visible", timeout=8000)
            page.keyboard.press("Enter")

    def open(self, width=1280, height=800, demo=False, mode=None):
        """mode: None (the default, Idea2Video), a creation-mode name, or "classic" (Idea2Video > narrated generator)."""
        page = self.browser.new_page(viewport={"width": width, "height": height})
        self.opened.append(page)
        page.goto(URL + ("/?demo=1" if demo else ""))
        # Streamlit paints progressively: wait for the real controls, not just the page shell
        page.get_by_label("Creation mode", exact=True).wait_for(state="visible", timeout=30000)
        if width > 640:
            page.get_by_label("Google Drive Folder ID", exact=True).wait_for(state="visible", timeout=30000)
        if mode == "classic":
            self.choose(page, "Output style", self.CLASSIC)
            page.get_by_label("What should the video be about?").wait_for(state="visible", timeout=15000)
        elif mode:
            self.choose(page, "Creation mode", mode)
            wait = {"Clip Repurposer": "YouTube URL"}.get(mode, "Your novel or long text" if mode == "Novel2Video" else "Your screenplay")
            page.get_by_label(wait).wait_for(state="visible", timeout=15000)
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
        page.get_by_role("heading", name="CinePulse AI").first.wait_for(state="visible", timeout=15000)
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
            page = self.open(w, h, demo=True, mode="Clip Repurposer")
            page.get_by_text("Top score").first.wait_for(timeout=15000)
            page.get_by_text("Clips created").first.wait_for(state="visible", timeout=15000)
            page.locator(".cp-badge.score").first.wait_for(state="visible", timeout=15000)
            self.assertGreaterEqual(page.locator(".cp-quote").count(), 2)
            self.assert_no_horizontal_overflow(page)
            page.screenshot(path=str(SHOTS / f"{name}_demo_clips.png"), full_page=True)
            self.choose(page, "Creation mode", "Idea2Video")
            self.choose(page, "Output style", self.CLASSIC)
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

    def test_avatar_checkbox_enables_uploader_and_phone_fits(self):
        for w, h, name in ((1280, 800, "desktop"), (390, 844, "phone")):
            page = self.open(w, h, mode="classic")
            box = page.get_by_text("Enable AI Avatar Lip-Sync Commentary")
            box.wait_for(state="visible", timeout=15000)
            drop = page.locator('[data-testid="stFileUploader"]').filter(has_text="Your model").locator(
                '[data-testid="stFileUploaderDropzone"]')  # avatar model / picture
            self.assertTrue(drop.locator("input[type=file]").is_disabled())
            box.click()
            page.wait_for_function(
                """() => { const u = [...document.querySelectorAll('[data-testid="stFileUploader"]')]
                        .find(e => e.textContent.includes('Your model'));
                    return !!u && !u.querySelector('input[type=file]').disabled; }""",
                timeout=10000)
            self.assert_no_horizontal_overflow(page)
            box.scroll_into_view_if_needed()
            page.screenshot(path=str(SHOTS / f"{name}_avatar.png"), full_page=True)
            page.close()

    def test_four_creation_modes_and_sidebar(self):
        page = self.open()
        sidebar = page.locator('[data-testid="stSidebar"]')
        self.assertTrue(sidebar.get_by_label("Google Drive Folder ID", exact=True).is_visible())
        self.assertEqual(page.get_by_role("tab").count(), 0)  # one Lab, not separate tabs
        page.get_by_label("Creation mode", exact=True).click()
        for name in ("Idea2Video", "Novel2Video", "Script2Video", "Clip Repurposer"):
            self.assertTrue(page.get_by_role("option", name=name, exact=True).is_visible(), name)
        page.keyboard.press("Escape")
        page.screenshot(path=str(SHOTS / "desktop_home_modes.png"), full_page=True)
        page.close()

    def test_story_modes_show_their_own_controls(self):
        for mode, field, uploader in (("Idea2Video", "Your idea", False), ("Novel2Video", "Your novel or long text", True),
                                      ("Script2Video", "Your screenplay", True)):
            page = self.open(mode=None if mode == "Idea2Video" else mode)
            page.get_by_label(field).wait_for(state="visible", timeout=15000)
            self.assertEqual(page.get_by_text("...or upload a text file").count(), 1 if uploader else 0, mode)
            page.get_by_role("button", name="Parse the story and continue").wait_for(state="visible", timeout=10000)
            for header in ("Story options", "Character consistency", "Camera angle preferences", "Podcast corner avatar overlay"):
                page.get_by_text(header, exact=True).wait_for(state="visible", timeout=10000)
            self.assertEqual(page.get_by_label("YouTube URL").count(), 0)
            page.screenshot(path=str(SHOTS / f"desktop_{mode}.png"), full_page=True)
            page.close()

    def test_consistency_camera_and_corner_controls(self):
        page = self.open()
        page.get_by_text("Character consistency", exact=True).click()
        page.get_by_text("Show each character's portrait in every shot they appear in").wait_for(state="visible", timeout=10000)
        page.get_by_text("Start each shot from the previous shot's last frame (scene continuity)").wait_for(state="visible", timeout=10000)
        page.get_by_text("Camera angle preferences", exact=True).click()
        for label in ("Shot size", "Camera angle", "Camera movement"):
            page.get_by_label(label, exact=True).wait_for(state="visible", timeout=10000)
        page.get_by_text("Podcast corner avatar overlay", exact=True).click()
        box = page.get_by_text("Show a host / narrator avatar in a corner")
        box.wait_for(state="visible", timeout=10000)
        corner = page.get_by_label("Corner", exact=True)
        self.assertTrue(corner.is_disabled())
        box.click()
        page.wait_for_function("""() => { const e = [...document.querySelectorAll('input')].find(
            i => i.getAttribute('aria-label') === 'Corner'); return !!e && !e.disabled; }""", timeout=10000)
        self.choose(page, "Corner", "Custom Coordinates")
        page.get_by_label("X (pixels from the left)").wait_for(state="visible", timeout=10000)
        page.get_by_label("Y (pixels from the top)").wait_for(state="visible", timeout=10000)
        self.assert_no_horizontal_overflow(page)
        page.screenshot(path=str(SHOTS / "desktop_story_options.png"), full_page=True)
        page.close()

    def test_meme_reaction_controls(self):
        for mode in ("Clip Repurposer", "classic"):
            page = self.open(mode=mode) if mode else self.open()
            page.get_by_text("Contextual meme reactions").first.click()
            box = page.get_by_text("Enable Contextual Meme Reactions")
            box.wait_for(state="visible", timeout=10000)
            page.get_by_text("Reaction clip library").wait_for(state="visible", timeout=10000)
            page.get_by_role("button", name="Create the starter pack").wait_for(state="visible", timeout=10000)
            self.assertTrue(page.get_by_label("Tile corner", exact=True).is_disabled())
            box.click()
            page.wait_for_function("""() => { const e = [...document.querySelectorAll('input')].find(
                i => i.getAttribute('aria-label') === 'Tile corner'); return !!e && !e.disabled; }""", timeout=10000)
            self.assert_no_horizontal_overflow(page)
            page.close()

    def test_three_step_wizard_flow(self):
        """Story -> Asset Studio (cards, generate pictures, gate) -> Render step with locked assets and meme toggle."""
        script = ("Title: Kitchen\n\nINT. KITCHEN - NIGHT\n\nMaya, a tall woman in a red apron with long black hair, stirs a pot.\n\n"
                  "MAYA\nLOL that was hilarious!\n\nBOB\nNo way!\n\nEXT. STREET - DAY\n\nBob, an old man in a grey coat, waves.\n\nBOB\nGood morning!\n")
        page = self.open(mode="Script2Video")
        box = page.get_by_label("Your screenplay")
        box.fill(script)
        box.press("Control+Enter")
        page.get_by_role("button", name="Parse the story and continue").click()
        page.get_by_text("Pre-production asset gate").wait_for(timeout=60000)
        for name in ("Maya", "Bob", "Kitchen", "Street"):
            page.get_by_text(name, exact=True).first.wait_for(state="visible", timeout=10000)
        self.assertTrue(page.get_by_role("button", name="Lock the assets and continue to step 3").is_disabled())
        self.choose(page, "How should pictures be generated?", "Built-in painter (offline, simple flat art)")
        page.get_by_role("button", name="Generate Reference Visuals").click()
        page.get_by_text("Ready").first.wait_for(timeout=60000)
        page.wait_for_function("""() => { const b = [...document.querySelectorAll('button')].find(
            x => x.innerText.includes('Lock the assets')); return !!b && !b.disabled; }""", timeout=60000)
        self.assert_no_horizontal_overflow(page)
        page.screenshot(path=str(SHOTS / "desktop_asset_studio.png"), full_page=True)
        page.get_by_role("button", name="Lock the assets and continue to step 3").click()
        page.get_by_text("Storyboard (assets locked)").wait_for(timeout=30000)
        page.get_by_role("button", name="Render the episode").wait_for(state="visible", timeout=10000)
        page.get_by_text("Visual style").first.wait_for(state="visible", timeout=10000)
        page.get_by_text("2D Cinematic Composition Mode", exact=True).first.wait_for(state="visible", timeout=10000)
        page.get_by_text("Generative AI Video Mode", exact=True).first.click()
        page.get_by_text("Video model").first.wait_for(state="visible", timeout=15000)
        page.get_by_text("pollinations").first.wait_for(state="visible", timeout=10000)
        self.assert_no_horizontal_overflow(page)
        page.screenshot(path=str(SHOTS / "desktop_wizard_generative.png"), full_page=True)
        page.get_by_text("2D Cinematic Composition Mode", exact=True).first.click()
        page.get_by_text("Contextual meme reactions").first.click()
        page.get_by_text("Enable Contextual Meme Reactions").wait_for(state="visible", timeout=10000)
        self.assert_no_horizontal_overflow(page)
        page.screenshot(path=str(SHOTS / "desktop_wizard_step3.png"), full_page=True)
        page.close()

    def test_story_validation_message(self):
        page = self.open(mode="Script2Video")
        page.get_by_role("button", name="Parse the story and continue").click()
        page.get_by_text("Type or upload some text first.").wait_for(timeout=10000)
        page.close()

    def test_clip_tab_controls(self):
        page = self.open(mode="Clip Repurposer")
        page.get_by_label("YouTube URL").wait_for(state="visible", timeout=15000)
        page.get_by_text("or upload a video file").wait_for(state="visible", timeout=10000)
        page.get_by_role("button", name="Generate viral clips").wait_for(state="visible", timeout=10000)
        page.get_by_text("Clip options").click()
        page.get_by_text("Also read on-screen text (OCR)").wait_for(state="visible", timeout=10000)
        page.close()

    def test_lab_tab_controls_including_script_and_actions(self):
        page = self.open(mode="classic")
        page.get_by_label("What should the video be about?").wait_for(state="visible", timeout=15000)
        page.get_by_role("button", name="Generate video").wait_for(state="visible", timeout=10000)
        page.get_by_text("Script & action narration (optional)").click()
        page.get_by_label("Your own script").wait_for(state="visible", timeout=10000)
        page.get_by_label("Visual actions").wait_for(state="visible", timeout=10000)
        page.get_by_text("Read & narrate the on-screen text of the background video (OCR)").wait_for(state="visible", timeout=10000)
        box = page.get_by_text("Enable AI Avatar Lip-Sync Commentary")
        box.scroll_into_view_if_needed()
        self.assertTrue(box.is_visible())
        page.screenshot(path=str(SHOTS / "desktop_lab_script.png"), full_page=True)
        page.close()

    def test_validation_messages(self):
        page = self.open(mode="Clip Repurposer")
        page.get_by_role("button", name="Generate viral clips").click()
        page.get_by_text("Paste a YouTube URL or upload a video first.").wait_for(timeout=10000)
        self.choose(page, "Creation mode", "Idea2Video")
        self.choose(page, "Output style", self.CLASSIC)
        page.get_by_role("button", name="Generate video").click()
        page.get_by_text("Enter a prompt or paste your own script first.").wait_for(timeout=10000)
        page.close()

    def test_no_horizontal_overflow_desktop_and_phone(self):
        for w, h, name in ((1280, 800, "desktop"), (390, 844, "phone")):
            page = self.open(w, h)
            self.assert_no_horizontal_overflow(page)
            for header in ("Character consistency", "Camera angle preferences", "Podcast corner avatar overlay"):
                page.get_by_text(header, exact=True).click()
            self.assert_no_horizontal_overflow(page)
            page.screenshot(path=str(SHOTS / f"{name}_story_open.png"), full_page=True)
            page.close()
            page = self.open(w, h, mode="classic")
            page.get_by_text("Script & action narration (optional)").click()
            page.get_by_label("Your own script").wait_for(state="visible", timeout=10000)
            self.assert_no_horizontal_overflow(page)
            page.screenshot(path=str(SHOTS / f"{name}_lab_open.png"), full_page=True)
            page.close()

    def test_phone_starts_with_sidebar_closed(self):
        page = self.open(390, 844)
        self.assertEqual(page.locator('[data-testid="stSidebar"]').get_attribute("data-collapsed"), "true")
        page.screenshot(path=str(SHOTS / "phone_home.png"), full_page=True)
        page.close()


if __name__ == "__main__":
    unittest.main()
