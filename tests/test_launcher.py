"""Launcher tests. Real sockets and a real local HTTP server stand in for
Streamlit's health endpoint; streamlit.web.cli is replaced by a stub unless the
real package is installed and RUN_REAL_LAUNCH=1.
"""
import http.server
import os
import socket
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import launcher  # noqa: E402


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path == "/_stcore/health"
        self.send_response(200 if ok else 404)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


def start_health_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def occupy_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def stub_streamlit(main):
    cli = types.ModuleType("streamlit.web.cli")
    cli.main = main
    web = types.ModuleType("streamlit.web")
    web.cli = cli
    st = types.ModuleType("streamlit")
    st.web = web
    return {"streamlit": st, "streamlit.web": web, "streamlit.web.cli": cli}


class PathTests(unittest.TestCase):
    def test_app_found_from_source(self):
        self.assertEqual(launcher.app_path(), ROOT / "frontend" / "app.py")

    def test_frozen_root_uses_meipass(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sys, "_MEIPASS", d, create=True):
                self.assertEqual(launcher.resource_root(), Path(d))
                with self.assertRaises(FileNotFoundError):
                    launcher.app_path()
                (Path(d) / "frontend").mkdir()
                (Path(d) / "frontend" / "app.py").write_text("x")
                self.assertEqual(launcher.app_path(), Path(d) / "frontend" / "app.py")

    def test_prepare_environment_adds_bin_to_path(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bin").mkdir()
            with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}), mock.patch.object(sys, "path", list(sys.path)):
                launcher.prepare_environment(Path(d))
                self.assertTrue(os.environ["PATH"].startswith(str(Path(d) / "bin") + os.pathsep))
                self.assertIn(str(Path(d)), sys.path)
                self.assertIn(str(Path(d) / "backend"), sys.path)
                self.assertEqual(os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"], "false")

    def test_prepare_environment_without_bin(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"PATH": "/usr/bin"}):
            launcher.prepare_environment(Path(d))
            self.assertEqual(os.environ["PATH"], "/usr/bin")


class NetworkTests(unittest.TestCase):
    def test_port_free_and_busy(self):
        s, port = occupy_port()
        try:
            self.assertFalse(launcher.port_is_free(port))
        finally:
            s.close()
        self.assertTrue(launcher.port_is_free(port))

    def test_pick_port_skips_busy_preferred(self):
        s, port = occupy_port()
        try:
            got = launcher.pick_port(port)
            self.assertNotEqual(got, port)
            self.assertTrue(launcher.port_is_free(got))
        finally:
            s.close()

    def test_pick_port_falls_back_to_os_port(self):
        with mock.patch.object(launcher, "port_is_free", return_value=False):
            got = launcher.pick_port(8501, tries=3)
        self.assertGreater(got, 0)

    def test_health_detection(self):
        srv, port = start_health_server()
        try:
            self.assertTrue(launcher.is_streamlit_running(port))
        finally:
            srv.shutdown()
        s, dead = occupy_port()
        s.close()
        self.assertFalse(launcher.is_streamlit_running(dead))

    def test_non_streamlit_server_is_not_detected(self):
        class Other(Health):
            def do_GET(self):
                self.send_response(404)
                self.end_headers()
        srv = http.server.HTTPServer(("127.0.0.1", 0), Other)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            self.assertFalse(launcher.is_streamlit_running(srv.server_address[1]))
        finally:
            srv.shutdown()

    def test_wait_until_ready_success_and_timeout(self):
        srv, port = start_health_server()
        try:
            self.assertTrue(launcher.wait_until_ready(port, timeout=3))
        finally:
            srv.shutdown()
        s, dead = occupy_port()
        s.close()
        self.assertFalse(launcher.wait_until_ready(dead, timeout=0.6, interval=0.1))

    def test_server_that_comes_up_late(self):
        s, port = occupy_port()
        s.close()
        holder = {}

        def later():
            import time
            time.sleep(0.5)
            srv = http.server.HTTPServer(("127.0.0.1", port), Health)
            holder["srv"] = srv
            srv.serve_forever()

        threading.Thread(target=later, daemon=True).start()
        try:
            self.assertTrue(launcher.wait_until_ready(port, timeout=5, interval=0.1))
        finally:
            if "srv" in holder:
                holder["srv"].shutdown()


class BrowserTests(unittest.TestCase):
    def test_browser_opens_once_server_is_ready(self):
        srv, port = start_health_server()
        try:
            with mock.patch.object(launcher.webbrowser, "open") as op:
                launcher.open_browser_when_ready(port, timeout=3).join(5)
            op.assert_called_once_with(f"http://localhost:{port}")
        finally:
            srv.shutdown()

    def test_browser_not_opened_if_server_never_starts(self):
        s, dead = occupy_port()
        s.close()
        with mock.patch.object(launcher.webbrowser, "open") as op, mock.patch.object(
            launcher, "wait_until_ready", return_value=False
        ):
            launcher.open_browser_when_ready(dead, timeout=1).join(5)
        op.assert_not_called()


class MainTests(unittest.TestCase):
    def test_argv(self):
        argv = launcher.build_streamlit_argv(Path("/x/app.py"), 8600)
        self.assertEqual(argv[:3], ["streamlit", "run", str(Path("/x/app.py"))])
        pairs = dict(zip(argv[3::2], argv[4::2]))
        self.assertEqual(pairs["--server.port"], "8600")
        self.assertEqual(pairs["--server.headless"], "true")
        self.assertEqual(pairs["--server.address"], "127.0.0.1")
        self.assertEqual(pairs["--global.developmentMode"], "false")

    def test_main_runs_streamlit_and_opens_browser(self):
        seen = {}

        def fake_main():
            seen["argv"] = list(sys.argv)
            raise SystemExit(0)

        with mock.patch.dict(sys.modules, stub_streamlit(fake_main)), mock.patch.object(
            launcher, "is_streamlit_running", return_value=False
        ), mock.patch.object(launcher, "open_browser_when_ready") as ob, mock.patch.object(
            launcher, "pick_port", return_value=8777
        ), mock.patch.object(sys, "argv", list(sys.argv)):
            code = launcher.main()
        self.assertEqual(code, 0)
        ob.assert_called_once_with(8777)
        self.assertEqual(seen["argv"][:2], ["streamlit", "run"])
        self.assertTrue(seen["argv"][2].endswith(os.path.join("frontend", "app.py")))
        self.assertIn("8777", seen["argv"])

    def test_main_exit_code_is_passed_through(self):
        def fake_main():
            raise SystemExit(3)
        with mock.patch.dict(sys.modules, stub_streamlit(fake_main)), mock.patch.object(
            launcher, "is_streamlit_running", return_value=False
        ), mock.patch.object(launcher, "open_browser_when_ready"), mock.patch.object(
            sys, "argv", list(sys.argv)
        ):
            self.assertEqual(launcher.main(), 3)

    def test_ctrl_c_is_clean(self):
        def fake_main():
            raise KeyboardInterrupt
        with mock.patch.dict(sys.modules, stub_streamlit(fake_main)), mock.patch.object(
            launcher, "is_streamlit_running", return_value=False
        ), mock.patch.object(launcher, "open_browser_when_ready"), mock.patch.object(
            sys, "argv", list(sys.argv)
        ):
            self.assertEqual(launcher.main(), 0)

    def test_already_running_just_opens_browser(self):
        with mock.patch.object(launcher, "is_streamlit_running", return_value=True), mock.patch.object(
            launcher.webbrowser, "open"
        ) as op, mock.patch.object(launcher, "open_browser_when_ready") as ob:
            self.assertEqual(launcher.main(), 0)
        op.assert_called_once_with(f"http://localhost:{launcher.PREFERRED_PORT}")
        ob.assert_not_called()

    def test_missing_streamlit_gives_clear_message(self):
        blocked = {"streamlit": None, "streamlit.web": None, "streamlit.web.cli": None}
        with mock.patch.dict(sys.modules, blocked), mock.patch.object(
            launcher, "is_streamlit_running", return_value=False
        ), mock.patch("sys.stderr") as err:
            self.assertEqual(launcher.main(), 1)
        self.assertIn("pip install", "".join(c.args[0] for c in err.write.call_args_list))

    def test_missing_app_is_reported(self):
        with mock.patch.object(launcher, "app_path", side_effect=FileNotFoundError("nope")), mock.patch(
            "sys.stderr"
        ):
            self.assertEqual(launcher.main(), 1)


@unittest.skipUnless(os.environ.get("RUN_REAL_LAUNCH") == "1", "set RUN_REAL_LAUNCH=1 (needs streamlit)")
class RealLaunchTest(unittest.TestCase):
    def test_real_server_boots_and_answers_health(self):
        import subprocess
        port = launcher.pick_port(8650)
        env = {**os.environ, "BROWSER": "true"}
        proc = subprocess.Popen(
            [sys.executable, "-c",
             f"import launcher,sys;launcher.PREFERRED_PORT={port};launcher.pick_port=lambda *a,**k:{port};"
             "launcher.webbrowser.open=lambda *a,**k:True;sys.exit(launcher.main())"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(launcher.wait_until_ready(port, timeout=60))
        finally:
            proc.terminate()
            proc.wait(10)


if __name__ == "__main__":
    unittest.main()
