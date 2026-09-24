"""Tests for the demo web proxy; no checkpoints or network required."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import server


class FakeUpstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def _answer(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        FakeUpstream.seen.append((self.command, self.path, body))
        if self.path == "/v1/tokenize":
            status, out = 404, {"error": {"message": "no such model", "type": "not_found"}}
        else:
            status, out = 200, {"echo": self.path, "method": self.command}
        data = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = _answer


def serve(httpd):
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = serve(ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream))
        up = "http://127.0.0.1:%d" % cls.upstream.server_address[1]
        cls.proxy = serve(server.make_server("127.0.0.1", 0, up))
        cls.base = "http://127.0.0.1:%d" % cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.upstream.shutdown()

    def setUp(self):
        FakeUpstream.seen.clear()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data, method=method,
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_allowed_call_is_forwarded_with_its_body(self):
        status, body = self.call("POST", "/api/v1/probe", {"text": "x"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"echo": "/v1/probe", "method": "POST"})
        self.assertEqual(FakeUpstream.seen, [("POST", "/v1/probe", b'{"text": "x"}')])

    def test_unlisted_path_never_reaches_upstream(self):
        for method, path in (("POST", "/api/v1/classify"), ("GET", "/api/readyz"),
                             ("POST", "/api/v1/opinion/specs/../../probe")):
            with self.subTest(path=path):
                status, _ = self.call(method, path, {} if method == "POST" else None)
                self.assertEqual(status, 404)
        self.assertEqual(FakeUpstream.seen, [])

    def test_wrong_method_on_a_listed_path_is_refused(self):
        status, _ = self.call("GET", "/api/v1/probe")
        self.assertEqual(status, 404)
        self.assertEqual(FakeUpstream.seen, [])

    def test_upstream_error_status_is_passed_through_not_masked(self):
        status, body = self.call("POST", "/api/v1/tokenize", {"model": "nope", "text": "x"})
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"]["type"], "not_found")

    def test_dead_upstream_is_a_502(self):
        dead = serve(server.make_server("127.0.0.1", 0, "http://127.0.0.1:1"))
        try:
            base = "http://127.0.0.1:%d" % dead.server_address[1]
            req = urllib.request.Request(base + "/api/v1/adjudicator")
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(e.exception.code, 502)
        finally:
            dead.shutdown()

    def test_static_page_is_served_and_traversal_is_not(self):
        status, body = self.call("GET", "/sour-note")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>", body)
        for path in ("/../server.py", "/%2e%2e/server.py", "/static/../server.py"):
            with self.subTest(path=path):
                status, _ = self.call("GET", path)
                self.assertEqual(status, 404)


class HostTests(unittest.TestCase):
    def test_explicit_host_wins(self):
        self.assertEqual(server.resolve_host("127.0.0.1", run=None), "127.0.0.1")

    def test_wildcard_bind_is_refused(self):
        for host in ("0.0.0.0", "::", ""):
            with self.subTest(host=host), self.assertRaises(SystemExit):
                server.resolve_host(host, run=None)

    def test_default_is_the_tailnet_address(self):
        class Done:
            returncode, stdout = 0, "100.64.0.7\nfd7a::1\n"
        self.assertEqual(server.resolve_host(None, run=lambda *a, **k: Done()), "100.64.0.7")

    def test_no_tailnet_and_no_host_fails_loudly(self):
        def missing(*a, **k):
            raise FileNotFoundError("tailscale")
        with self.assertRaises(SystemExit):
            server.resolve_host(None, run=missing)


class PageTests(unittest.TestCase):
    def test_every_page_has_a_title_and_only_calls_the_proxy(self):
        pages = sorted((Path(server.__file__).parent / "static").glob("*.html"))
        self.assertTrue(pages)
        for page in pages:
            text = page.read_text()
            with self.subTest(page=page.name):
                self.assertIn("<title>", text)
                self.assertNotIn("ts.net", text)
                self.assertNotIn("http://", text.replace("http://www.w3.org", ""))

    def test_one_pass_props_are_well_formed(self):
        static = Path(server.__file__).parent / "static"
        spec = json.loads((static / "command-verdict-enum-v1.json").read_text())
        self.assertIsInstance(spec.get("input_label"), str)
        rows = json.loads((static / "one-pass-commands.json").read_text())["rows"]
        inputs = [r["input"] for r in rows]
        self.assertEqual(len(inputs), len(set(inputs)))
        self.assertTrue(all(isinstance(r["hurts"], bool) for r in rows))
        self.assertEqual({r["hurts"] for r in rows}, {True, False})

    def test_two_worlds_props_are_well_formed(self):
        static = Path(server.__file__).parent / "static"
        cases = json.loads((static / "two-worlds.json").read_text())["cases"]
        self.assertEqual(len({c["input"] for c in cases}), len(cases))
        for c in cases:
            self.assertEqual(len(c["worlds"]), 2)
            for w in c["worlds"]:
                for key in ("name", "before", "output"):
                    self.assertTrue(w[key].strip(), key)
                self.assertTrue(w["fact"].strip() and "\n" not in w["fact"])


if __name__ == "__main__":
    unittest.main()
