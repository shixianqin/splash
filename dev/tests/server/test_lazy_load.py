"""HTTP behavior while an optional model load is pending."""

import concurrent.futures
import json
import threading
import time

from dev.tests.server_fixtures import HarnessTestCase


class LazyLoadTests(HarnessTestCase):
    @staticmethod
    def body():
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4,
            "reasoning_effort": "none",
        }

    def cold_server(self, *, queue_size=4, start=None):
        harness = self.harness(queue_size=queue_size)
        app = harness.app
        harness.server.app = None
        harness.server.configure_lazy(
            start or (lambda: app),
            model="test-model",
            model_names=("alias", "test-model"),
        )
        return harness

    def test_catalog_and_probes_do_not_start_model(self):
        calls = []
        harness = self.cold_server(start=lambda: calls.append(1) or harness.app)
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        self.assertEqual(harness.request("GET", "/ready")[0], 503)
        status, _, payload = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(
            [item["id"] for item in json.loads(payload)["data"]],
            ["alias", "test-model"],
        )
        self.assertNotIn("max_model_len", json.loads(payload)["data"][0])
        self.assertEqual(harness.request("GET", "/v1/models/alias")[0], 200)
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["startup"]["state"], "waiting")
        self.assertIn("splash_ready 0", harness.request("GET", "/metrics")[2].decode())
        self.assertEqual(calls, [])

    def test_concurrent_requests_wait_for_one_startup(self):
        entered = threading.Event()
        release = threading.Event()
        harness = self.cold_server()
        app = harness.app
        calls = []

        def start():
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(2))
            return app

        harness.server.configure_lazy(
            start, model="test-model", model_names=("test-model",)
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                harness.request, "POST", "/v1/chat/completions", self.body()
            )
            self.assertTrue(entered.wait(1))
            second = pool.submit(
                harness.request, "POST", "/v1/chat/completions", self.body()
            )
            deadline = time.monotonic() + 1
            while (
                harness.server.requests.stats()["active"] < 2
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            self.assertEqual(harness.server.requests.stats()["active"], 2)
            self.assertEqual(harness.request("GET", "/ready")[0], 503)
            release.set()
            self.assertEqual(first.result(3)[0], 200)
            self.assertEqual(second.result(3)[0], 200)
        self.assertEqual(calls, [1])
        self.assertEqual(harness.request("GET", "/ready")[0], 200)
        models = json.loads(harness.request("GET", "/v1/models")[2])
        self.assertEqual(models["data"][0]["max_model_len"], 128)

    def test_startup_failure_is_shared_and_latched(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def start():
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(2))
            raise RuntimeError("load failed")

        harness = self.cold_server(start=start)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                harness.request, "POST", "/v1/chat/completions", self.body()
            )
            self.assertTrue(entered.wait(1))
            second = pool.submit(
                harness.request, "POST", "/v1/chat/completions", self.body()
            )
            deadline = time.monotonic() + 1
            while (
                harness.server.requests.stats()["active"] < 2
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            self.assertEqual(harness.server.requests.stats()["active"], 2)
            release.set()
            first_status, _, first_payload = first.result(3)
            second_status, _, second_payload = second.result(3)
            self.assertEqual((first_status, second_status), (500, 500))
            self.assertEqual(first_payload, second_payload)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", self.body())[0], 500
        )
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["startup"]["state"], "failed")
        self.assertEqual(calls, [1])

    def test_cold_waiters_obey_request_capacity(self):
        entered = threading.Event()
        release = threading.Event()
        harness = self.cold_server(queue_size=1)
        app = harness.app

        def start():
            entered.set()
            self.assertTrue(release.wait(2))
            return app

        harness.server.configure_lazy(
            start, model="test-model", model_names=("test-model",)
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                harness.request, "POST", "/v1/chat/completions", self.body()
            )
            self.assertTrue(entered.wait(1))
            self.assertEqual(
                harness.request("POST", "/v1/chat/completions", self.body())[0], 503
            )
            release.set()
            self.assertEqual(first.result(3)[0], 200)

    def test_request_timeout_does_not_cancel_shared_startup(self):
        entered = threading.Event()
        release = threading.Event()
        harness = self.cold_server()
        app = harness.app
        calls = []

        def start():
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(2))
            return app

        harness.server.configure_lazy(
            start,
            model="test-model",
            model_names=("test-model",),
            request_timeout=0.05,
        )
        status, _, _ = harness.request("POST", "/v1/chat/completions", self.body())
        self.assertEqual(status, 504)
        self.assertTrue(entered.is_set())
        cold = json.loads(harness.request("GET", "/status")[2])
        self.assertEqual(cold["startup"]["state"], "loading")
        release.set()
        deadline = time.monotonic() + 2
        while harness.request("GET", "/ready")[0] != 200:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", self.body())[0], 200
        )
        self.assertEqual(calls, [1])
