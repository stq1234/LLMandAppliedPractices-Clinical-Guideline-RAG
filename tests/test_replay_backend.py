"""Offline regression tests for persistent replay and per-thread singleflight."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import model_backend  # noqa: F401
except ImportError:
    stub_module = types.ModuleType("model_backend")

    class StubBackend:
        def reset_usage(self):
            self._local.stats = {
                "chat_calls": 0,
                "doc_vectors": 0,
                "query_vectors": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "output_tokens": 0,
                "chat_seconds": 0.0,
                "embedding_seconds": 0.0,
                "length_limited": 0,
            }

        def usage(self):
            if not hasattr(self._local, "stats"):
                self.reset_usage()
            return self._local.stats

        def chat(self, system, user):
            raise AssertionError("FakeReplayBackend overrides the physical call")

    stub_module.Backend = StubBackend
    sys.modules["model_backend"] = stub_module

from eval.replay_backend import ReplayBackend


class FakeReplayBackend(ReplayBackend):
    def __init__(self, cache_path, engine_version="engine-v1", delay=0.03):
        self.cache_path = Path(cache_path)
        self.engine_version = engine_version
        self.chat_url = "http://stub.invalid/v1"
        self.model = "stub-model"
        self.embedding_signature = "stub-embedding-v1"
        self._backend_fingerprint = "stub-backend-v1"
        self._local = threading.local()
        self._flight_lock = threading.Lock()
        self._inflight = {}
        self._initialize_cache()
        self.calls = 0
        self.calls_lock = threading.Lock()
        self.delay = delay
        self.fail_next = False
        self.reset_usage()

    def _call_backend(self, system, user):
        with self.calls_lock:
            self.calls += 1
            should_fail = self.fail_next
            self.fail_next = False
        time.sleep(self.delay)
        if should_fail:
            raise RuntimeError("synthetic backend failure")
        stats = self.usage()
        stats["chat_calls"] += 1
        stats["prompt_tokens"] += 17
        stats["completion_tokens"] += 5
        stats["output_tokens"] += 5
        stats["chat_seconds"] += self.delay
        return "deterministic response"

    def cache_rows(self):
        with self._connection() as connection:
            return connection.execute("SELECT COUNT(*) FROM chat_cache").fetchone()[0]


class ReplayBackendTests(unittest.TestCase):
    def test_persistent_hit_replays_output_and_logical_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "responses.sqlite3"
            cold = FakeReplayBackend(path)
            cold_result = cold.chat("system", "exact user prompt")
            cold_usage = dict(cold.usage())
            self.assertEqual(cold_result, "deterministic response")
            self.assertEqual(cold.calls, 1)
            self.assertEqual(cold_usage["chat_calls"], 1)
            self.assertEqual(cold_usage["real_chat_calls"], 1)
            self.assertEqual(cold_usage["replayed_chat_calls"], 0)
            self.assertEqual(cold_usage["prompt_tokens"], 17)
            self.assertEqual(cold.cache_rows(), 1)

            warm = FakeReplayBackend(path)
            warm_result = warm.chat("system", "exact user prompt")
            warm_usage = warm.usage()
            self.assertEqual(warm_result, cold_result)
            self.assertEqual(warm.calls, 0)
            self.assertEqual(warm_usage["chat_calls"], 1)
            self.assertEqual(warm_usage["real_chat_calls"], 0)
            self.assertEqual(warm_usage["replayed_chat_calls"], 1)
            self.assertEqual(warm_usage["prompt_tokens"], 17)
            self.assertEqual(warm_usage["completion_tokens"], 5)
            self.assertAlmostEqual(
                warm_usage["nominal_chat_seconds"], cold_usage["chat_seconds"], places=3
            )
            self.assertEqual(warm_usage["real_model_wait_seconds"], 0.0)

    def test_exact_prompt_and_model_engine_config_are_keyed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "responses.sqlite3"
            backend = FakeReplayBackend(path, delay=0)
            backend.chat("system", "user")
            backend.chat("system ", "user")
            backend.chat("system", "user ")
            backend.model = "other-model"
            backend.chat("system", "user")
            backend.model = "stub-model"
            backend.engine_version = "engine-v2"
            backend.chat("system", "user")
            backend.engine_version = "engine-v1"
            backend.chat_url = "http://another-stub.invalid/v1"
            backend.chat("system", "user")
            self.assertEqual(backend.calls, 6)
            self.assertEqual(backend.cache_rows(), 6)

    def test_exceptions_are_not_cached_and_can_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = FakeReplayBackend(Path(temp) / "responses.sqlite3", delay=0)
            backend.fail_next = True
            with self.assertRaisesRegex(RuntimeError, "synthetic backend failure"):
                backend.chat("system", "user")
            self.assertEqual(backend.cache_rows(), 0)
            self.assertEqual(backend.usage()["real_chat_calls"], 1)

            self.assertEqual(backend.chat("system", "user"), "deterministic response")
            self.assertEqual(backend.calls, 2)
            self.assertEqual(backend.cache_rows(), 1)
            self.assertEqual(backend.chat("system", "user"), "deterministic response")
            self.assertEqual(backend.calls, 2)
            self.assertEqual(backend.usage()["replayed_chat_calls"], 1)

    def test_same_process_concurrent_requests_singleflight(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = FakeReplayBackend(Path(temp) / "responses.sqlite3", delay=0.08)
            workers = 8
            barrier = threading.Barrier(workers)

            def call_once():
                backend.reset_usage()
                barrier.wait(timeout=5)
                response = backend.chat("same system", "same user")
                return response, dict(backend.usage())

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda _: call_once(), range(workers)))

            self.assertEqual({response for response, _ in results}, {"deterministic response"})
            usage = [item for _, item in results]
            self.assertEqual(backend.calls, 1)
            self.assertEqual(sum(item["chat_calls"] for item in usage), workers)
            self.assertEqual(sum(item["real_chat_calls"] for item in usage), 1)
            self.assertEqual(sum(item["replayed_chat_calls"] for item in usage), workers - 1)
            self.assertEqual(sum(item["prompt_tokens"] for item in usage), workers * 17)
            self.assertGreater(sum(item["singleflight_wait_seconds"] for item in usage), 0.0)
            self.assertEqual(backend.cache_rows(), 1)


if __name__ == "__main__":
    unittest.main()
