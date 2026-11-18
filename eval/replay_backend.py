"""Evaluation-only exact-prompt replay wrapper for the resident Backend."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from model_backend import Backend


class ReplayBackend(Backend):
    """Persist successful greedy chat outputs and coalesce same-key thread calls.

    Embeddings remain uncached here. Cached responses are keyed by the exact UTF-8
    system and user strings plus model, endpoint, decoding configuration, backend
    implementation, embedding signature and clinical-engine fingerprint.
    """

    CACHE_SCHEMA = 1
    DECODING_CONFIG = {
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 768,
        "chat_template_kwargs": {"enable_thinking": False},
        "greedy": True,
    }
    USAGE_DEFAULTS = {
        "real_chat_calls": 0,
        "replayed_chat_calls": 0,
        "sqlite_cache_hits": 0,
        "singleflight_cache_hits": 0,
        "cache_misses": 0,
        "cache_write_errors": 0,
        "real_model_wait_seconds": 0.0,
        "nominal_chat_seconds": 0.0,
        "singleflight_wait_seconds": 0.0,
        "queue_wait_seconds": 0.0,
    }

    def __init__(
        self,
        cache_path: str = "results/chat_cache.sqlite3",
        engine_version: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cache_path = Path(cache_path)
        self.engine_version = self._clinical_fingerprint(engine_version)
        self._backend_fingerprint = self._file_fingerprint(
            Path(inspect.getfile(Backend))
        )
        self._flight_lock = threading.Lock()
        self._inflight: Dict[str, Future] = {}
        self._initialize_cache()
        self.reset_usage()

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return "unavailable"

    @classmethod
    def _clinical_fingerprint(cls, requested_version: Optional[str] = None) -> str:
        try:
            module = importlib.import_module("clinical")
            version = str(requested_version or getattr(module, "VERSION", "unknown"))
            module_path = Path(inspect.getfile(module))
            return version + ":" + cls._file_fingerprint(module_path)
        except (ImportError, OSError, TypeError):
            version = requested_version or os.getenv("CLINICAL_ENGINE_VERSION", "unknown")
            return str(version) + ":unavailable"

    def _request_config(self) -> Dict[str, Any]:
        return {
            "cache_schema": self.CACHE_SCHEMA,
            "engine_version": self.engine_version,
            "backend_fingerprint": self._backend_fingerprint,
            "endpoint": self.chat_url.rstrip("/") + "/chat/completions",
            "model": self.model,
            "embedding_signature": str(getattr(self, "embedding_signature", "")),
            "decoding": self.DECODING_CONFIG,
        }

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _key(self, system: str, user: str) -> Tuple[str, Dict[str, Any], str, str]:
        system_hash = self._hash_text(system)
        user_hash = self._hash_text(user)
        config = self._request_config()
        material = {
            "config": config,
            "system_sha256": system_hash,
            "user_sha256": user_hash,
        }
        payload = json.dumps(
            material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), config, system_hash, user_hash

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.cache_path), timeout=60.0)
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_cache (
                    cache_key TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    system_sha256 TEXT NOT NULL,
                    user_sha256 TEXT NOT NULL,
                    response TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    length_limited INTEGER NOT NULL,
                    nominal_chat_seconds REAL NOT NULL,
                    created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _read(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT response, prompt_tokens, completion_tokens, output_tokens,
                       length_limited, nominal_chat_seconds
                FROM chat_cache WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "response": str(row[0]),
            "prompt_tokens": int(row[1]),
            "completion_tokens": int(row[2]),
            "output_tokens": int(row[3]),
            "length_limited": int(row[4]),
            "nominal_chat_seconds": float(row[5]),
        }

    def _write(
        self,
        cache_key: str,
        config: Dict[str, Any],
        system_hash: str,
        user_hash: str,
        response: str,
        usage: Dict[str, Any],
    ) -> bool:
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO chat_cache
                    (cache_key, config_json, system_sha256, user_sha256, response,
                     prompt_tokens, completion_tokens, output_tokens, length_limited,
                     nominal_chat_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        json.dumps(config, ensure_ascii=False, sort_keys=True),
                        system_hash,
                        user_hash,
                        response,
                        int(usage["prompt_tokens"]),
                        int(usage["completion_tokens"]),
                        int(usage["output_tokens"]),
                        int(usage["length_limited"]),
                        float(usage["nominal_chat_seconds"]),
                    ),
                )
            return True
        except sqlite3.Error:
            # A cache write failure must not discard a successful model response.
            self.usage()["cache_write_errors"] += 1
            return False

    def reset_usage(self) -> None:
        super().reset_usage()
        self._local.stats.update(self.USAGE_DEFAULTS)

    def usage(self) -> Dict[str, Any]:
        stats = super().usage()
        for key, default in self.USAGE_DEFAULTS.items():
            stats.setdefault(key, default)
        return stats

    def _apply_replay(self, entry: Dict[str, Any], source: str, wait_seconds: float = 0.0) -> str:
        stats = self.usage()
        stats["chat_calls"] += 1
        stats["replayed_chat_calls"] += 1
        if source == "sqlite":
            stats["sqlite_cache_hits"] += 1
        else:
            stats["singleflight_cache_hits"] += 1
        stats["prompt_tokens"] += entry["prompt_tokens"]
        stats["completion_tokens"] += entry["completion_tokens"]
        stats["output_tokens"] += entry["output_tokens"]
        stats["length_limited"] += entry["length_limited"]
        nominal = entry["nominal_chat_seconds"]
        stats["chat_seconds"] += nominal
        stats["nominal_chat_seconds"] += nominal
        stats["singleflight_wait_seconds"] += wait_seconds
        stats["queue_wait_seconds"] += wait_seconds
        return entry["response"]

    def _call_backend(self, system: str, user: str) -> str:
        """One real model call; kept separate to allow isolated deterministic tests."""
        return super().chat(system, user)

    def chat(self, system: str, user: str) -> str:
        if not isinstance(system, str) or not isinstance(user, str):
            raise TypeError("ReplayBackend.chat requires string system and user prompts")
        cache_key, config, system_hash, user_hash = self._key(system, user)
        cached = self._read(cache_key)
        if cached is not None:
            return self._apply_replay(cached, "sqlite")

        with self._flight_lock:
            future = self._inflight.get(cache_key)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[cache_key] = future

        if not owner:
            wait_started = time.perf_counter()
            try:
                entry = future.result()
            except Exception:
                stats = self.usage()
                stats["chat_calls"] += 1
                wait_seconds = time.perf_counter() - wait_started
                stats["singleflight_wait_seconds"] += wait_seconds
                stats["queue_wait_seconds"] += wait_seconds
                raise
            wait_seconds = time.perf_counter() - wait_started
            return self._apply_replay(entry, "singleflight", wait_seconds)

        try:
            # Another process may have filled the persistent cache after the first read.
            cached = self._read(cache_key)
            if cached is not None:
                future.set_result(cached)
                return self._apply_replay(cached, "sqlite")
            stats = self.usage()
            stats["cache_misses"] += 1
            stats["real_chat_calls"] += 1
            previous_calls = int(stats["chat_calls"])
            previous_seconds = float(stats["chat_seconds"])
            previous_tokens = {
                name: int(stats[name])
                for name in ("prompt_tokens", "completion_tokens", "output_tokens", "length_limited")
            }
            started = time.perf_counter()
            try:
                response = self._call_backend(system, user)
                if not isinstance(response, str) or not response.strip():
                    raise RuntimeError("The model returned no answer content.")
            except Exception as exc:
                elapsed = time.perf_counter() - started
                stats = self.usage()
                if int(stats["chat_calls"]) == previous_calls:
                    stats["chat_calls"] += 1
                if float(stats["chat_seconds"]) == previous_seconds:
                    stats["chat_seconds"] += elapsed
                stats["real_model_wait_seconds"] += elapsed
                future.set_exception(exc)
                raise

            stats = self.usage()
            if int(stats["chat_calls"]) == previous_calls:
                stats["chat_calls"] += 1
            measured_seconds = max(0.0, float(stats["chat_seconds"]) - previous_seconds)
            elapsed = time.perf_counter() - started
            nominal = measured_seconds if measured_seconds > 0 else elapsed
            stats["real_model_wait_seconds"] += elapsed
            stats["nominal_chat_seconds"] += nominal
            entry = {
                "response": response,
                "prompt_tokens": int(stats["prompt_tokens"]) - previous_tokens["prompt_tokens"],
                "completion_tokens": int(stats["completion_tokens"]) - previous_tokens["completion_tokens"],
                "output_tokens": int(stats["output_tokens"]) - previous_tokens["output_tokens"],
                "length_limited": int(stats["length_limited"]) - previous_tokens["length_limited"],
                "nominal_chat_seconds": nominal,
            }
            self._write(cache_key, config, system_hash, user_hash, response, entry)
            future.set_result(entry)
            return response
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            with self._flight_lock:
                if self._inflight.get(cache_key) is future:
                    del self._inflight[cache_key]
