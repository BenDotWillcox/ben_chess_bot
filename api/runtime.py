"""Bounded runtime primitives used by the BenBot API.

The module deliberately has no FastAPI or model dependencies so the resource
controls can be unit-tested without loading Maia2 or starting Stockfish.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


class WorkQueueFull(RuntimeError):
    """Raised when bounded background work cannot accept another request."""


class WorkTimedOut(TimeoutError):
    """Raised when accepted background work exceeds its caller timeout."""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after: float


@dataclass
class _SessionEntry(Generic[T]):
    value: T
    created_at: float
    last_accessed_at: float


class SessionStore(Generic[T]):
    """Thread-safe, TTL-bound least-recently-used in-memory session store."""

    def __init__(
        self,
        *,
        max_sessions: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, _SessionEntry[T]] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired_locked(self, now: float) -> list[str]:
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if now - entry.last_accessed_at >= self.ttl_seconds
        ]
        for session_id in expired:
            self._entries.pop(session_id, None)
        return expired

    def put(self, session_id: str, value: T) -> list[str]:
        """Insert a session and return IDs removed by expiry or LRU eviction."""
        now = self._clock()
        with self._lock:
            removed = self._purge_expired_locked(now)
            if session_id in self._entries:
                self._entries.pop(session_id)
            while len(self._entries) >= self.max_sessions:
                evicted_id, _ = self._entries.popitem(last=False)
                removed.append(evicted_id)
            self._entries[session_id] = _SessionEntry(value, now, now)
            return removed

    def get(self, session_id: str, *, touch: bool = True) -> T | None:
        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            if touch:
                entry.last_accessed_at = now
                self._entries.move_to_end(session_id)
            return entry.value

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._entries.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, float | int]:
        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            return {
                "active": len(self._entries),
                "capacity": self.max_sessions,
                "ttl_seconds": self.ttl_seconds,
            }


class SlidingWindowRateLimiter:
    """Bounded per-key sliding-window rate limiter."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 0:
            raise ValueError("max_requests cannot be negative")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_keys < 1:
            raise ValueError("max_keys must be at least 1")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> RateLimitDecision:
        if self.max_requests == 0:
            return RateLimitDecision(True, 0, 0.0)

        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._requests.get(key)
            if timestamps is None:
                while len(self._requests) >= self.max_keys:
                    self._requests.popitem(last=False)
                timestamps = deque()
                self._requests[key] = timestamps
            else:
                self._requests.move_to_end(key)

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = max(0.001, self.window_seconds - (now - timestamps[0]))
                return RateLimitDecision(False, 0, retry_after)

            timestamps.append(now)
            return RateLimitDecision(True, self.max_requests - len(timestamps), 0.0)


class BoundedExecutor:
    """Thread pool with explicit queue capacity, admission wait, and timeouts.

    Timed-out running work keeps its slot until it really finishes. This is
    important for model inference: a caller timeout must not allow an unlimited
    number of abandoned model calls to accumulate in the background.
    """

    def __init__(self, *, max_workers: int, max_queue: int, name: str) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        self.max_workers = max_workers
        self.max_queue = max_queue
        self.capacity = max_workers + max_queue
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._active = 0
        self._active_lock = threading.Lock()
        self._closed = False
        self._lifecycle_lock = threading.Lock()

    def _finished(self, _future: Future[object]) -> None:
        with self._active_lock:
            self._active -= 1
        self._slots.release()

    def run(
        self,
        fn: Callable[[], R],
        *,
        timeout_seconds: float,
        admission_timeout_seconds: float,
    ) -> R:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if admission_timeout_seconds < 0:
            raise ValueError("admission_timeout_seconds cannot be negative")
        with self._lifecycle_lock:
            if self._closed:
                raise WorkQueueFull("executor is shut down")
        if not self._slots.acquire(timeout=admission_timeout_seconds):
            raise WorkQueueFull("work queue is full")

        try:
            with self._lifecycle_lock:
                if self._closed:
                    raise WorkQueueFull("executor is shut down")
                future = self._executor.submit(fn)
        except BaseException:
            self._slots.release()
            raise

        with self._active_lock:
            self._active += 1
        future.add_done_callback(self._finished)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            # Cancel work that has not started. Running work cannot be killed,
            # and intentionally retains its slot until the callback fires.
            future.cancel()
            raise WorkTimedOut(f"work exceeded {timeout_seconds:g}s timeout") from exc

    def stats(self) -> dict[str, int]:
        with self._active_lock:
            active = self._active
        return {
            "active": active,
            "capacity": self.capacity,
            "workers": self.max_workers,
            "queue_capacity": self.max_queue,
        }

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._closed


class JsonLogFormatter(logging.Formatter):
    """One-line JSON formatter for service and request events."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_json_logger(name: str = "benbot") -> logging.Logger:
    logger = logging.getLogger(name)
    level_name = __import__("os").getenv("CHESS_BOT_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: object) -> None:
    logger.log(level, event, extra={"event": event, "fields": fields})


def process_memory_snapshot() -> dict[str, float | str | None]:
    """Return current process RSS using only the Python standard library."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            ok = get_process_memory_info(handle, ctypes.byref(counters), counters.cb)
            if not ok:
                raise OSError("GetProcessMemoryInfo failed")
            return {
                "rss_mb": round(counters.WorkingSetSize / (1024 * 1024), 2),
                "peak_rss_mb": round(counters.PeakWorkingSetSize / (1024 * 1024), 2),
                "source": "working_set",
                "scope": "api_process_only_excludes_stockfish_child",
            }

        status_path = "/proc/self/status"
        if os.path.exists(status_path):
            values: dict[str, float] = {}
            with open(status_path, encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        name, value, _unit = line.split()
                        values[name.rstrip(":")] = float(value) / 1024
            return {
                "rss_mb": round(values.get("VmRSS", 0.0), 2),
                "peak_rss_mb": round(values.get("VmHWM", values.get("VmRSS", 0.0)), 2),
                "source": "proc_status",
                "scope": "api_process_only_excludes_stockfish_child",
            }
    except (OSError, ValueError, AttributeError):
        pass
    return {
        "rss_mb": None,
        "peak_rss_mb": None,
        "source": "unavailable",
        "scope": "api_process_only_excludes_stockfish_child",
    }
