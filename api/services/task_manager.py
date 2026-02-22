"""Background task manager for long-running operations."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    progress: Any = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, repr=False)


class TaskManager:
    """Thread-safe background task manager."""

    def __init__(self):
        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a function to run in a background thread. Returns task_id."""
        tid = task_id or uuid.uuid4().hex[:12]

        with self._lock:
            if tid in self._tasks and self._tasks[tid].status == TaskStatus.RUNNING:
                return tid  # Already running

        task = BackgroundTask(task_id=tid, status=TaskStatus.RUNNING)

        def worker():
            try:
                task.result = fn(*args, _cancel_event=task.cancel_event, **kwargs)
                if task.cancel_event.is_set():
                    task.status = TaskStatus.CANCELLED
                else:
                    task.status = TaskStatus.COMPLETED
            except TypeError:
                try:
                    task.result = fn(*args, **kwargs)
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED

        t = threading.Thread(target=worker, daemon=True)
        task._thread = t

        with self._lock:
            self._tasks[tid] = task

        t.start()
        return tid

    def get(self, task_id: str) -> BackgroundTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.RUNNING:
                task.cancel_event.set()
                task.status = TaskStatus.CANCELLED
                return True
            return False

    def remove(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status != TaskStatus.RUNNING:
                self._tasks.pop(task_id, None)

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "task_id": t.task_id,
                    "status": t.status.value,
                    "error": t.error,
                    "progress": t.progress,
                }
                for t in self._tasks.values()
            ]


# Singleton instance
task_manager = TaskManager()
