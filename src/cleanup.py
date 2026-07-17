import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class RetentionCleaner:
    def __init__(
        self,
        repository,
        snapshot_dir: str,
        log_dir: str,
        snapshots_days: int = 7,
        logs_days: int = 14,
        events_days: int = 30,
    ):
        self._repository = repository
        self._snapshot_dir = os.path.abspath(snapshot_dir)
        self._log_dir = os.path.abspath(log_dir)
        self._snapshots_days = max(0, int(snapshots_days))
        self._logs_days = max(0, int(logs_days))
        self._events_days = max(0, int(events_days))

    def run_once(self) -> dict:
        now = datetime.now()
        snapshot_count = self._delete_old_files(
            self._snapshot_dir,
            {".jpg", ".jpeg", ".png"},
            now - timedelta(days=self._snapshots_days),
        )
        log_count = self._delete_old_files(
            self._log_dir,
            {".log"},
            now - timedelta(days=self._logs_days),
        )
        event_cutoff = (now - timedelta(days=self._events_days)).isoformat(
            timespec="seconds"
        )
        deleted_events = self._repository.delete_older_than(event_cutoff)
        for snapshot_path in deleted_events["snapshot_paths"]:
            self._safe_delete(snapshot_path, self._snapshot_dir)
        result = {
            "snapshots_deleted": snapshot_count,
            "logs_deleted": log_count,
            "events_deleted": deleted_events["count"],
        }
        logger.info("数据保留清理完成: %s", result)
        return result

    @staticmethod
    def _delete_old_files(root: str, extensions: set, cutoff: datetime) -> int:
        if not os.path.isdir(root):
            return 0
        count = 0
        for path in Path(root).iterdir():
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime)
            if modified < cutoff and RetentionCleaner._safe_delete(str(path), root):
                count += 1
        return count

    @staticmethod
    def _safe_delete(path: str, root: str) -> bool:
        absolute_path = os.path.abspath(path)
        absolute_root = os.path.abspath(root)
        try:
            if os.path.commonpath([absolute_path, absolute_root]) != absolute_root:
                return False
            if os.path.isfile(absolute_path):
                os.remove(absolute_path)
                return True
        except (OSError, ValueError):
            logger.warning("清理文件失败: %s", os.path.basename(absolute_path))
        return False


class RetentionWorker:
    def __init__(self, cleaner: RetentionCleaner, interval_hours: float = 12):
        self._cleaner = cleaner
        self._interval_seconds = max(60, float(interval_hours) * 3600)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="retention-cleaner",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._cleaner.run_once()
            except Exception:
                logger.exception("执行数据保留清理失败")
            self._stop_event.wait(self._interval_seconds)

    def close(self, timeout: float = 5) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
