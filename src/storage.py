import logging
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertEventRepository:
    def __init__(self, database_path: str):
        self._database_path = os.path.abspath(database_path)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self._database_path), exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self._database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    stranger_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    left_at TEXT,
                    snapshot_path TEXT,
                    notification_status TEXT NOT NULL DEFAULT 'pending',
                    notification_error TEXT,
                    handled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alert_events_created_at "
                "ON alert_events(created_at DESC)"
            )

    def record_observation(self, event_key: str, stranger_id: str) -> None:
        now = self._now()
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute("""
                    INSERT INTO alert_events (
                        event_key, stranger_id, first_seen_at, last_seen_at,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(event_key) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        left_at = NULL
                """, (event_key, stranger_id, now, now, now))
        except sqlite3.Error as exc:
            logger.error("记录告警事件失败: %s", exc)

    def mark_departed(self, stranger_id: str) -> None:
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute("""
                    UPDATE alert_events SET left_at = ?
                    WHERE stranger_id = ? AND left_at IS NULL
                """, (self._now(), stranger_id))
        except sqlite3.Error as exc:
            logger.error("更新陌生人离开状态失败: %s", exc)

    def update_notification(
        self,
        event_key: str,
        status: str,
        error: str | None = None,
        snapshot_path: str | None = None,
    ) -> None:
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute("""
                    UPDATE alert_events
                    SET notification_status = ?, notification_error = ?,
                        snapshot_path = COALESCE(?, snapshot_path)
                    WHERE event_key = ?
                """, (status, error, snapshot_path, event_key))
        except sqlite3.Error as exc:
            logger.error("更新告警通知状态失败: %s", exc)

    def count(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM alert_events"
            ).fetchone()[0])

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")
