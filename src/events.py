import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StrangerEvent:
    stranger_id: str
    event_number: int
    first_seen: float
    last_seen: float
    confirmed: bool = False
    active: bool = True

    @property
    def event_id(self) -> str:
        return f"{self.stranger_id}-event-{self.event_number}"


class StrangerEventManager:
    """维护陌生人的进入、停留、离开和再次出现状态。"""

    def __init__(
        self,
        leave_timeout_seconds: float = 30,
        clock=time.monotonic,
        session_id: str | None = None,
    ):
        self._leave_timeout_seconds = max(0.1, float(leave_timeout_seconds))
        self._clock = clock
        self._session_id = session_id or uuid.uuid4().hex[:10]
        self._events: Dict[str, StrangerEvent] = {}
        self._event_counts: Dict[str, int] = {}

    def observe(self, stranger_id: str, confirmed: bool) -> Optional[str]:
        now = self._clock()
        event = self._events.get(stranger_id)

        if event and event.active and now - event.last_seen > self._leave_timeout_seconds:
            event.active = False

        if event is None or not event.active:
            event_number = self._event_counts.get(stranger_id, 0) + 1
            self._event_counts[stranger_id] = event_number
            event = StrangerEvent(
                stranger_id=stranger_id,
                event_number=event_number,
                first_seen=now,
                last_seen=now,
            )
            self._events[stranger_id] = event
        else:
            event.last_seen = now

        if confirmed:
            event.confirmed = True
            return f"{self._session_id}-{event.event_id}"
        return None

    def mark_departures(self) -> List[str]:
        now = self._clock()
        departed = []
        for stranger_id, event in self._events.items():
            if event.active and now - event.last_seen > self._leave_timeout_seconds:
                event.active = False
                departed.append(stranger_id)
        return departed

    def get_event(self, stranger_id: str) -> Optional[StrangerEvent]:
        return self._events.get(stranger_id)
