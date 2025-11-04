# agent_session.py

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

ASIA_JAKARTA = timezone(timedelta(hours=7))

@dataclass
class SessionState:
    is_on: bool = False
    started_at: str = ""
    topic: Optional[str] = None
    default_collection: Optional[str] = None
    style: str = "hangat"
    mode: str = "ringkas"
    turns: int = 0

class SessionManager:
    def __init__(self):
        # key = (user_id, channel_id)
        self._sessions: Dict[Tuple[int, int], SessionState] = {}

    def _key(self, user_id: int, channel_id: int):
        return (user_id, channel_id)

    def start(self, user_id: int, channel_id: int, default_coll: Optional[str], style: str, mode: str, topic: Optional[str]=None) -> SessionState:
        k = self._key(user_id, channel_id)
        s = SessionState(
            is_on=True,
            started_at=datetime.now(ASIA_JAKARTA).isoformat(),
            topic=topic,
            default_collection=default_coll,
            style=style,
            mode=mode,
            turns=0,
        )
        self._sessions[k] = s
        return s

    def end(self, user_id: int, channel_id: int) -> Optional[SessionState]:
        k = self._key(user_id, channel_id)
        s = self._sessions.get(k)
        if s:
            s.is_on = False
        return s

    def get(self, user_id: int, channel_id: int) -> Optional[SessionState]:
        return self._sessions.get(self._key(user_id, channel_id))

    def set_topic(self, user_id: int, channel_id: int, topic: Optional[str]):
        s = self.get(user_id, channel_id)
        if s:
            s.topic = topic

    def bump_turn(self, user_id: int, channel_id: int):
        s = self.get(user_id, channel_id)
        if s:
            s.turns += 1
