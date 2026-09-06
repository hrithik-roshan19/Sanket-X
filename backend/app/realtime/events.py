from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    CONNECTED = "connected"
    UPLOAD_RECEIVED = "upload_received"
    MAPPING_PENDING = "mapping_pending"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"
    NEW_ALERT = "new_alert"


class Event(BaseModel):
    event: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict = Field(default_factory=dict)


def make_event(event: EventType, **payload: Any) -> Event:
    return Event(event=event, payload=payload)
