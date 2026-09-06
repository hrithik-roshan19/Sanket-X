from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ml import registry
from app.realtime.broadcaster import manager
from app.realtime.events import EventType, make_event
from app.services import upload_service

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        await ws.send_json(make_event(
            EventType.CONNECTED,
            model_trained=registry.current_run_id() is not None,
            current_run_id=registry.current_run_id(),
            training_in_progress=upload_service.training_in_progress(),
        ).model_dump(mode="json"))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        await manager.disconnect(ws)
