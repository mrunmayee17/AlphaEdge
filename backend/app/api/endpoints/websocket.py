"""WebSocket endpoint for real-time analysis streaming."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/analysis/{analysis_id}")
async def analysis_ws(websocket: WebSocket, analysis_id: str):
    """Stream analysis updates to the frontend via WebSocket.

    Polls Redis every 500ms and sends deltas to the client.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for analysis {analysis_id}")

    redis = websocket.app.state.redis
    last_status = None
    last_keys_sent: set[str] = set()

    try:
        while True:
            data = await redis.get(f"analysis:{analysis_id}")
            if data is None:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Analysis not found or expired"},
                })
                break

            state = json.loads(data)
            current_status = state.get("status")

            # Send status update
            if current_status != last_status:
                await websocket.send_json({
                    "type": "status_update",
                    "data": {"status": current_status},
                })
                last_status = current_status

            # Send alpha prediction
            if state.get("alpha_prediction") and "alpha_prediction" not in last_keys_sent:
                await websocket.send_json({
                    "type": "alpha_prediction",
                    "data": state["alpha_prediction"],
                })
                last_keys_sent.add("alpha_prediction")

            # Send agent views as they complete
            for agent in ["quant", "fundamentals", "sentiment", "risk", "macro"]:
                view_key = f"{agent}_view"
                if state.get(view_key) and view_key not in last_keys_sent:
                    await websocket.send_json({
                        "type": "agent_view",
                        "data": {"agent": agent, "view": state[view_key]},
                    })
                    last_keys_sent.add(view_key)

                debate_key = f"{agent}_debate"
                if state.get(debate_key) and debate_key not in last_keys_sent:
                    await websocket.send_json({
                        "type": "agent_debate",
                        "data": {"agent": agent, "debate": state[debate_key]},
                    })
                    last_keys_sent.add(debate_key)

            # Send memo
            if state.get("memo") and "memo" not in last_keys_sent:
                await websocket.send_json({
                    "type": "memo",
                    "data": state["memo"],
                })
                last_keys_sent.add("memo")

            # Terminal states
            if current_status in ("complete", "error"):
                if current_status == "error":
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": state.get("error", "Unknown error")},
                    })
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for analysis {analysis_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {analysis_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass
