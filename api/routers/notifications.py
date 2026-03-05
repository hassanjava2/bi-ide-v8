"""
WebSocket Notification Router — Real-time notifications
"""

import os
import json
import logging
import asyncio
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

logger = logging.getLogger(__name__)

router = APIRouter()

# Active WebSocket connections
active_connections: Dict[str, Set[WebSocket]] = {}


@router.websocket("/ws/notifications")
async def websocket_notifications(
    websocket: WebSocket,
    user_id: str = Query(default="default"),
):
    """WebSocket endpoint for real-time notifications"""
    await websocket.accept()
    
    # Register connection
    if user_id not in active_connections:
        active_connections[user_id] = set()
    active_connections[user_id].add(websocket)
    
    logger.info(f"WebSocket connected: user={user_id}, total={sum(len(v) for v in active_connections.values())}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "مرحباً! أنت متصل بنظام الإشعارات",
            "user_id": user_id,
        })
        
        # Keep connection alive and listen for messages
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type", "")
                
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg_type == "subscribe":
                    channel = message.get("channel", "general")
                    await websocket.send_json({
                        "type": "subscribed",
                        "channel": channel,
                    })
                elif msg_type == "mark_read":
                    notification_id = message.get("notification_id")
                    await websocket.send_json({
                        "type": "marked_read",
                        "notification_id": notification_id,
                    })
                else:
                    await websocket.send_json({
                        "type": "echo",
                        "data": message,
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
                
    except WebSocketDisconnect:
        # Remove connection
        if user_id in active_connections:
            active_connections[user_id].discard(websocket)
            if not active_connections[user_id]:
                del active_connections[user_id]
        logger.info(f"WebSocket disconnected: user={user_id}")


async def broadcast_notification(user_id: str, notification: dict):
    """Send notification to all connections of a user"""
    if user_id in active_connections:
        disconnected = set()
        for ws in active_connections[user_id]:
            try:
                await ws.send_json({
                    "type": "notification",
                    "data": notification,
                })
            except Exception:
                disconnected.add(ws)
        
        # Clean up disconnected
        active_connections[user_id] -= disconnected


async def broadcast_to_all(notification: dict):
    """Send notification to all connected users"""
    for user_id in list(active_connections.keys()):
        await broadcast_notification(user_id, notification)


@router.get("/notifications/ws-stats")
async def ws_stats():
    """Get WebSocket connection statistics"""
    return {
        "total_connections": sum(len(v) for v in active_connections.values()),
        "connected_users": len(active_connections),
        "users": {uid: len(conns) for uid, conns in active_connections.items()},
    }
