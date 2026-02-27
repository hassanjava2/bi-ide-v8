"""
Training data ingestion routes.

Hostinger node use case:
- Accept desktop training samples.
- Store them locally for local training.
- Relay them to upstream Ubuntu main node.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import requests


router = APIRouter(prefix="/api/v1/training-data", tags=["training-data"])


class TrainingSample(BaseModel):
    source: str = "desktop"
    kind: str = "chat_pair"
    input_text: str = Field(min_length=1)
    output_text: Optional[str] = None
    workspace_path: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp_ms: Optional[int] = None


class TrainingIngestRequest(BaseModel):
    samples: List[TrainingSample] = Field(default_factory=list)
    relay: bool = True
    store_local: bool = True


def _base_dir() -> Path:
    path = os.getenv("TRAINING_DATA_DIR", "learning_data/training_ingest")
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _append_jsonl(path: Path, items: List[Dict[str, Any]]) -> int:
    if not items:
        return 0
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _relay_to_upstream(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    relay_enabled = os.getenv("TRAINING_RELAY_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    upstream = os.getenv("TRAINING_RELAY_UPSTREAM_URL", "").strip()

    if not relay_enabled:
        return {"attempted": False, "relayed": False, "reason": "relay_disabled"}
    if not upstream:
        return {"attempted": False, "relayed": False, "reason": "upstream_not_set"}

    base = upstream.rstrip("/")
    endpoint = f"{base}/api/v1/training-data/ingest"
    timeout_sec = float(os.getenv("TRAINING_RELAY_TIMEOUT_SEC", "8"))

    try:
        response = requests.post(
            endpoint,
            json={"samples": samples, "relay": False, "store_local": True},
            timeout=timeout_sec,
        )
        if response.status_code >= 400:
            return {
                "attempted": True,
                "relayed": False,
                "reason": f"upstream_http_{response.status_code}",
            }
        return {"attempted": True, "relayed": True}
    except Exception as exc:
        return {"attempted": True, "relayed": False, "reason": str(exc)}


@router.post("/ingest")
async def ingest_training_data(request: TrainingIngestRequest) -> Dict[str, Any]:
    if not request.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    base = _base_dir()
    samples_payload = [sample.model_dump() for sample in request.samples]

    stored = 0
    if request.store_local:
        stored = _append_jsonl(base / "samples.jsonl", samples_payload)

    relay_result = {"attempted": False, "relayed": False}
    if request.relay:
        relay_result = _relay_to_upstream(samples_payload)
        if relay_result.get("attempted") and not relay_result.get("relayed"):
            _append_jsonl(base / "relay_failed.jsonl", samples_payload)

    return {
        "status": "ok",
        "accepted": len(samples_payload),
        "stored": stored,
        "relay": relay_result,
    }


@router.get("/status")
async def training_data_status() -> Dict[str, Any]:
    base = _base_dir()
    samples_file = base / "samples.jsonl"
    relay_failed_file = base / "relay_failed.jsonl"
    return {
        "status": "ok",
        "node": os.getenv("TRAINING_NODE_NAME", "hostinger"),
        "data_dir": str(base),
        "samples_count": _line_count(samples_file),
        "relay_failed_count": _line_count(relay_failed_file),
        "relay_enabled": os.getenv("TRAINING_RELAY_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        "relay_upstream": os.getenv("TRAINING_RELAY_UPSTREAM_URL", "").strip(),
    }
