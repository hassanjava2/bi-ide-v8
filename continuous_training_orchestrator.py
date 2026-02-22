import argparse
import random
import time
from datetime import datetime
from typing import Any, Dict, List

import requests


DEFAULT_TOPICS = [
    "Bayesian inference for noisy production data",
    "Convex optimization for scheduling under constraints",
    "Time-series forecasting for demand spikes",
    "Discrete optimization for resource allocation",
    "Fluid simulation stability with reduced compute",
    "Probabilistic calibration for uncertainty estimates",
]


def api_get(url: str, timeout: int = 20) -> Dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(url: str, payload: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def choose_nodes(graph_payload: Dict[str, Any]) -> List[str]:
    nodes = graph_payload.get("nodes", [])
    if not nodes:
        return ["root-math"]
    leaf_ids = [node.get("node_id") for node in nodes if not node.get("children")]
    return [node_id for node_id in leaf_ids if node_id] or ["root-math"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--min-queue", type=int, default=30)
    parser.add_argument("--burst", type=int, default=10)
    parser.add_argument("--sleep-sec", type=int, default=8)
    parser.add_argument("--priority", type=int, default=8)
    args = parser.parse_args()

    api = args.api.rstrip("/")

    print(f"🚀 Orchestrator started -> {api}")

    while True:
        try:
            status_payload = api_get(f"{api}/api/v1/network/status")
            status = status_payload.get("status", {})
            queued = int(status.get("tasks", {}).get("queued", 0))

            if queued < args.min_queue:
                graph_payload = api_get(f"{api}/api/v1/network/graph")
                node_ids = choose_nodes(graph_payload)
                to_enqueue = min(args.burst, max(1, args.min_queue - queued))

                for _ in range(to_enqueue):
                    topic = random.choice(DEFAULT_TOPICS)
                    node_id = random.choice(node_ids)
                    stamped_topic = f"{topic} | tick={datetime.utcnow().isoformat()}Z"
                    api_post(
                        f"{api}/api/v1/network/training/enqueue",
                        {
                            "topic": stamped_topic,
                            "node_id": node_id,
                            "priority": args.priority,
                        },
                    )

                print(f"🧠 Enqueued {to_enqueue} tasks (queued was {queued})")
            else:
                print(f"✅ Queue healthy: {queued}")

        except Exception as error:
            print(f"⚠️ Orchestrator error: {error}")

        time.sleep(max(2, args.sleep_sec))


if __name__ == "__main__":
    main()
