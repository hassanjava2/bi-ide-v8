# Training Relay Setup (Desktop → Hostinger → Ubuntu)

This setup enables:
- Desktop app sends chat-derived training pairs.
- Hostinger stores local samples for its own training.
- Hostinger relays the same samples to Ubuntu main trainer.

## API Endpoints

- `POST /api/v1/training-data/ingest`
- `GET /api/v1/training-data/status`

## Payload

```json
{
  "samples": [
    {
      "source": "desktop-tauri",
      "kind": "ai_chat_pair",
      "input_text": "user prompt",
      "output_text": "assistant answer",
      "workspace_path": "/path/to/workspace",
      "file_path": "/path/to/file",
      "metadata": {},
      "timestamp_ms": 1730000000000
    }
  ],
  "relay": true,
  "store_local": true
}
```

## Hostinger (.env)

```env
TRAINING_NODE_NAME=hostinger
TRAINING_DATA_DIR=learning_data/training_ingest
TRAINING_RELAY_ENABLED=true
TRAINING_RELAY_UPSTREAM_URL=http://<ubuntu-ip>:8000
TRAINING_RELAY_TIMEOUT_SEC=8
```

## Ubuntu main trainer (.env)

```env
TRAINING_NODE_NAME=ubuntu-main
TRAINING_DATA_DIR=learning_data/training_ingest
TRAINING_RELAY_ENABLED=false
```

Ubuntu should run the same API service so it can receive `POST /api/v1/training-data/ingest`.

## Quick Check

```bash
curl http://127.0.0.1:8000/api/v1/training-data/status
```

On Hostinger, `relay_failed_count > 0` means retries are needed (samples are preserved in `relay_failed.jsonl`).
