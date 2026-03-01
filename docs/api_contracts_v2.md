# BI-IDE v8 API Contracts V2

> توثيق عقود API الرسمية - API Contracts Documentation
> تاريخ التحديث: 1 مارس 2026

## Overview

هذا الملف يحدد عقود API الموحدة لضمان الاتساق بين جميع مكونات النظام:
- Desktop (Tauri/Rust)
- API (Python/FastAPI)
- Hierarchy (Python)
- Workers (Python)

---

## Base URLs

| Environment | URL |
|------------|-----|
| Production | `https://bi-iq.com/api/v1` |
| Development | `http://localhost:8000/api/v1` |
| RTX 4090 | `http://192.168.1.164:8090` |

### RTX Configuration (Standardized)
```
RTX_HOST=192.168.1.164
RTX_PORT=8090
```

**All components must use these exact values.**

---

## Council API

### POST /council/message

Send a message to the AI Council and receive a response.

#### Request
```json
{
  "message": "string (required)",
  "context": {
    "session_id": "string (optional)",
    "user_id": "string (optional)",
    "previous_messages": []
  }
}
```

#### Response (Standard Schema)
```json
{
  "response": "string - The AI response text",
  "source": "string - Source of response (rtx4090, local-fallback, hierarchy)",
  "confidence": "float - 0.0 to 1.0",
  "evidence": ["array of supporting evidence"],
  "response_source": "string - same as source for backward compatibility",
  "wise_man": "string - Name of the responding council member",
  "processing_time_ms": "integer",
  "timestamp": "ISO 8601 datetime"
}
```

#### Error Response
```json
{
  "error": "string",
  "fallback_used": true,
  "response": "Fallback response if available"
}
```

### GET /council/status

Get current council status.

#### Response
```json
{
  "is_active": true,
  "members_online": 16,
  "meeting_status": "continuous",
  "president_present": false,
  "topics_discussed": 42,
  "consensus_rate": 0.75
}
```

### GET /council/members

List all council members.

#### Response
```json
{
  "members": [
    {
      "id": "S001",
      "name": "string",
      "role": "string",
      "is_active": true,
      "current_focus": "string"
    }
  ]
}
```

---

## Orchestrator API

### POST /orchestrator/auto-schedule

Automatically schedule training jobs based on priority and resource availability.

#### Request
```json
{
  "priority": "high|medium|low",
  "layer_name": "string (optional)",
  "config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

#### Response
```json
{
  "job_id": "string",
  "status": "queued|running|scheduled",
  "scheduled_at": "ISO 8601 datetime",
  "estimated_start": "ISO 8601 datetime",
  "assigned_worker": "string|null"
}
```

### GET /orchestrator/brain/status

Get Brain scheduler status.

#### Response
```json
{
  "is_active": true,
  "scheduler_status": "running|paused|idle",
  "jobs_in_queue": 5,
  "active_jobs": 2,
  "completed_jobs_today": 12,
  "average_job_duration_minutes": 45,
  "next_scheduled_job": {
    "job_id": "string",
    "scheduled_at": "ISO 8601 datetime"
  }
}
```

### Workers Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orchestrator/workers/register` | POST | Register a new worker |
| `/orchestrator/workers/heartbeat` | POST | Send heartbeat from worker |
| `/orchestrator/workers` | GET | List all workers |
| `/orchestrator/workers/{id}/command` | POST | Send command to worker |

### Jobs Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orchestrator/jobs` | POST | Create new job |
| `/orchestrator/jobs` | GET | List jobs |
| `/orchestrator/jobs/{id}` | GET | Get job details |
| `/orchestrator/jobs/{id}/claim` | POST | Claim job for execution |
| `/orchestrator/jobs/{id}/complete` | POST | Mark job as complete |
| `/orchestrator/jobs/{id}/fail` | POST | Mark job as failed |
| `/orchestrator/jobs/next` | GET | Poll for next available job |

---

## Authentication API

### POST /auth/login

Login and obtain JWT tokens.

#### Request
```json
{
  "username": "string",
  "password": "string"
}
```

#### Response
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "user": {
    "id": "integer",
    "username": "string",
    "email": "string",
    "is_active": true
  }
}
```

### POST /auth/register

Register a new user.

#### Request
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "full_name": "string (optional)"
}
```

---

## Training API

### POST /training/start

Start a new training job.

#### Request
```json
{
  "model_name": "string",
  "dataset_path": "string",
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.001,
  "device_ids": ["string"],
  "distributed": true
}
```

#### Response
```json
{
  "job_id": "string",
  "status": "running",
  "devices_assigned": ["string"],
  "started_at": "ISO 8601 datetime"
}
```

### GET /training/status

Get overall training status.

#### Response
```json
{
  "active_jobs": 2,
  "queued_jobs": 3,
  "total_devices": 5,
  "active_devices": 2,
  "avg_cluster_utilization": 0.75,
  "jobs": []
}
```

---

## Monitoring API

### GET /monitoring/metrics

Get system metrics.

#### Response
```json
{
  "fallback_rate_pct": 15.5,
  "p95_council_latency_ms": 1200,
  "training_success_ratio": 0.95,
  "council_decision_confidence": 0.82,
  "workers_online": 5,
  "timestamp": "ISO 8601 datetime"
}
```

---

## Response Schema Standards

### Success Response
```json
{
  "status": "ok",
  "data": {},
  "timestamp": "ISO 8601 datetime"
}
```

### Error Response
```json
{
  "status": "error",
  "error": "string",
  "error_code": "string",
  "timestamp": "ISO 8601 datetime"
}
```

### Pagination
```json
{
  "data": [],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 100,
    "total_pages": 2
  }
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or expired token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Invalid request data |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `RTX_UNAVAILABLE` | 503 | RTX 4090 server not reachable |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-03-01 | Initial V2 contracts with standardized schemas |

---

## Compliance

All API implementations must:

1. Follow the exact response schemas defined above
2. Use standardized RTX_HOST/RTX_PORT values
3. Return consistent error formats
4. Include timestamps in ISO 8601 format
5. Support Arabic and English where applicable
6. Include source attribution for all AI responses
