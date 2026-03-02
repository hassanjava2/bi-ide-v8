# BI-IDE API Contracts v1.0.0

**Status:** ✅ Canonical (Approved for Implementation)  
**Version:** 1.0.0  
**Effective Date:** 2026-03-02  
**Review Cycle:** 6 months  

---

## 1. Overview

This document defines the canonical API contracts for BI-IDE Desktop v8. All clients (Desktop, Web, Workers) MUST implement these contracts exactly as specified.

### 1.1 Contract Principles

1. **No Breaking Changes Without Version Bump**: Any breaking change requires a new API version
2. **Backward Compatibility**: Legacy routes remain for ONE release cycle only
3. **Explicit Over Implicit**: All fields, types, and behaviors are explicitly defined
4. **Contract-First Development**: No implementation without contract definition

### 1.2 Compatibility Matrix

| Client Version | API Version | Compatibility |
|----------------|-------------|---------------|
| >= 1.0.0       | v1          | ✅ Full       |
| < 1.0.0        | v1          | ❌ None       |

---

## 2. Endpoint Matrix

### 2.1 Canonical Endpoints

| Domain    | Method | Path                          | Purpose                        | Owner       |
|-----------|--------|-------------------------------|--------------------------------|-------------|
| Council   | POST   | `/api/v1/council/message`       | Send message to AI Council     | Backend     |
| Council   | GET    | `/api/v1/council/status`        | Get council system status      | Backend     |
| Council   | POST   | `/api/v1/council/discuss`       | Multi-wise-man discussion      | Backend     |
| Training  | POST   | `/api/v1/training/start`        | Start training job             | Backend     |
| Training  | GET    | `/api/v1/training/status`       | Get training status            | Backend     |
| Training  | POST   | `/api/v1/training/stop`         | Stop/pause training            | Backend     |
| Sync      | POST   | `/api/v1/sync`                  | Perform sync operation         | Rust        |
| Sync      | GET    | `/api/v1/sync/status`           | Get sync status                | Rust        |
| Sync      | WS     | `/api/v1/sync/ws`               | WebSocket sync stream          | Rust        |
| Workers   | POST   | `/api/v1/workers/register`      | Register new worker            | Agent       |
| Workers   | POST   | `/api/v1/workers/heartbeat`     | Worker heartbeat               | Agent       |
| Workers   | POST   | `/api/v1/workers/apply-policy`  | Apply resource policy          | Agent       |
| Updates   | GET    | `/api/v1/updates/manifest`      | Get update manifest            | Platform    |
| Updates   | POST   | `/api/v1/updates/report`        | Report update status           | Platform    |
| Auth      | POST   | `/api/v1/auth/token`            | Refresh access token           | Platform    |
| Auth      | POST   | `/api/v1/auth/revoke`           | Revoke device access           | Platform    |

### 2.2 Legacy Routes (Deprecated)

None - this is v1 baseline.

---

## 3. Standard Request/Response Format

### 3.1 Request Headers

```http
Content-Type: application/json
Authorization: Bearer <access_token>
X-Request-ID: <uuid>
X-Trace-ID: <hex_string>
X-Device-ID: <device_id>
X-API-Version: v1
X-Contract-Version: 1.0.0
```

### 3.2 Response Format

```json
{
  "success": true,
  "data": { /* endpoint-specific data */ },
  "error": null,
  "meta": {
    "request_id": "uuid",
    "timestamp": 1709385600000,
    "api_version": "v1",
    "contract_version": "1.0.0"
  }
}
```

### 3.3 Error Response Format

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": { /* additional context */ }
  },
  "meta": { /* same as above */ }
}
```

### 3.4 Standard Error Codes

| Code                      | HTTP Status | Description                          |
|---------------------------|-------------|--------------------------------------|
| `VERSION_MISMATCH`        | 400         | API/Contract version mismatch        |
| `ENDPOINT_DEPRECATED`     | 410         | Endpoint no longer available         |
| `INVALID_REQUEST_FORMAT`  | 400         | Request body malformed               |
| `MISSING_REQUIRED_FIELD`  | 400         | Required field not provided          |
| `INVALID_FIELD_TYPE`      | 400         | Field type incorrect                 |
| `UNAUTHORIZED`            | 401         | Authentication required              |
| `FORBIDDEN`               | 403         | Insufficient permissions             |
| `NOT_FOUND`               | 404         | Resource not found                   |
| `RATE_LIMITED`            | 429         | Too many requests                    |
| `INTERNAL_ERROR`          | 500         | Server error                         |

---

## 4. Domain-Specific Contracts

### 4.1 Council API

#### POST /api/v1/council/message

Send a message to the AI Council.

**Request:**
```json
{
  "message": "Explain this code...",
  "context": "Optional context about the project",
  "wise_man_id": null,
  "conversation_id": null,
  "request_context": {
    "request_id": "uuid",
    "trace_id": "hex",
    "span_id": "hex",
    "device_id": "device-123",
    "timestamp": 1709385600000
  }
}
```

**Response:**
```json
{
  "response": "The code does...",
  "wise_man_id": "wise-001",
  "wise_man_name": "Senior Architect",
  "confidence": 0.94,
  "conversation_id": "conv-123",
  "processing_time_ms": 450,
  "sources": [
    {
      "file_path": "src/main.rs",
      "line_start": 10,
      "line_end": 25,
      "content": "..."
    }
  ]
}
```

#### GET /api/v1/council/status

**Response:**
```json
{
  "status": "operational",
  "connected": true,
  "wise_men_count": 16,
  "active_discussions": 3,
  "messages_total": 15234,
  "last_message_at": 1709385600000,
  "latency_ms": 45
}
```

---

### 4.2 Training API

#### POST /api/v1/training/start

**Request:**
```json
{
  "job_type": "fine_tune",
  "priority": 50,
  "dataset_query": {
    "languages": ["python", "rust"],
    "file_patterns": ["*.py", "*.rs"],
    "since_timestamp": 1706784000000,
    "min_privacy_level": "anonymized"
  },
  "resource_limits": {
    "max_cpu_percent": 85,
    "max_memory_gb": 24,
    "max_gpu_memory_percent": 90,
    "max_duration_hours": 8
  },
  "request_context": { /* ... */ }
}
```

**Response:**
```json
{
  "job_id": "job-abc123",
  "status": "queued",
  "estimated_start_time": 1709389200000,
  "estimated_completion_time": 1709410800000,
  "queue_position": 2
}
```

#### GET /api/v1/training/status

**Response:**
```json
{
  "enabled": true,
  "current_job": {
    "job_id": "job-abc123",
    "job_type": "fine_tune",
    "status": "running",
    "progress_percent": 45.5,
    "started_at": 1709385600000,
    "estimated_completion": 1709410800000,
    "current_epoch": 5,
    "total_epochs": 10,
    "current_metrics": {
      "loss": 0.0234,
      "accuracy": 0.945,
      "samples_processed": 150000,
      "epoch": 5,
      "total_epochs": 10,
      "learning_rate": 0.0001
    }
  },
  "metrics": {
    "jobs_completed": 12,
    "jobs_failed": 1,
    "total_training_time_hours": 156.5,
    "last_training_at": 1709385600000
  }
}
```

#### POST /api/v1/training/stop

**Request:**
```json
{
  "job_id": "job-abc123",
  "action": "pause"
}
```

**Response:**
```json
{
  "job_id": "job-abc123",
  "previous_status": "running",
  "current_status": "paused",
  "checkpoint_saved": true,
  "checkpoint_path": "/checkpoints/job-abc123-epoch-5.ckpt"
}
```

---

### 4.3 Sync API

#### POST /api/v1/sync

**Request:**
```json
{
  "device_id": "device-123",
  "workspace_id": "ws-456",
  "since_vector_clock": {
    "clocks": {
      "1": 100,
      "2": 50
    }
  },
  "local_operations": [
    {
      "op_id": { "node_id": 1, "logical_clock": 101 },
      "file_path": "src/main.rs",
      "op_type": "update",
      "content_hash": "sha256:abc123...",
      "timestamp": 1709385600000,
      "device_id": "device-123"
    }
  ],
  "request_context": { /* ... */ }
}
```

**Response:**
```json
{
  "server_vector_clock": {
    "clocks": {
      "1": 101,
      "2": 55,
      "3": 200
    }
  },
  "operations": [
    /* Remote operations to apply */
  ],
  "conflicts": [],
  "server_timestamp": 1709385600100
}
```

---

### 4.4 Workers API

#### POST /api/v1/workers/register

**Request:**
```json
{
  "device_name": "RTX 5090 Workstation",
  "device_type": "workstation",
  "capabilities": {
    "cpu_cores": 32,
    "memory_gb": 128,
    "has_gpu": true,
    "gpu_memory_gb": 24,
    "gpu_model": "RTX 5090",
    "os": "linux",
    "arch": "x86_64",
    "supports_training": true,
    "supports_inference": true
  },
  "public_key": "-----BEGIN PUBLIC KEY-----...",
  "request_context": { /* ... */ }
}
```

**Response:**
```json
{
  "device_id": "device-xyz789",
  "access_token": "eyJhbGc...",
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2g...",
  "expires_at": 1709392800000,
  "assigned_policies": [
    {
      "policy_id": "policy-001",
      "policy_type": "resource_limits",
      "parameters": {
        "cpu_max_percent": 85,
        "ram_max_gb": 100
      }
    }
  ]
}
```

#### POST /api/v1/workers/heartbeat

**Request:**
```json
{
  "device_id": "device-xyz789",
  "status": "training",
  "resource_usage": {
    "cpu_percent": 75.5,
    "memory_percent": 60.0,
    "memory_used_gb": 76.8,
    "memory_total_gb": 128.0,
    "gpu_percent": 95.0,
    "gpu_memory_used_gb": 20.5,
    "gpu_memory_total_gb": 24.0,
    "disk_percent": 45.0
  },
  "active_jobs": ["job-abc123"],
  "request_context": { /* ... */ }
}
```

**Response:**
```json
{
  "acknowledged": true,
  "policy_updates": [
    {
      "policy_id": "policy-001",
      "action": "update",
      "parameters": {
        "cpu_max_percent": 90
      }
    }
  ],
  "command_queue": [],
  "next_heartbeat_interval_seconds": 60
}
```

#### POST /api/v1/workers/apply-policy

**Request:**
```json
{
  "device_id": "device-xyz789",
  "policy": {
    "policy_id": "policy-002",
    "mode": "training_only",
    "limits": {
      "max_cpu_percent": 90,
      "max_memory_gb": 100,
      "max_gpu_memory_percent": 95,
      "max_duration_hours": 8
    },
    "schedule": {
      "timezone": "Asia/Baghdad",
      "windows": [
        {
          "start": "22:00",
          "end": "07:00",
          "days": [0, 1, 2, 3, 4, 5, 6]
        }
      ],
      "idle_only": true
    },
    "safety": {
      "thermal_cutoff_c": 85,
      "auto_pause_on_user_activity": true,
      "max_consecutive_hours": 4,
      "required_break_minutes": 30
    }
  },
  "request_context": { /* ... */ }
}
```

**Response:**
```json
{
  "policy_id": "policy-002",
  "applied_at": 1709385600000,
  "effective_at": 1709385900000,
  "confirmation_code": "CONF-ABC123"
}
```

---

### 4.5 Updates API

#### GET /api/v1/updates/manifest

**Request:**
```json
{
  "device_id": "device-xyz789",
  "current_version": "1.0.0",
  "channel": "stable",
  "platform": "linux",
  "arch": "x86_64",
  "request_context": { /* ... */ }
}
```

**Response:**
```json
{
  "has_update": true,
  "version": "1.1.0",
  "download_url": "https://updates.bi-iq.com/v1.1.0/bi-ide-linux-x86_64.tar.gz",
  "signature_url": "https://updates.bi-iq.com/v1.1.0/bi-ide-linux-x86_64.tar.gz.sig",
  "release_notes": "Bug fixes and performance improvements...",
  "critical": false,
  "mandatory": false,
  "estimated_download_size_mb": 150.5,
  "rollout_percentage": 25.0
}
```

---

## 5. Client Implementation Guide

### 5.1 TypeScript Client Pattern

```typescript
class BiIdeClient {
  private baseUrl: string;
  private deviceId: string;
  private accessToken: string;

  async request<TRequest, TResponse>(
    endpoint: string,
    method: 'GET' | 'POST',
    body?: TRequest
  ): Promise<ApiResponse<TResponse>> {
    const requestContext = {
      request_id: crypto.randomUUID(),
      trace_id: generateTraceId(),
      device_id: this.deviceId,
      timestamp: Date.now()
    };

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.accessToken}`,
        'X-Request-ID': requestContext.request_id,
        'X-Trace-ID': requestContext.trace_id,
        'X-Device-ID': this.deviceId,
        'X-API-Version': 'v1',
        'X-Contract-Version': '1.0.0'
      },
      body: body ? JSON.stringify({ ...body, request_context: requestContext }) : undefined
    });

    return response.json();
  }
}
```

### 5.2 Rust Client Pattern

```rust
use bi_ide_protocol::contracts::v1::*;
use bi_ide_protocol::contracts::*;

pub struct BiIdeClient {
    client: reqwest::Client,
    base_url: String,
    device_id: String,
    access_token: String,
}

impl BiIdeClient {
    pub async fn council_message(
        &self,
        message: impl Into<String>
    ) -> Result<CouncilMessageResponse, ApiError> {
        let request = CouncilMessageRequest {
            message: message.into(),
            context: None,
            wise_man_id: None,
            conversation_id: None,
            request_context: RequestContext::new(&self.device_id),
        };

        self.post("/api/v1/council/message", request).await
    }

    async fn post<T: Serialize, R: DeserializeOwned>(
        &self,
        endpoint: &str,
        body: T
    ) -> Result<R, ApiError> {
        // Implementation with proper headers and error handling
    }
}
```

---

## 6. Change Control

### 6.1 Change Process

1. **Proposal**: Submit ADR (Architecture Decision Record)
2. **Review**: Code review by contract owner
3. **Approval**: Sign-off by Backend Lead + Platform Lead
4. **Implementation**: Update contracts, tests, clients
5. **Rollout**: Staged deployment with backward compatibility

### 6.2 Version History

| Version | Date       | Changes                          |
|---------|------------|----------------------------------|
| 1.0.0   | 2026-03-02 | Initial canonical release        |

---

## 7. Signatures

| Role              | Name | Signature | Date       |
|-------------------|------|-----------|------------|
| Backend Lead      | TBD  | _________ | __________ |
| Platform Lead     | TBD  | _________ | __________ |
| Rust Lead         | TBD  | _________ | __________ |
| Desktop Lead      | TBD  | _________ | __________ |

---

**END OF CONTRACT SPECIFICATION**
