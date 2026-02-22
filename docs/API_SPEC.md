# BI-IDE API Specification
# مواصفات واجهة برمجة التطبيقات BI-IDE

---

## Base URL / الرابط الأساسي

```
Development: http://localhost:8000/api/v1
Production:  https://api.bi-ide.com/api/v1
RTX 4090:    http://192.168.68.111:9090
```

---

## Authentication / المصادقة

### API Key Authentication
```http
GET /api/v1/status
Authorization: Bearer {api_key}
```

### Response Codes
| Code | Meaning | Arabic |
|------|---------|--------|
| 200 | OK | نجاح |
| 201 | Created | تم الإنشاء |
| 400 | Bad Request | طلب غير صالح |
| 401 | Unauthorized | غير مصرح |
| 403 | Forbidden | ممنوع |
| 404 | Not Found | غير موجود |
| 500 | Server Error | خطأ في الخادم |

---

## System Endpoints / نقاط النهاية للنظام

### GET /status
Get complete system status

**Request:**
```http
GET /api/v1/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "services": {
    "ai_hierarchy": true,
    "ide": {
      "files_count": 150,
      "copilot_ready": true
    },
    "erp": {
      "accounting": {},
      "inventory": {},
      "hr": {}
    }
  },
  "hierarchy": {
    "council": {
      "wise_men_count": 16,
      "status": "active"
    },
    "scouts": {
      "intel_buffer_size": 42,
      "high_priority_queue": 0
    },
    "meta": {
      "performance_score": 85,
      "managers_active": 16
    },
    "experts": {
      "total": 11,
      "domains": ["tech", "business", "science"]
    },
    "execution": {
      "active_crises": 0,
      "pending_tasks": 3,
      "completed_tasks": 42
    }
  },
  "guardian": {
    "active": true,
    "layers": 5,
    "active_crises": 0,
    "total_requests": 1250,
    "threats_blocked": 0,
    "violations_prevented": 0,
    "current_mode": "ACTIVE"
  },
  "rtx4090": {
    "connected": true,
    "models_loaded": 15,
    "gpu_utilization": "45%"
  },
  "layers": 15,
  "version": "3.0.0",
  "timestamp": "2026-02-20T10:30:00Z"
}
```

---

### GET /hierarchy/status
Get detailed AI hierarchy status

**Response:**
```json
{
  "layers": [
    {
      "layer": 1,
      "name": "Eternity Layer",
      "status": "active",
      "backup_count": 3
    },
    {
      "layer": 14,
      "name": "High Council",
      "status": "active",
      "wise_men_active": 16
    }
  ],
  "overall_health": "healthy",
  "last_check": "2026-02-20T10:30:00Z"
}
```

---

### POST /command
Execute command through AI hierarchy

**Request:**
```json
{
  "command": "تحليل المشروع الحالي",
  "alert_level": "GREEN",
  "context": {
    "project_id": "proj_123",
    "user_id": "user_456"
  }
}
```

**Alert Levels:**
- `GREEN` - وضع طبيعي
- `YELLOW` - تنبيه خفيف
- `ORANGE` - حالة حرجة
- `RED` - أزمة
- `BLACK` - طوارئ قصوى

**Response:**
```json
{
  "success": true,
  "result": {
    "action": "analysis",
    "output": "تحليل المشروع...",
    "recommendations": ["rec1", "rec2"]
  },
  "layers_involved": [4, 11, 14],
  "processing_time": 1.25
}
```

---

### GET /wisdom
Get wisdom from the seventh dimension

**Query Parameters:**
- `horizon` (optional): "day", "week", "month", "year", "century" (default: "century")

**Response:**
```json
{
  "wisdom": "التأسيس المتين يحتاج صبراً ورؤية طويلة المدى.",
  "horizon": "century",
  "layer": 15,
  "timestamp": "2026-02-20T10:30:00Z"
}
```

---

### GET /guardian/status
Get guardian layer security status

**Response:**
```json
{
  "active": true,
  "layers": 5,
  "active_crises": 0,
  "total_requests": 1250,
  "threats_blocked": 0,
  "violations_prevented": 0,
  "current_mode": "ACTIVE",
  "security_level": "normal",
  "compliance_status": "compliant",
  "bridge_connections": 3,
  "eternity_backup": true,
  "sub_layers": [
    {"layer": 1, "name": "Validation", "status": "active"},
    {"layer": 2, "name": "Authorization", "status": "active"},
    {"layer": 3, "name": "Rate Limiting", "status": "active"},
    {"layer": 4, "name": "Encryption", "status": "active"},
    {"layer": 5, "name": "Audit", "status": "active"}
  ]
}
```

---

## IDE Endpoints / نقاط النهاية لـ IDE

### GET /ide/files
Get file tree

**Response:**
```json
{
  "id": "root",
  "name": "project",
  "type": "folder",
  "path": "/",
  "language": null,
  "children": [
    {
      "id": "file_1",
      "name": "main.py",
      "type": "file",
      "path": "/main.py",
      "language": "python",
      "children": []
    }
  ]
}
```

---

### GET /ide/files/{file_id}
Get file content

**Response:**
```json
{
  "id": "file_1",
  "content": "print('Hello World')",
  "language": "python",
  "last_modified": "2026-02-20T10:30:00Z"
}
```

---

### POST /ide/files/{file_id}
Save file content

**Request:**
```json
{
  "content": "print('Hello World')"
}
```

**Response:**
```json
{
  "success": true,
  "file_id": "file_1",
  "timestamp": "2026-02-20T10:30:00Z"
}
```

---

### POST /ide/copilot/suggest
Get code suggestions

**Request:**
```json
{
  "code": "def calculate_sum(a, b):\n    ",
  "cursor_position": 26,
  "language": "python",
  "file_path": "/main.py"
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "label": "return a + b",
      "detail": "Add two numbers",
      "insertText": "return a + b",
      "confidence": 0.95
    }
  ],
  "model_used": "copilot_transformer",
  "inference_time": 0.3
}
```

---

### POST /ide/terminal/execute
Execute terminal command

**Request:**
```json
{
  "session_id": "term_123",
  "command": "ls -la"
}
```

**Response:**
```json
{
  "session_id": "term_123",
  "output": "total 32\ndrwxr-xr-x  5 user user 4096 Feb 20 10:30 .",
  "exit_code": 0,
  "execution_time": 0.1
}
```

---

## ERP Endpoints / نقاط النهاية لـ ERP

### GET /erp/dashboard
Get ERP dashboard data

**Response:**
```json
{
  "accounting": {
    "total_revenue": 150000,
    "total_expenses": 75000,
    "net_profit": 75000,
    "outstanding_invoices": 12
  },
  "inventory": {
    "total_items": 1500,
    "low_stock_items": 15,
    "total_value": 250000
  },
  "hr": {
    "total_employees": 45,
    "active_projects": 8,
    "monthly_payroll": 120000
  },
  "recent_transactions": [
    {
      "id": "tx_1",
      "type": "invoice",
      "amount": 5000,
      "date": "2026-02-20T10:30:00Z"
    }
  ]
}
```

---

### GET /erp/invoices
List all invoices

**Query Parameters:**
- `status` (optional): "pending", "paid", "overdue"
- `page` (optional): page number
- `limit` (optional): items per page

**Response:**
```json
{
  "invoices": [
    {
      "id": "inv_123",
      "customer_name": "شركة ABC",
      "customer_id": "cust_456",
      "amount": 5000,
      "tax": 250,
      "total": 5250,
      "status": "pending",
      "created_at": "2026-02-20T10:30:00Z",
      "items": [
        {"name": "Product A", "quantity": 10, "price": 500}
      ]
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

---

### POST /erp/invoices
Create new invoice

**Request:**
```json
{
  "customer_name": "شركة ABC",
  "customer_id": "cust_456",
  "amount": 5000,
  "tax": 250,
  "total": 5250,
  "items": [
    {"name": "Product A", "quantity": 10, "price": 500}
  ],
  "notes": "فاتورة خدمات شهر فبراير"
}
```

**Response:**
```json
{
  "id": "inv_124",
  "status": "created",
  "created_at": "2026-02-20T10:30:00Z"
}
```

---

## Council Endpoints / نقاط النهاية للمجلس

### POST /council/message
Send message to the council

**Request:**
```json
{
  "message": "ما رأيك في توسيع المشروع؟",
  "user_id": "president",
  "alert_level": "GREEN",
  "context": {
    "project_id": "proj_123",
    "domain": "business"
  }
}
```

**Response:**
```json
{
  "response": "سيادة الرئيس، توسيع المشروع يتطلب دراسة دقيقة للجوانب المالية والسوقية...",
  "council_member": "حكيم الاستراتيجية",
  "member_id": 13,
  "confidence": 0.92,
  "alert_level": "GREEN",
  "from_rtx4090": true,
  "inference_time": 0.85,
  "timestamp": "2026-02-20T10:30:00Z"
}
```

---

### POST /council/discussion
Start group discussion

**Request:**
```json
{
  "message": "نحتاج لتحليل شامل للمشروع",
  "participants": 5,
  "user_id": "president",
  "alert_level": "YELLOW",
  "mode": "sequential"
}
```

**Modes:**
- `sequential` - الردود متتالية
- `parallel` - الردود متوازية
- `debate` - نقاش/جدال

**Response:**
```json
{
  "discussion_id": "disc_123",
  "responses": [
    {
      "council_member": "حكيم الاستراتيجية",
      "response": "...",
      "order": 1
    },
    {
      "council_member": "حكيم المخاطر",
      "response": "...",
      "order": 2
    }
  ],
  "summary": "الآراء متقاربة حول الحاجة للتحليل...",
  "from_rtx4090": true,
  "total_time": 4.5
}
```

---

### GET /council/history
Get chat history

**Query Parameters:**
- `user_id` (required): user identifier
- `limit` (optional): number of messages (default: 50)
- `before` (optional): timestamp for pagination

**Response:**
```json
{
  "messages": [
    {
      "id": "msg_1",
      "user_message": "...",
      "council_response": "...",
      "council_member": "...",
      "alert_level": "GREEN",
      "timestamp": "2026-02-20T10:30:00Z"
    }
  ],
  "has_more": true,
  "total_count": 150
}
```

---

### GET /council/members
Get council members info

**Response:**
```json
{
  "members": [
    {
      "id": 1,
      "name": "حكيم القرار",
      "role": "Decision Maker",
      "specialty": "القيادة والقرارات",
      "status": "active",
      "model_loaded": true,
      "response_count": 1250
    },
    {
      "id": 2,
      "name": "حكيم البصيرة",
      "role": "Visionary",
      "specialty": "التحليل العميق",
      "status": "active",
      "model_loaded": true,
      "response_count": 980
    }
  ],
  "total_members": 16,
  "active_members": 16
}
```

---

## RTX 4090 Endpoints / نقاط النهاية لـ RTX 4090

### GET /rtx4090/health
Check RTX 4090 server health

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory": {
    "total": 24,
    "used": 8.5,
    "free": 15.5
  },
  "models_loaded": 15,
  "uptime": 86400,
  "timestamp": "2026-02-20T10:30:00Z"
}
```

---

### POST /rtx4090/generate
Generate text using AI models

**Request:**
```json
{
  "prompt": "اكتب نصاً عن",
  "model": "layer_0",
  "max_length": 100,
  "temperature": 0.8,
  "top_p": 0.95,
  "repetition_penalty": 1.2
}
```

**Models:**
- `layer_0` through `layer_14` - Individual models
- `auto` - Auto-select based on context
- `ensemble` - Combine multiple models

**Response:**
```json
{
  "generated_text": "الذكاء الاصطناعي هو...",
  "model_used": "layer_0",
  "input_tokens": 10,
  "output_tokens": 50,
  "inference_time": 0.75,
  "timestamp": "2026-02-20T10:30:00Z"
}
```

---

### POST /rtx4090/council/message
Direct council message to RTX 4090

**Request:**
```json
{
  "message": "...",
  "user_id": "president",
  "alert_level": "GREEN",
  "preferred_member": null
}
```

**Response:**
```json
{
  "response": "...",
  "council_member": "...",
  "member_id": 5,
  "model_confidence": 0.92,
  "processing_time": 1.2
}
```

---

### GET /rtx4090/training/status
Get training status

**Response:**
```json
{
  "training_active": true,
  "current_epoch": 62000,
  "total_epochs": 100000,
  "accuracy": 0.94,
  "loss": 0.12,
  "last_checkpoint": "layer_0_epoch_62000.pt",
  "checkpoint_time": "2026-02-20T10:00:00Z",
  "estimated_completion": "2026-03-15T00:00:00Z"
}
```

---

### GET /rtx4090/models
List loaded models

**Response:**
```json
{
  "models": [
    {
      "id": "layer_0",
      "name": "Wise Decision Maker",
      "vocab_size": 126,
      "parameters": "95M",
      "status": "loaded",
      "last_used": "2026-02-20T10:30:00Z"
    }
  ],
  "total_models": 15,
  "loaded_models": 15,
  "memory_usage": "1.4 GB"
}
```

---

## WebSocket Endpoints / نقاط النهاية WebSocket

### WS /ws/council/stream
Real-time council streaming

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/council/stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'start_discussion',
    message: '...',
    alert_level: 'GREEN'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle streaming response
};
```

**Stream Events:**
```json
// Member speaking
{
  "type": "member_start",
  "member": "حكيم الاستراتيجية",
  "timestamp": "2026-02-20T10:30:00Z"
}

// Token generated
{
  "type": "token",
  "content": "ال",
  "member": "حكيم الاستراتيجية"
}

// Member finished
{
  "type": "member_end",
  "member": "حكيم الاستراتيجية",
  "full_response": "الاستراتيجية تتطلب..."
}

// Discussion complete
{
  "type": "complete",
  "summary": "..."
}
```

---

## Error Responses / ردود الأخطاء

### Standard Error Format
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is invalid",
    "details": {
      "field": "alert_level",
      "issue": "Invalid value"
    },
    "timestamp": "2026-02-20T10:30:00Z",
    "request_id": "req_123"
  }
}
```

### Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_REQUEST | 400 | Request format is invalid |
| UNAUTHORIZED | 401 | Authentication required |
| FORBIDDEN | 403 | Access denied |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMITED | 429 | Too many requests |
| RTX4090_UNAVAILABLE | 503 | GPU server down |
| MODEL_ERROR | 500 | AI model error |
| VALIDATION_ERROR | 400 | Input validation failed |

---

## Rate Limiting / تقييد الطلبات

### Limits
| Endpoint | Rate Limit |
|----------|------------|
| /status | 100/minute |
| /council/message | 30/minute |
| /council/discussion | 10/minute |
| /rtx4090/generate | 60/minute |
| /ide/copilot/suggest | 120/minute |

### Rate Limit Headers
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1708425600
```

---

## Pagination / التصفح

### Request Parameters
```
GET /api/v1/erp/invoices?page=2&limit=20
```

### Response Format
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 20,
    "total": 150,
    "pages": 8,
    "has_next": true,
    "has_prev": true
  }
}
```

---

## Versioning / إصدار API

### Current Version
```
/api/v1/...
```

### Version Header
```http
Accept-Version: v1
```

### Deprecation
Deprecated endpoints return:
```http
Deprecation: true
Sunset: Sat, 01 Jun 2026 00:00:00 GMT
```

---

## SDK Examples / أمثلة SDK

### Python
```python
import requests

class BIIDEClient:
    def __init__(self, api_key, base_url="http://localhost:8000/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def send_to_council(self, message, alert_level="GREEN"):
        response = requests.post(
            f"{self.base_url}/council/message",
            headers=self.headers,
            json={
                "message": message,
                "alert_level": alert_level
            }
        )
        return response.json()

# Usage
client = BIIDEClient("your_api_key")
result = client.send_to_council("ما رأيك في المشروع؟")
print(result["response"])
```

### JavaScript
```javascript
class BIIDEClient {
  constructor(apiKey, baseUrl = 'http://localhost:8000/api/v1') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async sendToCouncil(message, alertLevel = 'GREEN') {
    const response = await fetch(`${this.baseUrl}/council/message`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        alert_level: alertLevel
      })
    });
    return response.json();
  }
}

// Usage
const client = new BIIDEClient('your_api_key');
const result = await client.sendToCouncil('ما رأيك في المشروع؟');
console.log(result.response);
```

---

*Document Version: 1.0*
*Last Updated: 2026-02-20*
*API Version: v1*
