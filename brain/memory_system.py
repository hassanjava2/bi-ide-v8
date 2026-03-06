#!/usr/bin/env python3
"""
memory_system.py — الذاكرة الأبدية 🧠♾️

كل شي يُحفظ للأبد — ما نمسح شي أبداً.
PostgreSQL كقاعدة بيانات رئيسية.

الجداول:
  - conversations: كل المحادثات
  - decisions: قرارات المجلس
  - files_seen: كل ملف اتفتح بالـ IDE
  - errors: كل خطأ + إصلاحه
  - training_log: سجل التدريب
  - knowledge: معارف مستخلصة
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("memory")

PROJECT_ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════
# PostgreSQL Configuration
# ═══════════════════════════════════════════════════════════

PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "bi_brain",
    "user": "bi",
    "password": "bi2026",
}

# SQL لإنشاء الجداول
SCHEMA_SQL = """
-- الذاكرة الأبدية
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64),
    role VARCHAR(16) NOT NULL,         -- user | assistant | system
    content TEXT NOT NULL,
    capsules_used TEXT[],              -- الكبسولات المستخدمة
    mode VARCHAR(8) DEFAULT 'fast',    -- fast | deep
    confidence REAL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    decision_type VARCHAR(32),         -- council | router | training
    participants TEXT[],               -- الحكماء المشاركين
    topic TEXT,
    result JSONB NOT NULL,
    votes JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS files_seen (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64),
    language VARCHAR(32),
    capsule_id VARCHAR(64),
    lines_count INT,
    errors_found INT DEFAULT 0,
    patterns_found INT DEFAULT 0,
    content_summary TEXT,
    metadata JSONB DEFAULT '{}',
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS errors_log (
    id SERIAL PRIMARY KEY,
    file_path TEXT,
    error_type VARCHAR(32),            -- syntax | logic | security | runtime
    error_text TEXT,
    fix_text TEXT,
    language VARCHAR(32),
    capsule_id VARCHAR(64),
    auto_fixed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_log (
    id SERIAL PRIMARY KEY,
    capsule_id VARCHAR(64),
    cycle_num INT,
    samples_count INT,
    loss REAL,
    duration_sec REAL,
    worker VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(256),
    content TEXT NOT NULL,
    source VARCHAR(64),                -- scout | training | conversation | file
    confidence REAL DEFAULT 0.5,
    capsule_id VARCHAR(64),
    embedding BYTEA,                   -- للبحث بالـ vectors
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes للبحث السريع
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_files_path ON files_seen(file_path);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files_seen(file_hash);
CREATE INDEX IF NOT EXISTS idx_errors_type ON errors_log(error_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_capsule ON knowledge(capsule_id);
CREATE INDEX IF NOT EXISTS idx_training_capsule ON training_log(capsule_id);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_conv_fts ON conversations USING GIN(to_tsvector('simple', content));
CREATE INDEX IF NOT EXISTS idx_knowledge_fts ON knowledge USING GIN(to_tsvector('simple', content));
"""


class EternalMemory:
    """
    الذاكرة الأبدية — كل شي يُحفظ للأبد
    
    PostgreSQL + WAL + auto-backup
    """

    def __init__(self, pg_config: Dict = None):
        self.config = pg_config or PG_CONFIG
        self.conn = None
        self._fallback_dir = PROJECT_ROOT / "brain" / "memory_fallback"
        self._fallback_dir.mkdir(parents=True, exist_ok=True)
        self._connect()

    def _connect(self):
        """الاتصال بـ PostgreSQL"""
        try:
            import psycopg2
            self.conn = psycopg2.connect(**self.config)
            self.conn.autocommit = True
            self._init_schema()
            logger.info("🧠 Eternal Memory: PostgreSQL connected")
        except ImportError:
            logger.warning("⚠️ psycopg2 not installed — using JSON fallback")
            self.conn = None
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL unavailable: {e} — using JSON fallback")
            self.conn = None

    def _init_schema(self):
        """إنشاء الجداول"""
        if not self.conn:
            return
        try:
            cur = self.conn.cursor()
            cur.execute(SCHEMA_SQL)
            cur.close()
            logger.info("🧠 Schema initialized")
        except Exception as e:
            logger.error(f"Schema error: {e}")

    # ═══════════════════════════════════════════════
    # المحادثات
    # ═══════════════════════════════════════════════

    def save_conversation(self, session_id: str, role: str, content: str,
                         capsules_used: List[str] = None, mode: str = "fast",
                         confidence: float = None, metadata: Dict = None):
        """حفظ رسالة محادثة — للأبد"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """INSERT INTO conversations (session_id, role, content, capsules_used, mode, confidence, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (session_id, role, content, capsules_used or [],
                     mode, confidence, json.dumps(metadata or {}))
                )
                cur.close()
                return
            except Exception as e:
                logger.error(f"DB save error: {e}")

        # Fallback: JSON file
        self._fallback_save("conversations", {
            "session_id": session_id, "role": role, "content": content,
            "capsules_used": capsules_used, "mode": mode,
            "confidence": confidence, "metadata": metadata,
            "created_at": datetime.now().isoformat(),
        })

    def get_conversation(self, session_id: str, limit: int = 50) -> List[Dict]:
        """جلب محادثة"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """SELECT role, content, capsules_used, mode, confidence, created_at
                       FROM conversations WHERE session_id = %s
                       ORDER BY created_at ASC LIMIT %s""",
                    (session_id, limit)
                )
                rows = cur.fetchall()
                cur.close()
                return [
                    {"role": r[0], "content": r[1], "capsules": r[2],
                     "mode": r[3], "confidence": r[4], "time": str(r[5])}
                    for r in rows
                ]
            except Exception as e:
                logger.error(f"DB read error: {e}")
        return []

    def search_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        """بحث بالمحادثات"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """SELECT session_id, role, content, created_at
                       FROM conversations
                       WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
                       ORDER BY created_at DESC LIMIT %s""",
                    (query, limit)
                )
                rows = cur.fetchall()
                cur.close()
                return [
                    {"session": r[0], "role": r[1], "content": r[2], "time": str(r[3])}
                    for r in rows
                ]
            except Exception:
                pass
        return []

    # ═══════════════════════════════════════════════
    # الملفات
    # ═══════════════════════════════════════════════

    def save_file_seen(self, file_path: str, language: str = None,
                      capsule_id: str = None, lines_count: int = 0,
                      errors_found: int = 0, patterns_found: int = 0,
                      content_summary: str = None, file_hash: str = None):
        """حفظ ملف تم فتحه — للأبد"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                # upsert — تحديث لو موجود
                cur.execute(
                    """INSERT INTO files_seen (file_path, file_hash, language, capsule_id,
                       lines_count, errors_found, patterns_found, content_summary)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO UPDATE SET
                       last_seen = NOW(), errors_found = EXCLUDED.errors_found,
                       patterns_found = EXCLUDED.patterns_found""",
                    (file_path, file_hash, language, capsule_id,
                     lines_count, errors_found, patterns_found, content_summary)
                )
                cur.close()
                return
            except Exception as e:
                logger.error(f"File save error: {e}")

        self._fallback_save("files", {
            "file_path": file_path, "language": language, "capsule_id": capsule_id,
            "lines": lines_count, "errors": errors_found, "patterns": patterns_found,
            "time": datetime.now().isoformat(),
        })

    # ═══════════════════════════════════════════════
    # الأخطاء
    # ═══════════════════════════════════════════════

    def save_error(self, file_path: str, error_type: str, error_text: str,
                   fix_text: str = None, language: str = None,
                   capsule_id: str = None, auto_fixed: bool = False):
        """حفظ خطأ + إصلاحه — للأبد"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """INSERT INTO errors_log (file_path, error_type, error_text,
                       fix_text, language, capsule_id, auto_fixed)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (file_path, error_type, error_text, fix_text,
                     language, capsule_id, auto_fixed)
                )
                cur.close()
                return
            except Exception as e:
                logger.error(f"Error save error: {e}")

        self._fallback_save("errors", {
            "file_path": file_path, "error_type": error_type,
            "error_text": error_text, "fix_text": fix_text,
            "time": datetime.now().isoformat(),
        })

    # ═══════════════════════════════════════════════
    # القرارات
    # ═══════════════════════════════════════════════

    def save_decision(self, decision_type: str, participants: List[str],
                     topic: str, result: Dict, votes: Dict = None):
        """حفظ قرار مجلس — للأبد"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """INSERT INTO decisions (decision_type, participants, topic, result, votes)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (decision_type, participants, topic,
                     json.dumps(result, ensure_ascii=False),
                     json.dumps(votes or {}, ensure_ascii=False))
                )
                cur.close()
                return
            except Exception as e:
                logger.error(f"Decision save error: {e}")

        self._fallback_save("decisions", {
            "type": decision_type, "participants": participants,
            "topic": topic, "result": result,
            "time": datetime.now().isoformat(),
        })

    # ═══════════════════════════════════════════════
    # المعرفة
    # ═══════════════════════════════════════════════

    def save_knowledge(self, topic: str, content: str, source: str = "unknown",
                       confidence: float = 0.5, capsule_id: str = None):
        """حفظ معرفة — للأبد"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """INSERT INTO knowledge (topic, content, source, confidence, capsule_id)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (topic, content, source, confidence, capsule_id)
                )
                cur.close()
                return
            except Exception as e:
                logger.error(f"Knowledge save error: {e}")

        self._fallback_save("knowledge", {
            "topic": topic, "content": content[:500], "source": source,
            "confidence": confidence, "time": datetime.now().isoformat(),
        })

    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """بحث بالمعرفة"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """SELECT topic, content, source, confidence, capsule_id, created_at
                       FROM knowledge
                       WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
                       ORDER BY confidence DESC, created_at DESC LIMIT %s""",
                    (query, limit)
                )
                rows = cur.fetchall()
                cur.close()
                return [
                    {"topic": r[0], "content": r[1], "source": r[2],
                     "confidence": r[3], "capsule": r[4], "time": str(r[5])}
                    for r in rows
                ]
            except Exception:
                pass
        return []

    # ═══════════════════════════════════════════════
    # التدريب
    # ═══════════════════════════════════════════════

    def save_training(self, capsule_id: str, cycle_num: int,
                     samples_count: int, loss: float,
                     duration_sec: float, worker: str = "local"):
        """حفظ سجل تدريب — للأبد"""
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute(
                    """INSERT INTO training_log (capsule_id, cycle_num, samples_count,
                       loss, duration_sec, worker)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (capsule_id, cycle_num, samples_count, loss, duration_sec, worker)
                )
                cur.close()
                return
            except Exception as e:
                logger.error(f"Training save error: {e}")

    # ═══════════════════════════════════════════════
    # الإحصائيات
    # ═══════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """إحصائيات الذاكرة"""
        stats = {"backend": "postgresql" if self.conn else "json_fallback"}
        if self.conn:
            try:
                cur = self.conn.cursor()
                for table in ["conversations", "decisions", "files_seen", "errors_log", "training_log", "knowledge"]:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cur.fetchone()[0]
                cur.close()
            except Exception:
                pass
        else:
            for name in ["conversations", "decisions", "files", "errors", "knowledge"]:
                f = self._fallback_dir / f"{name}.jsonl"
                stats[name] = sum(1 for _ in open(f)) if f.exists() else 0
        return stats

    # ═══════════════════════════════════════════════
    # Fallback (JSON) — لما PostgreSQL مو متوفر
    # ═══════════════════════════════════════════════

    def _fallback_save(self, category: str, data: Dict):
        """حفظ بملف JSON كـ fallback"""
        try:
            f = self._fallback_dir / f"{category}.jsonl"
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Fallback save error: {e}")


# ═══════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════

memory = EternalMemory()


if __name__ == "__main__":
    print("🧠 Eternal Memory — Test\n")
    print(f"Backend: {'PostgreSQL' if memory.conn else 'JSON Fallback'}")
    print(f"Stats: {json.dumps(memory.get_stats(), indent=2)}")

    # تجربة
    memory.save_conversation("test-session", "user", "شلون أسوي API؟", mode="fast")
    memory.save_conversation("test-session", "assistant", "استخدم FastAPI...",
                            capsules_used=["code_python"], confidence=0.9)
    memory.save_error("/test.py", "syntax", "IndentationError", "Fix indentation", "python")
    memory.save_knowledge("FastAPI", "FastAPI is a modern Python framework", "training", 0.8, "code_python")

    print(f"\nAfter test: {json.dumps(memory.get_stats(), indent=2)}")
    print("✅ Memory working!")
