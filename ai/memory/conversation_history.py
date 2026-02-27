"""
Conversation History Storage
SQLite/PostgreSQL based conversation storage with CRUD operations
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    POSTGRES_AVAILABLE = False


@dataclass
class Conversation:
    """Conversation record."""
    id: Optional[int] = None
    user_id: str = ""
    session_id: str = ""
    topic: str = ""
    messages: List[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Message:
    """Individual message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class ConversationStore:
    """
    Store for conversation history with SQLite/PostgreSQL backend.
    """
    
    def __init__(
        self,
        db_type: str = 'sqlite',
        db_path: str = 'data/conversations.db',
        pg_connection_string: Optional[str] = None
    ):
        """
        Initialize conversation store.
        
        Args:
            db_type: 'sqlite' or 'postgresql'
            db_path: Path to SQLite database
            pg_connection_string: PostgreSQL connection string
        """
        self.db_type = db_type
        self.db_path = db_path
        self.pg_connection_string = pg_connection_string
        self._local = threading.local()
        
        self._init_db()
    
    def _get_connection(self):
        """Get database connection (thread-safe)."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            if self.db_type == 'sqlite':
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._local.connection.row_factory = sqlite3.Row
            elif self.db_type == 'postgresql':
                if not self.pg_connection_string:
                    raise ValueError("PostgreSQL connection string required")
                self._local.connection = psycopg2.connect(self.pg_connection_string)
        
        return self._local.connection
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.db_type == 'sqlite':
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    topic TEXT,
                    messages TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at)
            ''')
            
        elif self.db_type == 'postgresql':
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    topic VARCHAR(500),
                    messages JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_topic ON conversations USING gin(to_tsvector('english', topic))
            ''')
        
        conn.commit()
        cursor.close()
    
    def create_conversation(
        self,
        user_id: str,
        session_id: str,
        topic: str = "",
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Create a new conversation.
        
        Returns:
            Conversation ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now()
        messages_json = json.dumps([])
        metadata_json = json.dumps(metadata or {})
        
        if self.db_type == 'sqlite':
            cursor.execute('''
                INSERT INTO conversations 
                (user_id, session_id, topic, messages, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, session_id, topic, messages_json, now, now, metadata_json))
            conversation_id = cursor.lastrowid
        else:
            cursor.execute('''
                INSERT INTO conversations 
                (user_id, session_id, topic, messages, created_at, updated_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (user_id, session_id, topic, messages_json, now, now, metadata_json))
            conversation_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        
        return conversation_id
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get conversation by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.db_type == 'sqlite':
            cursor.execute('''
                SELECT * FROM conversations WHERE id = ?
            ''', (conversation_id,))
        else:
            cursor.execute('''
                SELECT * FROM conversations WHERE id = %s
            ''', (conversation_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            return self._row_to_conversation(row)
        return None
    
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add message to conversation.
        
        Returns:
            True if successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get existing messages
        if self.db_type == 'sqlite':
            cursor.execute('SELECT messages FROM conversations WHERE id = ?', (conversation_id,))
        else:
            cursor.execute('SELECT messages FROM conversations WHERE id = %s', (conversation_id,))
        
        row = cursor.fetchone()
        if not row:
            cursor.close()
            return False
        
        messages = json.loads(row[0])
        
        # Add new message
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        messages.append(message)
        
        # Update conversation
        messages_json = json.dumps(messages)
        now = datetime.now()
        
        if self.db_type == 'sqlite':
            cursor.execute('''
                UPDATE conversations 
                SET messages = ?, updated_at = ?
                WHERE id = ?
            ''', (messages_json, now, conversation_id))
        else:
            cursor.execute('''
                UPDATE conversations 
                SET messages = %s, updated_at = %s
                WHERE id = %s
            ''', (messages_json, now, conversation_id))
        
        conn.commit()
        cursor.close()
        
        return True
    
    def update_conversation(
        self,
        conversation_id: int,
        topic: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update conversation fields."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if topic is not None:
            updates.append(f"topic = {self._placeholder()}")
            params.append(topic)
        
        if metadata is not None:
            updates.append(f"metadata = {self._placeholder()}")
            params.append(json.dumps(metadata))
        
        if not updates:
            cursor.close()
            return False
        
        updates.append(f"updated_at = {self._placeholder()}")
        params.append(datetime.now())
        params.append(conversation_id)
        
        query = f"UPDATE conversations SET {', '.join(updates)} WHERE id = {self._placeholder()}"
        cursor.execute(query, params)
        
        conn.commit()
        success = cursor.rowcount > 0
        cursor.close()
        
        return success
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete conversation by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.db_type == 'sqlite':
            cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        else:
            cursor.execute('DELETE FROM conversations WHERE id = %s', (conversation_id,))
        
        conn.commit()
        success = cursor.rowcount > 0
        cursor.close()
        
        return success
    
    def search_by_user(self, user_id: str, limit: int = 100) -> List[Conversation]:
        """Search conversations by user ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.db_type == 'sqlite':
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
            ''', (user_id, limit))
        
        rows = cursor.fetchall()
        cursor.close()
        
        return [self._row_to_conversation(row) for row in rows]
    
    def search_by_topic(
        self,
        topic_query: str,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Conversation]:
        """Search conversations by topic."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.db_type == 'sqlite':
            if user_id:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE user_id = ? AND topic LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                ''', (user_id, f'%{topic_query}%', limit))
            else:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE topic LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                ''', (f'%{topic_query}%', limit))
        else:
            # PostgreSQL full-text search
            if user_id:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE user_id = %s 
                    AND to_tsvector('english', topic) @@ plainto_tsquery('english', %s)
                    ORDER BY updated_at DESC
                    LIMIT %s
                ''', (user_id, topic_query, limit))
            else:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE to_tsvector('english', topic) @@ plainto_tsquery('english', %s)
                    ORDER BY updated_at DESC
                    LIMIT %s
                ''', (topic_query, limit))
        
        rows = cursor.fetchall()
        cursor.close()
        
        return [self._row_to_conversation(row) for row in rows]
    
    def search_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Conversation]:
        """Search conversations by date range."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.db_type == 'sqlite':
            if user_id:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE user_id = ? AND created_at BETWEEN ? AND ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (user_id, start_date, end_date, limit))
            else:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE created_at BETWEEN ? AND ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (start_date, end_date, limit))
        else:
            if user_id:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE user_id = %s AND created_at BETWEEN %s AND %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ''', (user_id, start_date, end_date, limit))
            else:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE created_at BETWEEN %s AND %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ''', (start_date, end_date, limit))
        
        rows = cursor.fetchall()
        cursor.close()
        
        return [self._row_to_conversation(row) for row in rows]
    
    def get_recent_conversations(
        self,
        days: int = 7,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Conversation]:
        """Get recent conversations."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.search_by_date_range(start_date, end_date, user_id, limit)
    
    def get_conversation_count(self, user_id: Optional[str] = None) -> int:
        """Get total conversation count."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if user_id:
            if self.db_type == 'sqlite':
                cursor.execute('SELECT COUNT(*) FROM conversations WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('SELECT COUNT(*) FROM conversations WHERE user_id = %s', (user_id,))
        else:
            cursor.execute('SELECT COUNT(*) FROM conversations')
        
        count = cursor.fetchone()[0]
        cursor.close()
        
        return count
    
    def export_conversations(
        self,
        user_id: Optional[str] = None,
        filepath: Optional[str] = None
    ) -> str:
        """Export conversations to JSON file."""
        if filepath is None:
            filepath = f'conversations_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if user_id:
            if self.db_type == 'sqlite':
                cursor.execute('SELECT * FROM conversations WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('SELECT * FROM conversations WHERE user_id = %s', (user_id,))
        else:
            cursor.execute('SELECT * FROM conversations')
        
        rows = cursor.fetchall()
        cursor.close()
        
        conversations = [self._row_to_dict(row) for row in rows]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, default=str)
        
        return filepath
    
    def _row_to_conversation(self, row) -> Conversation:
        """Convert database row to Conversation object."""
        return Conversation(
            id=row['id'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            topic=row['topic'] or '',
            messages=json.loads(row['messages']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=json.loads(row['metadata'])
        )
    
    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary."""
        return {
            'id': row['id'],
            'user_id': row['user_id'],
            'session_id': row['session_id'],
            'topic': row['topic'],
            'messages': json.loads(row['messages']),
            'created_at': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else row['created_at'],
            'updated_at': row['updated_at'].isoformat() if hasattr(row['updated_at'], 'isoformat') else row['updated_at'],
            'metadata': json.loads(row['metadata'])
        }
    
    def _placeholder(self) -> str:
        """Get placeholder for SQL query."""
        return '?' if self.db_type == 'sqlite' else '%s'
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Convenience functions
def create_conversation_store(
    db_type: str = 'sqlite',
    db_path: str = 'data/conversations.db'
) -> ConversationStore:
    """Create conversation store."""
    return ConversationStore(db_type=db_type, db_path=db_path)


if __name__ == '__main__':
    print("Conversation History Module Demo")
    print("="*50)
    
    # Create store
    store = create_conversation_store()
    
    # Create conversation
    conv_id = store.create_conversation(
        user_id='user_123',
        session_id='session_456',
        topic='Python Programming'
    )
    print(f"Created conversation: {conv_id}")
    
    # Add messages
    store.add_message(conv_id, 'user', 'How do I define a function in Python?')
    store.add_message(conv_id, 'assistant', 'You can define a function using the def keyword.')
    
    # Get conversation
    conv = store.get_conversation(conv_id)
    print(f"Conversation topic: {conv.topic}")
    print(f"Messages: {len(conv.messages)}")
    
    # Search
    results = store.search_by_user('user_123')
    print(f"Found {len(results)} conversations for user")
    
    store.close()
