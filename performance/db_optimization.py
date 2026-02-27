"""
BI-IDE v8 - Database Optimization Module
Connection pooling, query optimization, and performance monitoring
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    import asyncpg

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    ASYNCPG_AVAILABLE = False

try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    psycopg2 = None
    pool = None
    POSTGRES_AVAILABLE = False
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Query execution statistics"""
    query: str
    calls: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    max_time: float = 0.0
    min_time: float = float('inf')
    rows: int = 0
    last_executed: Optional[datetime] = None
    
    def update(self, execution_time: float, rows_returned: int = 0):
        self.calls += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.calls
        self.max_time = max(self.max_time, execution_time)
        self.min_time = min(self.min_time, execution_time)
        self.rows += rows_returned
        self.last_executed = datetime.now()


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration"""
    min_connections: int = 5
    max_connections: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    command_timeout: int = 60
    ssl_mode: str = "prefer"
    application_name: str = "bi-ide-v8"


class DatabaseOptimizer:
    """Database optimization and monitoring"""
    
    def __init__(self, database_url: str, pool_config: Optional[ConnectionPoolConfig] = None):
        self.database_url = database_url
        self.pool_config = pool_config or ConnectionPoolConfig()
        self._pool: Optional[Any] = None
        self._query_stats: Dict[str, QueryStats] = {}
        self._slow_queries: List[Dict[str, Any]] = []
        self._slow_query_threshold = 1.0  # seconds
        self._stats_lock = asyncio.Lock()
        self._query_patterns = {
            'SELECT': re.compile(r'SELECT\s+(.+?)\s+FROM', re.IGNORECASE),
            'INSERT': re.compile(r'INSERT\s+INTO\s+(\w+)', re.IGNORECASE),
            'UPDATE': re.compile(r'UPDATE\s+(\w+)', re.IGNORECASE),
            'DELETE': re.compile(r'DELETE\s+FROM\s+(\w+)', re.IGNORECASE),
        }
    
    async def initialize(self):
        """Initialize connection pool"""
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.pool_config.min_connections,
            max_size=self.pool_config.max_connections,
            max_inactive_time=self.pool_config.pool_recycle,
            command_timeout=self.pool_config.command_timeout,
            server_settings={
                'application_name': self.pool_config.application_name
            }
        )
        logger.info(f"Database pool initialized: {self.pool_config.min_connections}-{self.pool_config.max_connections} connections")
    
    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool"""
        async with self._pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def transaction(self):
        """Execute in transaction"""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute(self, query: str, *args, timeout: Optional[int] = None) -> str:
        """Execute query with monitoring"""
        start_time = time.time()
        
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, *args, timeout=timeout)
        
        execution_time = time.time() - start_time
        await self._record_query(query, execution_time, 0)
        
        return result
    
    async def fetch(self, query: str, *args, timeout: Optional[int] = None) -> List[Dict]:
        """Fetch results with monitoring"""
        start_time = time.time()
        
        async with self._pool.acquire() as conn:
            result = await conn.fetch(query, *args, timeout=timeout)
        
        execution_time = time.time() - start_time
        await self._record_query(query, execution_time, len(result))
        
        return result
    
    async def fetchrow(self, query: str, *args, timeout: Optional[int] = None) -> Optional[Dict]:
        """Fetch single row with monitoring"""
        start_time = time.time()
        
        async with self._pool.acquire() as conn:
            result = await conn.fetchrow(query, *args, timeout=timeout)
        
        execution_time = time.time() - start_time
        await self._record_query(query, execution_time, 1 if result else 0)
        
        return result
    
    async def fetchval(self, query: str, *args, timeout: Optional[int] = None) -> Any:
        """Fetch single value with monitoring"""
        start_time = time.time()
        
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(query, *args, timeout=timeout)
        
        execution_time = time.time() - start_time
        await self._record_query(query, execution_time, 1)
        
        return result
    
    async def _record_query(self, query: str, execution_time: float, rows: int):
        """Record query statistics"""
        # Normalize query for grouping
        normalized = self._normalize_query(query)
        
        async with self._stats_lock:
            if normalized not in self._query_stats:
                self._query_stats[normalized] = QueryStats(query=normalized)
            
            self._query_stats[normalized].update(execution_time, rows)
            
            # Record slow query
            if execution_time > self._slow_query_threshold:
                self._slow_queries.append({
                    'query': query[:200],
                    'normalized': normalized,
                    'execution_time': execution_time,
                    'rows': rows,
                    'timestamp': datetime.now()
                })
                # Keep only last 1000 slow queries
                if len(self._slow_queries) > 1000:
                    self._slow_queries = self._slow_queries[-1000:]
                
                logger.warning(f"Slow query detected ({execution_time:.2f}s): {query[:100]}...")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for grouping similar queries"""
        # Remove excess whitespace
        normalized = ' '.join(query.split())
        # Replace literals with placeholders
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r"\b\d+\b", "?", normalized)
        return normalized[:200]
    
    # Query Optimization
    
    async def explain_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Get query execution plan"""
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
        
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(explain_query, *args)
            return result[0]['Plan'] if result else {}
    
    async def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Run ANALYZE on table"""
        async with self._pool.acquire() as conn:
            await conn.execute(f"ANALYZE {table_name}")
            
            # Get table stats
            stats = await conn.fetchrow("""
                SELECT 
                    schemaname,
                    relname as table_name,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables
                WHERE relname = $1
            """, table_name)
            
            return dict(stats) if stats else {}
    
    # Index Management
    
    async def get_index_recommendations(self) -> List[Dict[str, Any]]:
        """Get missing index recommendations"""
        async with self._pool.acquire() as conn:
            # Find frequently scanned tables without proper indexes
            recommendations = await conn.fetch("""
                SELECT 
                    schemaname,
                    relname as table_name,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    n_live_tup as estimated_rows
                FROM pg_stat_user_tables
                WHERE seq_scan > 0
                    AND (idx_scan IS NULL OR seq_scan > idx_scan * 10)
                    AND n_live_tup > 1000
                ORDER BY seq_tup_read DESC
                LIMIT 20
            """)
            
            return [dict(rec) for rec in recommendations]
    
    async def get_index_usage_stats(self) -> List[Dict[str, Any]]:
        """Get index usage statistics"""
        async with self._pool.acquire() as conn:
            stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    relname as table_name,
                    indexrelname as index_name,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                    idx_scan as index_scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes
                ORDER BY idx_scan ASC
                LIMIT 50
            """)
            
            return [dict(stat) for stat in stats]
    
    async def create_index_concurrently(self, table: str, columns: List[str], 
                                         index_name: Optional[str] = None) -> bool:
        """Create index concurrently to avoid locks"""
        if not index_name:
            index_name = f"idx_{table}_{'_'.join(columns)}"
        
        column_str = ', '.join(columns)
        query = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} ON {table} ({column_str})"
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(query)
            logger.info(f"Created index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False
    
    # Performance Monitoring
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self._pool:
            return {}
        
        return {
            'size': self._pool.get_size(),
            'free_size': self._pool.get_free_size(),
            'min_size': self._pool._min_size,
            'max_size': self._pool._max_size,
            'max_inactive_time': self._pool._max_inactive_time,
        }
    
    async def get_slow_queries(self, limit: int = 50, min_time: float = 1.0) -> List[Dict[str, Any]]:
        """Get slow query log"""
        async with self._stats_lock:
            slow = [q for q in self._slow_queries if q['execution_time'] >= min_time]
            slow.sort(key=lambda x: x['execution_time'], reverse=True)
            return slow[:limit]
    
    async def get_query_stats(self, sort_by: str = 'total_time', limit: int = 50) -> List[Dict[str, Any]]:
        """Get aggregated query statistics"""
        async with self._stats_lock:
            stats = sorted(
                self._query_stats.values(),
                key=lambda x: getattr(x, sort_by),
                reverse=True
            )
            
            return [
                {
                    'query': s.query[:100],
                    'calls': s.calls,
                    'total_time': s.total_time,
                    'avg_time': s.avg_time,
                    'max_time': s.max_time,
                    'min_time': s.min_time if s.min_time != float('inf') else 0,
                    'rows': s.rows
                }
                for s in stats[:limit]
            ]
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics"""
        async with self._pool.acquire() as conn:
            # Database size
            db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size(current_database()))")
            
            # Table statistics
            table_stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    relname,
                    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
                    n_live_tup,
                    n_dead_tup
                FROM pg_stat_user_tables
                ORDER BY pg_total_relation_size(relid) DESC
                LIMIT 20
            """)
            
            # Connection count
            connections = await conn.fetchval("""
                SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()
            """)
            
            # Cache hit ratio
            cache_hit = await conn.fetchval("""
                SELECT 
                    CASE WHEN sum(heap_blks_hit) + sum(heap_blks_read) = 0 
                        THEN 0 
                        ELSE round(100.0 * sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)), 2)
                    END as cache_hit_ratio
                FROM pg_statio_user_tables
            """)
            
            return {
                'database_size': db_size,
                'connections': connections,
                'cache_hit_ratio': cache_hit,
                'top_tables': [dict(t) for t in table_stats]
            }
    
    # Maintenance
    
    async def vacuum_analyze(self, table_name: Optional[str] = None, 
                             full: bool = False) -> bool:
        """Run VACUUM ANALYZE"""
        async with self._pool.acquire() as conn:
            if full:
                # Note: VACUUM FULL requires exclusive lock
                query = f"VACUUM FULL {'ANALYZE' if table_name else ''} {table_name or ''}"
            else:
                query = f"VACUUM ANALYZE {table_name or ''}"
            
            try:
                await conn.execute(query)
                logger.info(f"Completed: {query}")
                return True
            except Exception as e:
                logger.error(f"VACUUM failed: {e}")
                return False
    
    async def reset_stats(self):
        """Reset collected statistics"""
        async with self._stats_lock:
            self._query_stats.clear()
            self._slow_queries.clear()
        logger.info("Query statistics reset")


class QueryBuilder:
    """SQL Query Builder with optimization hints"""
    
    def __init__(self):
        self._select: List[str] = []
        self._from: str = ""
        self._joins: List[str] = []
        self._where: List[str] = []
        self._where_params: List[Any] = []
        self._order_by: List[str] = []
        self._group_by: List[str] = []
        self._having: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._use_index: Optional[str] = None
    
    def select(self, *columns: str) -> 'QueryBuilder':
        self._select.extend(columns)
        return self
    
    def from_table(self, table: str) -> 'QueryBuilder':
        self._from = table
        return self
    
    def join(self, table: str, on: str, join_type: str = "INNER") -> 'QueryBuilder':
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self
    
    def where(self, condition: str, *params) -> 'QueryBuilder':
        self._where.append(condition)
        self._where_params.extend(params)
        return self
    
    def order_by(self, column: str, direction: str = "ASC") -> 'QueryBuilder':
        self._order_by.append(f"{column} {direction}")
        return self
    
    def group_by(self, *columns: str) -> 'QueryBuilder':
        self._group_by.extend(columns)
        return self
    
    def limit(self, n: int) -> 'QueryBuilder':
        self._limit = n
        return self
    
    def offset(self, n: int) -> 'QueryBuilder':
        self._offset = n
        return self
    
    def use_index(self, index_name: str) -> 'QueryBuilder':
        self._use_index = index_name
        return self
    
    def build(self) -> Tuple[str, List[Any]]:
        """Build SQL query"""
        parts = ["SELECT"]
        
        # Columns
        parts.append(", ".join(self._select) if self._select else "*")
        
        # From with index hint if specified
        if self._use_index:
            parts.append(f"FROM {self._from} WITH (INDEX({self._use_index}))")
        else:
            parts.append(f"FROM {self._from}")
        
        # Joins
        parts.extend(self._joins)
        
        # Where
        if self._where:
            parts.append("WHERE " + " AND ".join(f"({w})" for w in self._where))
        
        # Group by
        if self._group_by:
            parts.append("GROUP BY " + ", ".join(self._group_by))
        
        # Having
        if self._having:
            parts.append("HAVING " + " AND ".join(self._having))
        
        # Order by
        if self._order_by:
            parts.append("ORDER BY " + ", ".join(self._order_by))
        
        # Limit/Offset
        if self._limit:
            parts.append(f"LIMIT {self._limit}")
        if self._offset:
            parts.append(f"OFFSET {self._offset}")
        
        return " ".join(parts), self._where_params


# Global optimizer instance
_optimizer: Optional[DatabaseOptimizer] = None


async def init_optimizer(database_url: str, pool_config: Optional[ConnectionPoolConfig] = None):
    """Initialize global database optimizer"""
    global _optimizer
    _optimizer = DatabaseOptimizer(database_url, pool_config)
    await _optimizer.initialize()


def get_optimizer() -> DatabaseOptimizer:
    """Get global optimizer instance"""
    if _optimizer is None:
        raise RuntimeError("Database optimizer not initialized")
    return _optimizer


async def close_optimizer():
    """Close global optimizer"""
    global _optimizer
    if _optimizer:
        await _optimizer.close()
        _optimizer = None
