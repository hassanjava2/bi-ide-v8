"""
Background Tasks - المهام في الخلفية
"""
import os
from typing import Dict, Any
from .celery_config import celery_app
from .logging_config import logger


@celery_app.task(bind=True, max_retries=3)
def process_learning_experience(self, experience_data: Dict[str, Any]):
    """Process a learning experience in background and store in DB"""
    import asyncio
    
    async def _store_experience():
        from core.database import db_manager
        
        exp_id = experience_data.get('id') or f"exp_{self.request.id}"
        await db_manager.store_learning_experience(
            exp_id=exp_id,
            exp_type=experience_data.get('type', 'unknown'),
            context=experience_data.get('context', {}),
            action=experience_data.get('action', ''),
            outcome=experience_data.get('outcome', ''),
            reward=experience_data.get('reward', 0.0)
        )
        return exp_id
    
    try:
        logger.info(f"Processing learning experience: {experience_data.get('type')}")
        
        # Run async DB operation in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        exp_id = loop.run_until_complete(_store_experience())
        loop.close()
        
        logger.info(f"Learning experience stored: {exp_id}")
        return {"status": "success", "experience_id": exp_id}
    
    except Exception as exc:
        logger.error(f"Learning experience processing failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=2)
def generate_code_embeddings(self, code_snippets: list):
    """Generate simple hash-based embeddings for code snippets (fallback until real model)"""
    import hashlib
    import json
    
    try:
        logger.info(f"Generating embeddings for {len(code_snippets)} snippets")
        
        embeddings = []
        for snippet in code_snippets:
            content = snippet.get('content', snippet.get('code', ''))
            
            # Simple hash-based embedding (128-dim) as fallback
            # In production, replace with sentence-transformers or similar
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Create 128-dim vector from hash (each 2 chars = 1 dim)
            embedding = [int(content_hash[i:i+2], 16) / 255.0 for i in range(0, 256, 2)]
            
            embeddings.append({
                "id": snippet.get('id'),
                "embedding": embedding,
                "content_hash": content_hash[:16]  # First 16 chars for reference
            })
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return {"status": "success", "embeddings": embeddings, "method": "hash_fallback"}
    
    except Exception as exc:
        logger.error(f"Embedding generation failed: {exc}")
        raise self.retry(exc=exc, countdown=30)


@celery_app.task(bind=True, max_retries=2)
def council_deliberation_task(self, topic: str, context: Dict):
    """
    Run council deliberation in background
    
    This task uses the High Council's discussion mechanism through the
    AI Hierarchy to conduct deliberations on important topics.
    """
    import asyncio
    
    try:
        logger.info(f"Starting council deliberation on: {topic}")
        
        from hierarchy import ai_hierarchy
        from hierarchy.high_council import Discussion
        
        # Get the high council instance
        council = ai_hierarchy.council
        
        # Create a new event loop for async operations in Celery
        async def _conduct_deliberation():
            """Conduct deliberation asynchronously"""
            # Create a discussion instance
            discussion = Discussion(
                topic=topic,
                initiator=context.get("initiator", "background_task"),
                opinions={}
            )
            
            # Use the council's internal discussion method
            await council._conduct_discussion(topic)
            
            # Get the current discussion result
            current = council.current_discussion
            if current:
                return {
                    "topic": current.topic,
                    "opinions": current.opinions,
                    "consensus": current.consensus,
                    "timestamp": current.timestamp.isoformat() if current.timestamp else None
                }
            return {"topic": topic, "status": "no_discussion_record"}
        
        # Run the async function
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(_conduct_deliberation())
        
        logger.info(f"Council deliberation completed: {topic}")
        return {
            "status": "success",
            "topic": topic,
            "decision": result.get("consensus", "no_consensus"),
            "opinions_count": len(result.get("opinions", {})),
            "timestamp": result.get("timestamp")
        }
    
    except Exception as exc:
        logger.error(f"Council deliberation failed: {exc}")
        # Retry with countdown
        raise self.retry(exc=exc, countdown=30)


@celery_app.task(bind=True)
def cleanup_old_data(self, days: int = 30):
    """Cleanup old data from database"""
    import asyncio
    from datetime import datetime, timedelta
    from sqlalchemy import delete
    
    async def _cleanup():
        from core.database import db_manager, SystemMetrics
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted_counts = {}
        
        async with db_manager.get_session() as session:
            # Cleanup old metrics
            try:
                stmt = delete(SystemMetrics).where(SystemMetrics.timestamp < cutoff)
                result = await session.execute(stmt)
                deleted_counts['metrics'] = result.rowcount
            except Exception as e:
                logger.warning(f"Failed to cleanup metrics: {e}")
            
            # Add more tables here as needed
            # stmt = delete(OtherTable).where(OtherTable.created_at < cutoff)
            # result = await session.execute(stmt)
            # deleted_counts['other'] = result.rowcount
        
        return deleted_counts
    
    try:
        logger.info(f"Cleaning up data older than {days} days")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        deleted = loop.run_until_complete(_cleanup())
        loop.close()
        
        total_deleted = sum(deleted.values())
        logger.info(f"Cleanup completed: {total_deleted} records deleted")
        
        return {"status": "success", "cleaned_days": days, "deleted": deleted}
    
    except Exception as exc:
        logger.error(f"Cleanup failed: {exc}")
        return {"status": "error", "error": str(exc)}


@celery_app.task(bind=True)
def sync_to_rtx4090(self, model_data: Dict):
    """Sync model data to RTX 5090 server"""
    try:
        import requests
        
        # يقرأ RTX5090_* أولاً مع fallback للقديم
        rtx_host = os.getenv("RTX5090_HOST", os.getenv("RTX4090_HOST", "192.168.1.164"))
        rtx_port = os.getenv("RTX5090_PORT", os.getenv("RTX4090_PORT", "8090"))
        rtx_url = f"http://{rtx_host}:{rtx_port}"
        
        response = requests.post(
            f"{rtx_url}/sync",
            json=model_data,
            timeout=60
        )
        
        return {
            "status": "success" if response.status_code == 200 else "failed",
            "rtx_response": response.status_code
        }
    
    except Exception as exc:
        logger.error(f"RTX 4090 sync failed: {exc}")
        return {"status": "error", "error": str(exc)}
