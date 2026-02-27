"""
RTX 4090 Trainer - Training Pipeline for AI Models
التدريب على RTX 4090 Server
"""
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
import json
import os
from pathlib import Path
from datetime import datetime
import hashlib

from ai.rtx4090_client import RTX4090Client, get_rtx4090_client


class RTX4090Trainer:
    """Train models on RTX 4090 Server"""
    
    def __init__(
        self,
        client: Optional[RTX4090Client] = None,
        checkpoint_dir: str = "learning_data/checkpoints"
    ):
        self.client = client or get_rtx4090_client()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_jobs: Dict[str, Dict] = {}
    
    async def train_wise_man(
        self,
        wise_man_name: str,
        training_data: List[Dict[str, str]],
        epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        save_every: int = 1
    ) -> Dict[str, Any]:
        """
        Fine-tune a model for specific wise man on RTX 4090
        
        Args:
            wise_man_name: Name of the wise man (e.g., 'حكيم القرار')
            training_data: List of {"input": "...", "output": "..."} dicts
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            save_every: Save checkpoint every N epochs
        
        Returns:
            Training job info and status
        """
        job_id = self._generate_job_id(wise_man_name)
        
        payload = {
            "job_id": job_id,
            "wise_man": wise_man_name,
            "training_data": training_data,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "save_every": save_every,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_jobs[job_id] = {
            "status": "starting",
            "wise_man": wise_man_name,
            "started_at": datetime.now().isoformat()
        }
        
        async with self.client as client:
            try:
                async with client.session.post(
                    f"{client.base_url}/train",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        self.training_jobs[job_id].update({
                            "status": "running",
                            "rtx_job_id": result.get("job_id"),
                            "estimated_time": result.get("estimated_minutes")
                        })
                        return {
                            "success": True,
                            "job_id": job_id,
                            "rtx_job_id": result.get("job_id"),
                            "status": "started",
                            "estimated_minutes": result.get("estimated_minutes")
                        }
                    else:
                        error = await resp.text()
                        self.training_jobs[job_id]["status"] = "failed"
                        return {
                            "success": False,
                            "job_id": job_id,
                            "error": f"RTX 4090 error {resp.status}: {error}"
                        }
            except Exception as e:
                self.training_jobs[job_id]["status"] = "failed"
                return {
                    "success": False,
                    "job_id": job_id,
                    "error": str(e)
                }
    
    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job"""
        if job_id not in self.training_jobs:
            return {"error": "Job not found"}
        
        job = self.training_jobs[job_id]
        rtx_job_id = job.get("rtx_job_id")
        
        if not rtx_job_id:
            return job
        
        async with self.client as client:
            try:
                async with client.session.get(
                    f"{client.base_url}/train/status/{rtx_job_id}"
                ) as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        job.update(status)
                        return job
                    else:
                        return {**job, "error": f"Failed to get status: {resp.status}"}
            except Exception as e:
                return {**job, "error": str(e)}
    
    async def upload_checkpoint(
        self, 
        checkpoint_path: str,
        layer_name: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Upload checkpoint to RTX 4090 server
        
        Args:
            checkpoint_path: Local path to checkpoint file
            layer_name: Layer name (e.g., 'high_council', 'balance')
            metadata: Optional metadata about the checkpoint
        
        Returns:
            Upload result with checkpoint info
        """
        path = Path(checkpoint_path)
        if not path.exists():
            return {"success": False, "error": f"Checkpoint not found: {checkpoint_path}"}
        
        async with self.client as client:
            try:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    path.open("rb"),
                    filename=path.name,
                    content_type="application/octet-stream"
                )
                data.add_field("layer", layer_name)
                if metadata:
                    data.add_field("metadata", json.dumps(metadata))
                
                async with client.session.post(
                    f"{client.base_url}/checkpoints/upload",
                    data=data
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return {
                            "success": True,
                            "checkpoint_id": result.get("checkpoint_id"),
                            "layer": layer_name,
                            "filename": path.name,
                            "size_mb": round(path.stat().st_size / (1024*1024), 2)
                        }
                    else:
                        error = await resp.text()
                        return {
                            "success": False,
                            "error": f"Upload failed: {resp.status} - {error}"
                        }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    async def download_checkpoint(
        self,
        layer_name: str,
        filename: str,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download checkpoint from RTX 4090 server
        
        Args:
            layer_name: Layer name
            filename: Checkpoint filename
            save_dir: Local directory to save (defaults to checkpoint_dir/layer_name)
        
        Returns:
            Download result with local path
        """
        if save_dir is None:
            save_dir = self.checkpoint_dir / layer_name
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        local_path = save_dir / filename
        
        async with self.client as client:
            try:
                async with client.session.get(
                    f"{client.base_url}/checkpoints/download/{layer_name}/{filename}"
                ) as resp:
                    if resp.status == 200:
                        with open(local_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        return {
                            "success": True,
                            "local_path": str(local_path),
                            "size_mb": round(local_path.stat().st_size / (1024*1024), 2)
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Download failed: {resp.status}"
                        }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    async def sync_all_checkpoints(self) -> Dict[str, Any]:
        """Sync all checkpoints from RTX 4090 to local storage"""
        async with self.client as client:
            try:
                # Get list of checkpoints from RTX 4090
                checkpoints = await client.list_checkpoints()
                
                synced = []
                failed = []
                
                for ckpt in checkpoints:
                    layer = ckpt.get("layer")
                    filename = ckpt.get("file")
                    
                    result = await self.download_checkpoint(layer, filename)
                    if result.get("success"):
                        synced.append(f"{layer}/{filename}")
                    else:
                        failed.append(f"{layer}/{filename}: {result.get('error')}")
                
                return {
                    "success": len(failed) == 0,
                    "synced_count": len(synced),
                    "failed_count": len(failed),
                    "synced": synced,
                    "failed": failed
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    async def evaluate_model(
        self,
        checkpoint_path: str,
        test_data: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        
        Args:
            checkpoint_path: Path to checkpoint to evaluate
            test_data: List of test examples
        
        Returns:
            Evaluation metrics
        """
        payload = {
            "checkpoint_path": checkpoint_path,
            "test_data": test_data
        }
        
        async with self.client as client:
            try:
                async with client.session.post(
                    f"{client.base_url}/evaluate",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        error = await resp.text()
                        return {"error": f"Evaluation failed: {resp.status} - {error}"}
            except Exception as e:
                return {"error": str(e)}
    
    def _generate_job_id(self, wise_man_name: str) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(wise_man_name.encode()).hexdigest()[:8]
        return f"{name_hash}_{timestamp}"
    
    async def create_training_dataset(
        self,
        wise_man_name: str,
        source_files: List[str],
        output_format: str = "jsonl"
    ) -> Dict[str, Any]:
        """
        Create training dataset from source files
        
        Args:
            wise_man_name: Target wise man
            source_files: List of file paths with training examples
            output_format: Output format (jsonl, csv)
        
        Returns:
            Dataset info
        """
        dataset = []
        total_examples = 0
        
        for source_file in source_files:
            path = Path(source_file)
            if not path.exists():
                continue
            
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if path.suffix == ".jsonl":
                        for line in f:
                            data = json.loads(line.strip())
                            if self._matches_wise_man(data, wise_man_name):
                                dataset.append(data)
                                total_examples += 1
                    elif path.suffix == ".json":
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if self._matches_wise_man(item, wise_man_name):
                                    dataset.append(item)
                                    total_examples += 1
            except Exception as e:
                print(f"Error reading {source_file}: {e}")
        
        # Save dataset
        dataset_path = self.checkpoint_dir / f"{wise_man_name}_dataset.jsonl"
        with open(dataset_path, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        return {
            "success": True,
            "wise_man": wise_man_name,
            "total_examples": total_examples,
            "dataset_path": str(dataset_path),
            "source_files": source_files
        }
    
    def _matches_wise_man(self, data: Dict, wise_man_name: str) -> bool:
        """Check if training example matches wise man"""
        # Check various possible keys
        wm_indicators = [
            data.get("wise_man"),
            data.get("wise_man_name"),
            data.get("expert"),
            data.get("personality")
        ]
        return any(ind == wise_man_name for ind in wm_indicators if ind)


class ContinuousTrainer:
    """Continuous training with automatic data collection"""
    
    def __init__(self, trainer: Optional[RTX4090Trainer] = None):
        self.trainer = trainer or RTX4090Trainer()
        self.is_training = False
        self.training_task = None
    
    async def start_continuous_training(
        self,
        wise_men: List[str],
        data_sources: List[str],
        interval_hours: float = 24.0
    ):
        """Start continuous training loop"""
        self.is_training = True
        
        while self.is_training:
            for wise_man in wise_men:
                if not self.is_training:
                    break
                
                # Collect new training data
                dataset = await self.trainer.create_training_dataset(
                    wise_man,
                    data_sources
                )
                
                if dataset.get("total_examples", 0) > 0:
                    # Start training
                    result = await self.trainer.train_wise_man(
                        wise_man,
                        [],  # Will be loaded from dataset
                        epochs=1  # Quick incremental training
                    )
                    print(f"Continuous training for {wise_man}: {result}")
            
            # Wait for next cycle
            await asyncio.sleep(interval_hours * 3600)
    
    def stop(self):
        """Stop continuous training"""
        self.is_training = False


# Global trainer instance
_default_trainer: Optional[RTX4090Trainer] = None


def get_rtx4090_trainer() -> RTX4090Trainer:
    """Get or create default RTX 4090 trainer"""
    global _default_trainer
    if _default_trainer is None:
        _default_trainer = RTX4090Trainer()
    return _default_trainer


# Test function
async def test_trainer():
    """Test RTX 4090 trainer"""
    print("🧪 Testing RTX 4090 Trainer...")
    
    trainer = RTX4090Trainer()
    
    # Test training job
    print("\n📚 Testing training job creation...")
    result = await trainer.train_wise_man(
        wise_man_name="حكيم القرار",
        training_data=[
            {"input": "كيف أتخذ قراراً صائباً؟", "output": "القرار الصائب يحتاج لتحليل البيانات والنظر في المستقبل."}
        ],
        epochs=1
    )
    print(f"Training result: {result}")
    
    # Test checkpoint sync
    print("\n🔄 Testing checkpoint sync...")
    sync_result = await trainer.sync_all_checkpoints()
    print(f"Synced {sync_result.get('synced_count', 0)} checkpoints")
    
    print("\n✅ Trainer tests completed!")


if __name__ == "__main__":
    asyncio.run(test_trainer())
