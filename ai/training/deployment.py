"""
Model Deployment Module
Canary deployment, rollback, and version management with RTX 4090 integration
"""

import json
import shutil
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading
import time
import requests


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_name: str
    checkpoint_path: str
    metadata: Dict[str, Any]
    created_at: datetime
    metrics: Dict[str, float]
    status: str = 'staging'  # 'staging', 'canary', 'production', 'deprecated'
    
    def to_dict(self) -> Dict:
        return {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'checkpoint_path': self.checkpoint_path,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'metrics': self.metrics,
            'status': self.status
        }


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    canary_percentage: float = 10.0  # Percentage of traffic to canary
    canary_duration: int = 300  # Seconds to run canary
    rollback_threshold: float = 0.95  # Error rate threshold for auto-rollback
    health_check_interval: int = 30  # Seconds between health checks
    auto_promote: bool = True  # Auto-promote after successful canary


class ModelRegistry:
    """Registry for model versions."""
    
    def __init__(self, registry_dir: str = 'models/registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, ModelVersion] = {}
        self.production_version: Optional[str] = None
        self.canary_version: Optional[str] = None
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_dir / 'registry.json'
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                
                for v_data in data.get('versions', []):
                    version = ModelVersion(
                        version_id=v_data['version_id'],
                        model_name=v_data['model_name'],
                        checkpoint_path=v_data['checkpoint_path'],
                        metadata=v_data['metadata'],
                        created_at=datetime.fromisoformat(v_data['created_at']),
                        metrics=v_data['metrics'],
                        status=v_data['status']
                    )
                    self.versions[version.version_id] = version
                
                self.production_version = data.get('production_version')
                self.canary_version = data.get('canary_version')
    
    def _save_registry(self):
        """Save registry to disk."""
        data = {
            'versions': [v.to_dict() for v in self.versions.values()],
            'production_version': self.production_version,
            'canary_version': self.canary_version,
            'updated_at': datetime.now().isoformat()
        }
        
        registry_file = self.registry_dir / 'registry.json'
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_version(
        self,
        model_name: str,
        checkpoint_path: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register new model version.
        
        Returns:
            Version ID
        """
        # Generate version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{model_name}_{timestamp}"
        
        # Copy checkpoint to registry
        dest_dir = self.registry_dir / 'checkpoints' / version_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_src = Path(checkpoint_path)
        if checkpoint_src.is_file():
            shutil.copy2(checkpoint_src, dest_dir / checkpoint_src.name)
        elif checkpoint_src.is_dir():
            shutil.copytree(checkpoint_src, dest_dir / 'model', dirs_exist_ok=True)
        
        # Create version record
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            checkpoint_path=str(dest_dir),
            metadata=metadata or {},
            created_at=datetime.now(),
            metrics=metrics,
            status='staging'
        )
        
        self.versions[version_id] = version
        self._save_registry()
        
        print(f"Registered model version: {version_id}")
        return version_id
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get version by ID."""
        return self.versions.get(version_id)
    
    def list_versions(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """List all versions, optionally filtered by model name."""
        versions = list(self.versions.values())
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def update_version_status(self, version_id: str, status: str) -> bool:
        """Update version status."""
        if version_id in self.versions:
            self.versions[version_id].status = status
            self._save_registry()
            return True
        return False
    
    def delete_version(self, version_id: str) -> bool:
        """Delete version from registry."""
        if version_id in self.versions:
            version = self.versions[version_id]
            
            # Remove checkpoint
            checkpoint_path = Path(version.checkpoint_path)
            if checkpoint_path.exists():
                if checkpoint_path.is_dir():
                    shutil.rmtree(checkpoint_path)
                else:
                    checkpoint_path.unlink()
            
            del self.versions[version_id]
            
            # Update references
            if self.production_version == version_id:
                self.production_version = None
            if self.canary_version == version_id:
                self.canary_version = None
            
            self._save_registry()
            return True
        return False


class ModelDeployment:
    """
    Model deployment with canary and rollback support.
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        config: Optional[DeploymentConfig] = None,
        rtx4090_server_url: Optional[str] = None
    ):
        self.registry = registry
        self.config = config or DeploymentConfig()
        self.rtx4090_server_url = rtx4090_server_url
        
        self.deployment_status: Dict[str, Any] = {}
        self.metrics_collector = MetricsCollector()
        
        self._stop_event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
    
    def deploy(
        self,
        version_id: str,
        skip_canary: bool = False
    ) -> bool:
        """
        Deploy model version.
        
        Args:
            version_id: Version to deploy
            skip_canary: Skip canary and deploy directly to production
            
        Returns:
            True if successful
        """
        version = self.registry.get_version(version_id)
        if not version:
            print(f"Version {version_id} not found")
            return False
        
        print(f"\n{'='*50}")
        print(f"Deploying model: {version_id}")
        print(f"{'='*50}\n")
        
        if skip_canary:
            return self._deploy_to_production(version_id)
        else:
            return self._start_canary(version_id)
    
    def _start_canary(self, version_id: str) -> bool:
        """Start canary deployment."""
        print(f"Starting canary deployment ({self.config.canary_percentage}% traffic)")
        
        # Update registry
        self.registry.canary_version = version_id
        self.registry.update_version_status(version_id, 'canary')
        self.registry._save_registry()
        
        # Deploy to RTX 4090 server if configured
        if self.rtx4090_server_url:
            self._deploy_to_rtx4090(version_id)
        
        # Start monitoring
        self._start_monitoring(version_id)
        
        # Schedule canary evaluation
        def evaluate_canary():
            time.sleep(self.config.canary_duration)
            self._evaluate_canary(version_id)
        
        threading.Thread(target=evaluate_canary, daemon=True).start()
        
        print(f"Canary deployment started. Monitoring for {self.config.canary_duration}s")
        return True
    
    def _evaluate_canary(self, version_id: str):
        """Evaluate canary deployment and decide to promote or rollback."""
        print("\nEvaluating canary deployment...")
        
        # Collect metrics
        canary_metrics = self.metrics_collector.get_metrics(version_id)
        
        # Check error rate
        error_rate = canary_metrics.get('error_rate', 0)
        latency_p99 = canary_metrics.get('latency_p99', 0)
        
        print(f"  Error rate: {error_rate:.2%}")
        print(f"  P99 latency: {latency_p99:.2f}ms")
        
        # Decision
        if error_rate > (1 - self.config.rollback_threshold):
            print("❌ Canary failed - initiating rollback")
            self.rollback()
        elif self.config.auto_promote:
            print("✅ Canary successful - promoting to production")
            self._promote_to_production(version_id)
        else:
            print("⏸️ Canary complete - awaiting manual promotion")
    
    def _promote_to_production(self, version_id: str) -> bool:
        """Promote canary to full production."""
        print(f"Promoting {version_id} to production")
        
        # Update previous production version
        if self.registry.production_version:
            self.registry.update_version_status(
                self.registry.production_version,
                'deprecated'
            )
        
        # Set new production version
        self.registry.production_version = version_id
        self.registry.canary_version = None
        self.registry.update_version_status(version_id, 'production')
        self.registry._save_registry()
        
        # Update RTX 4090 server
        if self.rtx4090_server_url:
            self._update_rtx4090_production(version_id)
        
        print(f"✅ {version_id} is now in production")
        return True
    
    def _deploy_to_production(self, version_id: str) -> bool:
        """Deploy directly to production (skip canary)."""
        print(f"Deploying {version_id} directly to production")
        
        # Deploy to RTX 4090
        if self.rtx4090_server_url:
            self._deploy_to_rtx4090(version_id)
        
        return self._promote_to_production(version_id)
    
    def rollback(self, target_version: Optional[str] = None) -> bool:
        """
        Rollback to previous version.
        
        Args:
            target_version: Specific version to rollback to (default: previous production)
            
        Returns:
            True if successful
        """
        if target_version is None:
            # Find previous production version
            versions = self.registry.list_versions()
            production_versions = [
                v for v in versions
                if v.status == 'deprecated' and v.version_id != self.registry.production_version
            ]
            
            if not production_versions:
                print("No previous version to rollback to")
                return False
            
            target_version = production_versions[0].version_id
        
        print(f"\n{'='*50}")
        print(f"Rolling back to: {target_version}")
        print(f"{'='*50}\n")
        
        # Stop canary if running
        if self.registry.canary_version:
            self.registry.update_version_status(self.registry.canary_version, 'staging')
            self.registry.canary_version = None
        
        # Deprecate current production
        if self.registry.production_version:
            self.registry.update_version_status(
                self.registry.production_version,
                'deprecated'
            )
        
        # Restore target version
        self.registry.production_version = target_version
        self.registry.update_version_status(target_version, 'production')
        self.registry._save_registry()
        
        # Update RTX 4090 server
        if self.rtx4090_server_url:
            self._update_rtx4090_production(target_version)
        
        print(f"✅ Rolled back to {target_version}")
        return True
    
    def _deploy_to_rtx4090(self, version_id: str):
        """Deploy model to RTX 4090 server."""
        if not self.rtx4090_server_url:
            return
        
        version = self.registry.get_version(version_id)
        if not version:
            return
        
        try:
            response = requests.post(
                f"{self.rtx4090_server_url}/deploy",
                json={
                    'version_id': version_id,
                    'checkpoint_path': version.checkpoint_path,
                    'mode': 'canary' if version.status == 'canary' else 'production'
                },
                timeout=60
            )
            
            if response.status_code == 200:
                print(f"  Deployed to RTX 4090: {version_id}")
            else:
                print(f"  Failed to deploy to RTX 4090: {response.text}")
        
        except Exception as e:
            print(f"  Error deploying to RTX 4090: {e}")
    
    def _update_rtx4090_production(self, version_id: str):
        """Update production model on RTX 4090."""
        if not self.rtx4090_server_url:
            return
        
        try:
            response = requests.post(
                f"{self.rtx4090_server_url}/set_production",
                json={'version_id': version_id},
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"  Updated RTX 4090 production model: {version_id}")
        
        except Exception as e:
            print(f"  Error updating RTX 4090: {e}")
    
    def _start_monitoring(self, version_id: str):
        """Start monitoring thread."""
        self._stop_event.clear()
        
        def monitor():
            while not self._stop_event.is_set():
                # Collect metrics
                metrics = self._collect_metrics(version_id)
                self.metrics_collector.record(version_id, metrics)
                
                # Check for auto-rollback
                if metrics.get('error_rate', 0) > (1 - self.config.rollback_threshold):
                    print(f"⚠️ High error rate detected: {metrics['error_rate']:.2%}")
                    self.rollback()
                    break
                
                time.sleep(self.config.health_check_interval)
        
        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()
    
    def _collect_metrics(self, version_id: str) -> Dict[str, float]:
        """Collect deployment metrics."""
        # In production, this would query actual metrics
        # For demo, return simulated metrics
        return {
            'error_rate': 0.01,
            'latency_p99': 150.0,
            'requests_per_second': 100.0,
            'success_rate': 0.99
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        return {
            'production_version': self.registry.production_version,
            'canary_version': self.registry.canary_version,
            'total_versions': len(self.registry.versions),
            'deployment_status': self.deployment_status
        }


class MetricsCollector:
    """Collect and aggregate deployment metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = defaultdict(list)
    
    def record(self, version_id: str, metrics: Dict[str, float]):
        """Record metrics for version."""
        self.metrics[version_id].append({
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def get_metrics(self, version_id: str) -> Dict[str, float]:
        """Get aggregated metrics for version."""
        version_metrics = self.metrics.get(version_id, [])
        
        if not version_metrics:
            return {}
        
        # Calculate averages
        result = {}
        for key in version_metrics[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in version_metrics if key in m]
                result[key] = sum(values) / len(values) if values else 0
        
        return result


def create_deployment_pipeline(
    registry_dir: str = 'models/registry',
    rtx4090_url: Optional[str] = None
) -> ModelDeployment:
    """Create deployment pipeline."""
    registry = ModelRegistry(registry_dir)
    config = DeploymentConfig()
    return ModelDeployment(registry, config, rtx4090_url)


if __name__ == '__main__':
    print("Model Deployment Module Demo")
    print("="*50)
    
    # Create deployment pipeline
    deployment = create_deployment_pipeline()
    
    # Register a version
    version_id = deployment.registry.register_version(
        model_name='bi-ide-model',
        checkpoint_path='models/checkpoint.pt',
        metrics={'perplexity': 15.2, 'accuracy': 0.85},
        metadata={'trained_on': 'v8_corpus'}
    )
    
    print(f"\nRegistered version: {version_id}")
    
    # Get status
    status = deployment.get_status()
    print(f"\nDeployment Status:")
    print(f"  Production: {status['production_version']}")
    print(f"  Canary: {status['canary_version']}")
    print(f"  Total versions: {status['total_versions']}")
