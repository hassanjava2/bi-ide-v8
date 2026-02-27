"""
محمّل النوى المدربة - Checkpoint Loader
يحمّل أوزان النماذج المدربة مسبقاً لكل طبقة من طبقات المجلس

الطبقات المدربة:
- president (100% accuracy, 38K epochs)
- high_council (100%)
- guardian (100%)
- builder_council (100%)
- domain_experts (100%)
- execution (100%)
- meta_team (100%)
- learning_core (98.2%)
- scouts (97.7%)
- cosmic_bridge (97.6%)
- shadow_light (97.2%)
- seventh_dimension (97.0%)
- executive_controller (96.9%)
- meta_architect (93.3%)
- eternity (97.4%)
"""

import os
import glob
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Layer name mapping: checkpoint folder → hierarchy attribute
LAYER_MAPPING = {
    'president': 'president',
    'high_council': 'council',
    'guardian': 'guardian',
    'cosmic_bridge': 'cosmic_bridge',
    'domain_experts': 'experts',
    'execution': 'execution',
    'meta_team': 'meta',
    'scouts': 'scouts',
    'seventh_dimension': 'seventh',
    'shadow_light': 'balance',
    'eternity': 'eternity',
    'meta_architect': 'meta_architect',
    'builder_council': 'builder_council',
    'executive_controller': 'executive_controller',
    'learning_core': 'learning_core',
}


class CheckpointLoader:
    """
    محمّل النوى المدربة
    يحمّل .pt checkpoints ويربطها بطبقات المجلس
    """
    
    def __init__(self, checkpoint_dir: str = None):
        if checkpoint_dir is None:
            # Default: data/checkpoints relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_dir = os.path.join(project_root, 'data', 'checkpoints')
        
        self.checkpoint_dir = checkpoint_dir
        self.loaded_checkpoints: Dict[str, Dict[str, Any]] = {}
        self.load_errors: List[str] = []
    
    def discover_checkpoints(self) -> Dict[str, List[str]]:
        """
        اكتشاف كل الـ checkpoints المتوفرة
        Returns: dict of layer_name -> list of checkpoint files
        """
        available = {}
        
        if not os.path.exists(self.checkpoint_dir):
            print(f"⚠️ Checkpoint directory not found: {self.checkpoint_dir}")
            return available
        
        for layer_dir in sorted(os.listdir(self.checkpoint_dir)):
            layer_path = os.path.join(self.checkpoint_dir, layer_dir)
            if os.path.isdir(layer_path):
                pt_files = sorted(glob.glob(os.path.join(layer_path, '*.pt')))
                if pt_files:
                    available[layer_dir] = [os.path.basename(f) for f in pt_files]
        
        return available
    
    def get_best_checkpoint(self, layer_name: str) -> Optional[str]:
        """
        الحصول على أفضل checkpoint لطبقة معينة
        يفضل best_acc > latest
        """
        layer_dir = os.path.join(self.checkpoint_dir, layer_name)
        if not os.path.isdir(layer_dir):
            return None
        
        pt_files = sorted(glob.glob(os.path.join(layer_dir, '*.pt')))
        if not pt_files:
            return None
        
        # Find best accuracy file
        best_files = [f for f in pt_files if 'best_acc' in os.path.basename(f)]
        if best_files:
            # Sort by accuracy (descending), then by epoch (descending)
            def parse_accuracy(filepath):
                name = os.path.basename(filepath)
                try:
                    acc = float(name.split('acc')[1].split('_')[0])
                    return acc
                except (IndexError, ValueError):
                    return 0.0
            
            best_files.sort(key=parse_accuracy, reverse=True)
            return best_files[0]
        
        # Fallback to latest.pt
        latest = os.path.join(layer_dir, 'latest.pt')
        if os.path.exists(latest):
            return latest
        
        return pt_files[-1]  # Last file
    
    def load_checkpoint(self, layer_name: str, checkpoint_path: str = None) -> Optional[Dict]:
        """
        تحميل checkpoint لطبقة معينة
        """
        if not TORCH_AVAILABLE:
            self.load_errors.append(f"{layer_name}: torch not available")
            return None
        
        if checkpoint_path is None:
            checkpoint_path = self.get_best_checkpoint(layer_name)
        
        if checkpoint_path is None:
            self.load_errors.append(f"{layer_name}: no checkpoint found")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            info = {
                'layer': layer_name,
                'path': checkpoint_path,
                'epoch': checkpoint.get('epoch', 0),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'vocab_size': checkpoint.get('vocab_size', 0),
                'stats': checkpoint.get('stats', {}),
                'model_state': checkpoint.get('model_state', None),
                'has_optimizer': 'optimizer_state' in checkpoint,
                'has_scheduler': 'scheduler_state' in checkpoint,
            }
            
            self.loaded_checkpoints[layer_name] = info
            return info
            
        except Exception as e:
            self.load_errors.append(f"{layer_name}: {str(e)}")
            return None
    
    def load_all_checkpoints(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        تحميل كل الـ checkpoints المتوفرة
        """
        available = self.discover_checkpoints()
        
        if verbose:
            print(f"\n🧠 Loading Council Checkpoints from: {self.checkpoint_dir}")
            print(f"   Found {len(available)} layers with checkpoints")
            print("   " + "─" * 50)
        
        for layer_name in available:
            info = self.load_checkpoint(layer_name)
            
            if info and verbose:
                stats = info.get('stats', {})
                acc = stats.get('accuracy', 0)
                epoch = info.get('epoch', 0)
                loss = stats.get('loss', 0)
                samples = stats.get('samples', 0)
                
                print(f"   ✅ {layer_name:25s} | Acc: {acc:6.1f}% | Epoch: {epoch:,} | Samples: {samples:,}")
        
        if self.load_errors and verbose:
            print(f"\n   ⚠️ {len(self.load_errors)} errors:")
            for err in self.load_errors:
                print(f"      ❌ {err}")
        
        if verbose:
            print("   " + "─" * 50)
            print(f"   📊 Total loaded: {len(self.loaded_checkpoints)}/{len(available)}")
        
        return self.loaded_checkpoints
    
    def get_summary(self) -> Dict[str, Any]:
        """
        ملخص حالة النوى المحملة
        """
        total_samples = 0
        total_epochs = 0
        accuracies = []
        
        for name, info in self.loaded_checkpoints.items():
            stats = info.get('stats', {})
            total_samples += stats.get('samples', 0)
            total_epochs += info.get('epoch', 0)
            acc = stats.get('accuracy', 0)
            if acc > 0:
                accuracies.append(acc)
        
        return {
            'total_layers': len(self.loaded_checkpoints),
            'total_samples_trained': total_samples,
            'total_epochs': total_epochs,
            'average_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
            'min_accuracy': min(accuracies) if accuracies else 0,
            'max_accuracy': max(accuracies) if accuracies else 0,
            'layers': list(self.loaded_checkpoints.keys()),
            'errors': self.load_errors,
        }


# Singleton
_checkpoint_loader = None

def get_checkpoint_loader(checkpoint_dir: str = None) -> CheckpointLoader:
    """Get or create the singleton CheckpointLoader"""
    global _checkpoint_loader
    if _checkpoint_loader is None:
        _checkpoint_loader = CheckpointLoader(checkpoint_dir)
    return _checkpoint_loader
