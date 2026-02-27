"""
Model Pruning Module
Remove redundant weights to optimize model size and inference
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Callable
import copy
from pathlib import Path
import json


class ModelPruner:
    """Prune models to reduce size and improve efficiency."""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.pruning_history = []
    
    def prune_model(
        self, 
        model: nn.Module,
        amount: float = 0.3,
        method: str = 'l1_unstructured',
        global_pruning: bool = True
    ) -> nn.Module:
        """
        Prune model weights.
        
        Args:
            model: Model to prune
            amount: Fraction of weights to prune (0-1)
            method: Pruning method ('l1_unstructured', 'random_unstructured', 'ln_structured')
            global_pruning: Whether to prune globally or per-layer
            
        Returns:
            Pruned model
        """
        print(f"Pruning model with {method} (amount={amount})...")
        
        model = model.to(self.device)
        
        # Get parameters to prune
        parameters_to_prune = self._get_prunable_parameters(model)
        
        if global_pruning:
            # Global pruning across all layers
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured if method == 'l1_unstructured' else prune.RandomUnstructured,
                amount=amount
            )
        else:
            # Per-layer pruning
            pruning_method = self._get_pruning_method(method)
            for module, param_name in parameters_to_prune:
                pruning_method(module, name=param_name, amount=amount)
        
        # Record pruning
        self.pruning_history.append({
            'method': method,
            'amount': amount,
            'global': global_pruning,
            'layers_pruned': len(parameters_to_prune)
        })
        
        print(f"✓ Pruned {len(parameters_to_prune)} layers")
        
        # Compute sparsity
        sparsity = self._compute_sparsity(model)
        print(f"  Model sparsity: {sparsity:.2%}")
        
        return model
    
    def _get_prunable_parameters(self, model: nn.Module) -> List[tuple]:
        """Get list of prunable parameters."""
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
                if module.bias is not None:
                    parameters_to_prune.append((module, 'bias'))
            elif isinstance(module, nn.LSTM):
                # LSTM weights
                for param_name in ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']:
                    if hasattr(module, param_name):
                        parameters_to_prune.append((module, param_name))
        
        return parameters_to_prune
    
    def _get_pruning_method(self, method: str) -> Callable:
        """Get pruning method by name."""
        methods = {
            'l1_unstructured': prune.l1_unstructured,
            'random_unstructured': prune.random_unstructured,
            'ln_structured': prune.ln_structured,
        }
        
        if method not in methods:
            raise ValueError(f"Unknown pruning method: {method}")
        
        return methods[method]
    
    def remove_redundant_weights(
        self, 
        model: nn.Module,
        threshold: float = 1e-5
    ) -> nn.Module:
        """
        Remove redundant (duplicate or near-zero) weights.
        
        Args:
            model: Model to process
            threshold: Threshold for considering weights as zero
            
        Returns:
            Model with redundant weights removed
        """
        print("Removing redundant weights...")
        
        removed_count = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Find near-zero weights
                mask = torch.abs(param) > threshold
                
                # Zero out small weights
                with torch.no_grad():
                    param.data *= mask
                
                removed = (~mask).sum().item()
                removed_count += removed
                
                if removed > 0:
                    print(f"  {name}: removed {removed} near-zero weights")
        
        print(f"✓ Total weights removed: {removed_count}")
        
        return model
    
    def prune_heads(
        self,
        model: nn.Module,
        heads_to_prune: Dict[str, List[int]]
    ) -> nn.Module:
        """
        Prune specific attention heads (for transformer models).
        
        Args:
            model: Transformer model
            heads_to_prune: Dict mapping layer names to head indices to prune
            
        Returns:
            Model with pruned heads
        """
        print("Pruning attention heads...")
        
        for layer_name, heads in heads_to_prune.items():
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                if hasattr(layer, 'prune_heads'):
                    layer.prune_heads(heads)
                    print(f"  Pruned heads {heads} from {layer_name}")
        
        return model
    
    def _compute_sparsity(self, model: nn.Module) -> float:
        """Compute overall model sparsity."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0
    
    def make_permanent(self, model: nn.Module) -> nn.Module:
        """Make pruning permanent by removing mask buffers."""
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')
        
        print("✓ Pruning masks made permanent")
        return model
    
    def get_compression_ratio(self, model: nn.Module) -> Dict:
        """Get compression statistics."""
        sparsity = self._compute_sparsity(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = total_params * (1 - sparsity)
        
        return {
            'total_parameters': total_params,
            'nonzero_parameters': int(nonzero_params),
            'zero_parameters': int(total_params * sparsity),
            'sparsity': sparsity,
            'compression_ratio': total_params / nonzero_params if nonzero_params > 0 else 1.0
        }


class FineTuner:
    """Fine-tune model after pruning to recover accuracy."""
    
    def __init__(
        self,
        learning_rate: float = 1e-5,
        epochs: int = 3,
        warmup_steps: int = 100
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
    
    def fine_tune_after_pruning(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader = None,
        optimizer = None,
        scheduler = None
    ) -> nn.Module:
        """
        Fine-tune pruned model to recover accuracy.
        
        Args:
            model: Pruned model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optional optimizer
            scheduler: Optional learning rate scheduler
            
        Returns:
            Fine-tuned model
        """
        print(f"Fine-tuning after pruning ({self.epochs} epochs)...")
        
        device = next(model.parameters()).device
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
        
        model.train()
        global_step = 0
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = nn.functional.cross_entropy(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                total_loss += loss.item()
                global_step += 1
                
                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"  Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
            
            # Validation
            if val_dataloader:
                val_loss = self._validate(model, val_dataloader)
                print(f"  Validation Loss: {val_loss:.4f}")
        
        print("✓ Fine-tuning complete")
        return model
    
    def _validate(self, model: nn.Module, dataloader) -> float:
        """Run validation."""
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = nn.functional.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
        
        model.train()
        return total_loss / len(dataloader)


def iterative_pruning(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    target_sparsity: float = 0.7,
    pruning_iterations: int = 5,
    finetune_epochs: int = 2
) -> nn.Module:
    """
    Perform iterative pruning with fine-tuning.
    
    Args:
        model: Model to prune
        train_dataloader: Training data
        val_dataloader: Validation data
        target_sparsity: Target overall sparsity
        pruning_iterations: Number of pruning iterations
        finetune_epochs: Epochs to fine-tune after each iteration
        
    Returns:
        Pruned and fine-tuned model
    """
    pruner = ModelPruner()
    finetuner = FineTuner(epochs=finetune_epochs)
    
    # Calculate per-iteration pruning amount
    amount_per_iter = 1 - (1 - target_sparsity) ** (1 / pruning_iterations)
    
    for iteration in range(pruning_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{pruning_iterations}")
        print(f"{'='*50}")
        
        # Prune
        model = pruner.prune_model(
            model,
            amount=amount_per_iter,
            method='l1_unstructured',
            global_pruning=True
        )
        
        # Fine-tune
        model = finetuner.fine_tune_after_pruning(
            model,
            train_dataloader,
            val_dataloader
        )
        
        # Check sparsity
        stats = pruner.get_compression_ratio(model)
        print(f"Current sparsity: {stats['sparsity']:.2%}")
    
    # Make pruning permanent
    model = pruner.make_permanent(model)
    
    return model


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    method: str = 'l1_unstructured',
    fine_tune: bool = False,
    train_dataloader = None,
    val_dataloader = None
) -> nn.Module:
    """
    Convenience function to prune a model.
    
    Args:
        model: Model to prune
        amount: Pruning amount (0-1)
        method: Pruning method
        fine_tune: Whether to fine-tune after pruning
        train_dataloader: Required if fine_tune=True
        val_dataloader: Optional validation data
        
    Returns:
        Pruned (and optionally fine-tuned) model
    """
    pruner = ModelPruner()
    
    # Prune
    model = pruner.prune_model(model, amount=amount, method=method)
    
    # Fine-tune if requested
    if fine_tune:
        if train_dataloader is None:
            raise ValueError("train_dataloader required for fine-tuning")
        
        finetuner = FineTuner()
        model = finetuner.fine_tune_after_pruning(
            model, train_dataloader, val_dataloader
        )
    
    return model


if __name__ == '__main__':
    print("Pruning Module Demo")
    print("="*50)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )
    
    # Show original stats
    pruner = ModelPruner()
    print("\nOriginal model:")
    stats = pruner.get_compression_ratio(model)
    print(f"  Parameters: {stats['total_parameters']:,}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    # Prune
    model = pruner.prune_model(model, amount=0.5)
    
    print("\nAfter pruning:")
    stats = pruner.get_compression_ratio(model)
    print(f"  Parameters: {stats['total_parameters']:,}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
