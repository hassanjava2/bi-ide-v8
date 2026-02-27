"""
Knowledge Distillation Module
Train smaller student models from larger teacher models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from torch.utils.data import DataLoader
from pathlib import Path
import json


class KnowledgeDistillation:
    """
    Knowledge Distillation for model compression.
    Transfer knowledge from teacher to student model.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = 'auto'
    ):
        """
        Initialize knowledge distillation.
        
        Args:
            teacher_model: Larger, trained teacher model
            student_model: Smaller student model to train
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (1-alpha for student loss)
            device: Device to use
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
        # Teacher in eval mode
        self.teacher_model.eval()
        
        # Disable gradients for teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: True labels for hard target loss
            
        Returns:
            Combined loss
        """
        # Soft targets from teacher (with temperature scaling)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss (distillation)
        distillation_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        if labels is not None and self.alpha < 1.0:
            # Hard target loss (standard cross-entropy)
            student_loss = F.cross_entropy(student_logits, labels)
            
            # Combined loss
            loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        else:
            loss = distillation_loss
        
        return loss
    
    def train_student(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        save_dir: Optional[str] = None,
        eval_every: int = 1000
    ) -> Dict[str, Any]:
        """
        Train student model using knowledge distillation.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate
            save_dir: Directory to save checkpoints
            eval_every: Evaluate every N steps
            
        Returns:
            Training history
        """
        print(f"Training student model with knowledge distillation...")
        print(f"  Temperature: {self.temperature}")
        print(f"  Alpha (distillation weight): {self.alpha}")
        print(f"  Epochs: {epochs}")
        
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.student_model.train()
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Prepare batch
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    labels = inputs.pop('labels', None)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                # Forward pass - teacher
                with torch.no_grad():
                    if isinstance(inputs, dict):
                        teacher_outputs = self.teacher_model(**inputs)
                    else:
                        teacher_outputs = self.teacher_model(inputs)
                    
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
                
                # Forward pass - student
                if isinstance(inputs, dict):
                    student_outputs = self.student_model(**inputs)
                else:
                    student_outputs = self.student_model(inputs)
                
                student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
                
                # Compute loss
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                optimizer.step()
                
                if global_step >= warmup_steps:
                    scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Step {global_step}, "
                          f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                
                # Evaluation
                if val_dataloader and global_step % eval_every == 0:
                    val_metrics = self.evaluate(val_dataloader)
                    print(f"  Validation - Loss: {val_metrics['loss']:.4f}, "
                          f"Accuracy: {val_metrics['accuracy']:.2%}")
                    
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_accuracy'].append(val_metrics['accuracy'])
                    
                    # Save best model
                    if save_dir and val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        self.save_checkpoint(save_dir, 'best_student.pt')
                    
                    self.student_model.train()
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            history['train_loss'].append(avg_epoch_loss)
            print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if save_dir:
                self.save_checkpoint(save_dir, f'student_epoch_{epoch+1}.pt')
        
        print("✓ Knowledge distillation training complete")
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate student model."""
        self.student_model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    labels = inputs.pop('labels', None)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                # Get outputs
                if isinstance(inputs, dict):
                    outputs = self.student_model(**inputs)
                else:
                    outputs = self.student_model(inputs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Compute loss if labels available
                if labels is not None:
                    loss = F.cross_entropy(logits, labels)
                    total_loss += loss.item()
                    
                    # Accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
        
        metrics = {
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0,
            'accuracy': correct / total if total > 0 else 0
        }
        
        return metrics
    
    def save_checkpoint(self, save_dir: str, filename: str):
        """Save student model checkpoint."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'temperature': self.temperature,
            'alpha': self.alpha
        }
        
        torch.save(checkpoint, Path(save_dir) / filename)
        print(f"  Saved checkpoint: {filename}")
    
    def compare_models(self, test_dataloader: DataLoader) -> Dict[str, Any]:
        """Compare teacher and student models."""
        print("\nComparing teacher and student models...")
        
        teacher_metrics = self._evaluate_model(self.teacher_model, test_dataloader)
        student_metrics = self._evaluate_model(self.student_model, test_dataloader)
        
        # Count parameters
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        comparison = {
            'teacher': {
                'parameters': teacher_params,
                'loss': teacher_metrics['loss'],
                'accuracy': teacher_metrics['accuracy']
            },
            'student': {
                'parameters': student_params,
                'loss': student_metrics['loss'],
                'accuracy': student_metrics['accuracy']
            },
            'compression_ratio': teacher_params / student_params,
            'accuracy_retention': student_metrics['accuracy'] / teacher_metrics['accuracy'] if teacher_metrics['accuracy'] > 0 else 0
        }
        
        print(f"\nComparison Results:")
        print(f"  Teacher parameters: {teacher_params:,}")
        print(f"  Student parameters: {student_params:,}")
        print(f"  Compression ratio: {comparison['compression_ratio']:.2f}x")
        print(f"  Teacher accuracy: {teacher_metrics['accuracy']:.2%}")
        print(f"  Student accuracy: {student_metrics['accuracy']:.2%}")
        print(f"  Accuracy retention: {comparison['accuracy_retention']:.2%}")
        
        return comparison
    
    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate a single model."""
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    labels = inputs.pop('labels', None)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                if labels is not None:
                    loss = F.cross_entropy(logits, labels)
                    total_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
        
        return {
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0,
            'accuracy': correct / total if total > 0 else 0
        }


class TemperatureScaling:
    """
    Temperature scaling for calibration and distillation.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def get_soft_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Get soft probabilities with temperature scaling."""
        return F.softmax(self.scale_logits(logits), dim=-1)
    
    def tune_temperature(
        self,
        model: nn.Module,
        val_dataloader: DataLoader,
        device: str = 'cuda'
    ) -> float:
        """
        Tune temperature on validation set for calibration.
        
        Returns:
            Optimal temperature
        """
        model.eval()
        
        # Collect logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    labels = inputs.pop('labels', None)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                all_logits.append(logits.cpu())
                if labels is not None:
                    all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        # Optimize temperature
        temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(all_logits / temperature, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        print(f"Optimal temperature: {self.temperature:.4f}")
        
        return self.temperature


def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    temperature: float = 4.0,
    alpha: float = 0.7,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    save_dir: Optional[str] = None
) -> nn.Module:
    """
    Convenience function for knowledge distillation.
    
    Returns:
        Trained student model
    """
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=temperature,
        alpha=alpha
    )
    
    history = distiller.train_student(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    return distiller.student_model


if __name__ == '__main__':
    print("Knowledge Distillation Module Demo")
    print("="*50)
    
    # Create teacher and student models
    teacher = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    
    student = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression: {teacher_params/student_params:.2f}x")
