import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, Any
from pathlib import Path
import argparse
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from dataclasses import dataclass
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data import MockDataset
from models.strap import STRAP, MAESTRAP, CNNSTRAP


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    # Model parameters
    model_type: str = "STRAP"  # Options: "STRAP", "MAESTRAP", "CNNSTRAP"
    img_size: int = 64
    patch_size: int = 8
    in_chans: int = 1
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 8
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 8
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.5
    n_tissues: int = 3
    n_factors: int = 5
    loss_coefficient: float = 0.5
    
    # Training parameters
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    
    # Dataset parameters
    n_samples: int = 1000
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 20
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    wandb_project: str = "strap-training"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42


class Trainer:
    """Training class for STRAP models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.console = Console()
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Initialize wandb if enabled
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"{config.model_type}_{config.embed_dim}d_{config.epochs}e"
            )
        elif config.use_wandb and not WANDB_AVAILABLE:
            self.console.print("[yellow]Warning: wandb not available, logging disabled[/yellow]")
            config.use_wandb = False
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        # Initialize model, dataset, and training components
        self._setup_model()
        self._setup_data()
        self._setup_training()
    
    def _setup_model(self):
        """Initialize the model based on configuration."""
        
        if self.config.model_type == "STRAP":
            self.model = STRAP(
                img_size=self.config.img_size,
                patch_size=self.config.patch_size,
                in_chans=self.config.in_chans,
                embed_dim=self.config.embed_dim,
                depth=self.config.depth,
                num_heads=self.config.num_heads,
                decoder_embed_dim=self.config.decoder_embed_dim,
                decoder_depth=self.config.decoder_depth,
                decoder_num_heads=self.config.decoder_num_heads,
                mlp_ratio=self.config.mlp_ratio,
                mask_ratio=self.config.mask_ratio,
                n_tissues=self.config.n_tissues,
                n_factors=self.config.n_factors,
                loss_coefficient=self.config.loss_coefficient,
            )
        elif self.config.model_type == "MAESTRAP":
            self.model = MAESTRAP(
                img_size=self.config.img_size,
                patch_size=self.config.patch_size,
                in_chans=self.config.in_chans,
                embed_dim=self.config.embed_dim,
                depth=self.config.depth,
                num_heads=self.config.num_heads,
                decoder_embed_dim=self.config.decoder_embed_dim,
                decoder_depth=self.config.decoder_depth,
                decoder_num_heads=self.config.decoder_num_heads,
                mlp_ratio=self.config.mlp_ratio,
                mask_ratio=self.config.mask_ratio,
                n_tissues=self.config.n_tissues,
                n_factors=self.config.n_factors,
                loss_coefficient=self.config.loss_coefficient,
            )
        elif self.config.model_type == "CNNSTRAP":
            self.model = CNNSTRAP(
                embed_dim=self.config.embed_dim,
                n_tissues=self.config.n_tissues,
                n_factors=self.config.n_factors,
                pretrained_resnet=True,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.console.print(Panel(
            f"[bold green]{self.config.model_type} Model Initialized[/bold green]\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"Device: {self.device}",
            title="Model Info"
        ))
    
    def _setup_data(self):
        """Setup datasets and data loaders."""
        # Create dataset
        dataset = MockDataset(
            n_samples=self.config.n_samples,
            image_shape=(self.config.in_chans, self.config.img_size, self.config.img_size),
            n_tissues=self.config.n_tissues,
            n_risk_factors=self.config.n_factors,
            seed=self.config.seed
        )
        
        # Split dataset
        train_size = int(self.config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.console.print(Panel(
            f"[bold blue]Dataset Setup Complete[/bold blue]\n"
            f"Total samples: {len(dataset)}\n"
            f"Training samples: {len(self.train_dataset)}\n"
            f"Validation samples: {len(self.val_dataset)}\n"
            f"Batch size: {self.config.batch_size}",
            title="Data Info"
        ))
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and other training components."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        cox_losses = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            
            task = progress.add_task(
                f"[green]Epoch {self.current_epoch + 1}/{self.config.epochs}",
                total=len(self.train_loader)
            )
            
            for batch_idx, (images, masks, risk_factors, events, times) in enumerate(self.train_loader):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                risk_factors = risk_factors.to(self.device)
                events = events.to(self.device)
                times = times.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.config.model_type in ["STRAP", "MAESTRAP"]:
                    outputs = self.model(
                        images=images,
                        masks=masks,
                        factors=risk_factors,
                        time=times,
                        event=events,
                        mask_ratio=self.config.mask_ratio
                    )
                else:  # CNNSTRAP
                    outputs = self.model(
                        images=images,
                        masks=masks,
                        factors=risk_factors,
                        time=times,
                        event=events
                    )
                
                loss = outputs["loss"]
                cox_loss = outputs["cox_loss"]
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Record losses
                epoch_losses.append(loss.item())
                cox_losses.append(cox_loss.item())
                
                # Update progress
                progress.update(task, advance=1)
                
                # Log batch metrics
                if batch_idx % self.config.log_interval == 0:
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "batch_loss": loss.item(),
                            "batch_cox_loss": cox_loss.item(),
                            "learning_rate": self.optimizer.param_groups[0]['lr'],
                            "epoch": self.current_epoch,
                            "batch": batch_idx
                        })
        
        avg_loss = float(np.mean(epoch_losses))
        avg_cox_loss = float(np.mean(cox_losses))
        
        return {
            "train_loss": avg_loss,
            "train_cox_loss": avg_cox_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        cox_losses = []
        
        with torch.no_grad():
            for images, masks, risk_factors, events, times in self.val_loader:
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                risk_factors = risk_factors.to(self.device)
                events = events.to(self.device)
                times = times.to(self.device)
                
                # Forward pass
                if self.config.model_type in ["STRAP", "MAESTRAP"]:
                    outputs = self.model(
                        images=images,
                        masks=masks,
                        factors=risk_factors,
                        time=times,
                        event=events,
                        mask_ratio=0.0  # No masking during validation
                    )
                else:  # CNNSTRAP
                    outputs = self.model(
                        images=images,
                        masks=masks,
                        factors=risk_factors,
                        time=times,
                        event=events
                    )
                
                loss = outputs["loss"]
                cox_loss = outputs["cox_loss"]
                
                val_losses.append(loss.item())
                cox_losses.append(cox_loss.item())
        
        avg_val_loss = float(np.mean(val_losses))
        avg_cox_loss = float(np.mean(cox_losses))
        
        return {
            "val_loss": avg_val_loss,
            "val_cox_loss": avg_cox_loss
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.console.print(f"[green]New best model saved! Validation loss: {self.best_val_loss:.6f}[/green]")
    
    def train(self):
        """Main training loop."""
        self.console.print(Panel(
            f"[bold cyan]Starting Training[/bold cyan]\n"
            f"Model: {self.config.model_type}\n"
            f"Epochs: {self.config.epochs}\n"
            f"Batch size: {self.config.batch_size}\n"
            f"Learning rate: {self.config.learning_rate}",
            title="Training Configuration"
        ))
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record losses
            self.train_losses.append(train_metrics["train_loss"])
            self.val_losses.append(val_metrics["val_loss"])
            
            # Check if best model
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
            
            # Log metrics
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["train_loss"],
                    "train_cox_loss": train_metrics["train_cox_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "val_cox_loss": val_metrics["val_cox_loss"],
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "best_val_loss": self.best_val_loss
                })
            
            # Print epoch summary
            table = Table(title=f"Epoch {epoch + 1}/{self.config.epochs}")
            table.add_column("Metric", style="cyan")
            table.add_column("Train", style="green")
            table.add_column("Validation", style="yellow")
            
            table.add_row("Loss", f"{train_metrics['train_loss']:.6f}", f"{val_metrics['val_loss']:.6f}")
            table.add_row("Cox Loss", f"{train_metrics['train_cox_loss']:.6f}", f"{val_metrics['val_cox_loss']:.6f}")
            table.add_row("LR", f"{self.optimizer.param_groups[0]['lr']:.2e}", "")
            
            self.console.print(table)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
        
        self.console.print(Panel(
            f"[bold green]Training Complete![/bold green]\n"
            f"Best validation loss: {self.best_val_loss:.6f}\n"
            f"Model saved to: {self.config.checkpoint_dir}",
            title="Training Summary"
        ))
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train STRAP model")
    parser.add_argument("--model", type=str, default="STRAP", choices=["STRAP", "MAESTRAP", "CNNSTRAP"],
                      help="Model type to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of dataset samples")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embed_dim=args.embed_dim,
        n_samples=args.n_samples,
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()