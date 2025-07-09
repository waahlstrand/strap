#!/usr/bin/env python3
"""
Simple script to run training with predefined configurations.
"""

from train import TrainingConfig, Trainer


def run_strap_training():
    """Run STRAP model training with default configuration."""
    config = TrainingConfig(
        model_type="STRAP",
        epochs=50,
        batch_size=8,
        learning_rate=1e-4,
        embed_dim=384,  # Smaller for faster training
        img_size=64,
        patch_size=8,
        n_samples=500,  # Smaller dataset for testing
        use_wandb=False,  # Disable wandb by default
        checkpoint_dir="checkpoints_strap",
        log_interval=5,
        save_interval=10,
    )
    
    trainer = Trainer(config)
    trainer.train()


def run_maestrap_training():
    """Run MAESTRAP model training with default configuration."""
    config = TrainingConfig(
        model_type="MAESTRAP",
        epochs=50,
        batch_size=8,
        learning_rate=1e-4,
        embed_dim=384,
        img_size=64,
        patch_size=8,
        n_samples=500,
        use_wandb=False,
        checkpoint_dir="checkpoints_maestrap",
        log_interval=5,
        save_interval=10,
    )
    
    trainer = Trainer(config)
    trainer.train()


def run_cnnstrap_training():
    """Run CNNSTRAP model training with default configuration."""
    config = TrainingConfig(
        model_type="CNNSTRAP",
        epochs=50,
        batch_size=16,  # Can use larger batch size for CNN
        learning_rate=1e-4,
        embed_dim=1024,
        n_samples=500,
        use_wandb=False,
        checkpoint_dir="checkpoints_cnnstrap",
        log_interval=5,
        save_interval=10,
    )
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training for different STRAP models")
    parser.add_argument("--model", type=str, default="STRAP", 
                       choices=["STRAP", "MAESTRAP", "CNNSTRAP"],
                       help="Model type to train")
    
    args = parser.parse_args()
    
    if args.model == "STRAP":
        run_strap_training()
    elif args.model == "MAESTRAP":
        run_maestrap_training()
    elif args.model == "CNNSTRAP":
        run_cnnstrap_training()
    else:
        print(f"Unknown model type: {args.model}")
