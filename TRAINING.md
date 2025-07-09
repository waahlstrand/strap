# STRAP Training Guide

This guide explains how to train the STRAP models using the provided training scripts.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

#### Option A: Using the main training script
```bash
# Train STRAP model
python train.py --model STRAP --epochs 50 --batch_size 8

# Train MAESTRAP model  
python train.py --model MAESTRAP --epochs 50 --batch_size 8

# Train CNNSTRAP model
python train.py --model CNNSTRAP --epochs 50 --batch_size 16
```

#### Option B: Using the simplified runner
```bash
# Train STRAP model
python run_training.py --model STRAP

# Train MAESTRAP model
python run_training.py --model MAESTRAP

# Train CNNSTRAP model
python run_training.py --model CNNSTRAP
```

## Training Configuration

The training script supports various configuration options:

### Model Parameters
- `--model`: Model type (STRAP, MAESTRAP, CNNSTRAP)
- `--embed_dim`: Embedding dimension (default: 768)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)

### Dataset Parameters
- `--n_samples`: Number of samples in the mock dataset (default: 1000)

### Logging and Checkpoints
- `--use_wandb`: Enable Weights & Biases logging
- `--checkpoint_dir`: Directory to save checkpoints (default: "checkpoints")

## Model Descriptions

### STRAP
- Full STRAP model with masked autoencoder and survival analysis
- Uses only the encoder during inference (no reconstruction loss)
- Focuses on survival prediction

### MAESTRAP  
- STRAP model with both autoencoder and survival losses
- Uses both reconstruction and Cox partial likelihood losses
- Good for pretraining and joint learning

### CNNSTRAP
- CNN-based survival model using ResNet50 backbone
- Simpler architecture, faster training
- Good baseline model

## Output

During training, you'll see:
- Rich formatted progress bars and tables
- Training and validation metrics
- Model checkpoints saved to the specified directory
- Best model saved when validation loss improves

## Example Output

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                Model Info              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
STRAP Model Initialized
Total parameters: 85,123,456
Trainable parameters: 85,123,456
Device: cuda

Epoch 1/50 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                 Epoch 1/50                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
┏━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric  ┃ Train        ┃ Validation     ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Loss    │ 2.456789     │ 2.123456       │
│ Cox Loss│ 2.456789     │ 2.123456       │
│ LR      │ 1.00e-04     │                │
└─────────┴──────────────┴────────────────┘
```

## Tips

1. **Start Small**: Begin with smaller models and datasets to test the setup
2. **Monitor GPU Memory**: Reduce batch size if you encounter OOM errors
3. **Experiment with Hyperparameters**: Try different learning rates and model sizes
4. **Use Checkpoints**: Training can be resumed from saved checkpoints
5. **Enable Logging**: Use `--use_wandb` for detailed experiment tracking (requires wandb installation)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model dimensions
2. **Slow Training**: Ensure CUDA is available and being used
3. **Import Errors**: Install missing dependencies with pip
4. **Wandb Issues**: Disable wandb logging with `--use_wandb False` if not needed

### Performance Tips

- Use mixed precision training for faster training (modify the script to add autocast)
- Increase num_workers for faster data loading
- Use larger batch sizes if GPU memory allows
- Consider using gradient accumulation for effective larger batch sizes
