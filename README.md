# 🩻 STRAP 

**STRAP** from the paper [*Separable tissue representations for attributable risk prediction*]() is a method for improved risk prediction and attribution of risk to pre-segmented regions-of-interet (ROIs), using a modified vision transformer architecture. The model is based on masked autoencoders, using only the patch tokens from each corresponding ROI as embeddings, drastically reducing the token sequence length.

The paper was accepted at the [MICCAI 2025]() conference, and a prepring version is available on [arXiv]().

## Installation

### Quick Install from GitHub
Either clone the repository, or preferably, fork it to modify it for your own project.

#### Using uv (fastest and recommended)
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install STRAP directly from GitHub
uv pip install git+https://github.com/waahlstrand/strap.git

# Or with optional dependencies
uv pip install "git+https://github.com/waahlstrand/strap.git[logging,dev]"
```

**Why uv?** It is fast, and I recommend it.

#### Using pip
```bash
pip install git+https://github.com/waahlstrand/strap.git

# Or with optional dependencies
pip install "git+https://github.com/waahlstrand/strap.git[logging,dev]"
```

### Development Installation

#### Using uv
```bash
# Clone the repository
git clone https://github.com/waahlstrand/strap.git
cd strap

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,logging]"
```

#### Using pip
```bash
# Clone the repository
git clone https://github.com/waahlstrand/strap.git
cd strap

# Install in development mode
pip install -e ".[dev,logging]"
```

## Quick Start

### 1. Run Training

#### Using the main training script
```bash
# Train STRAP model
python train.py --model STRAP --epochs 50 --batch_size 8

# Train MAESTRAP model  
python train.py --model MAESTRAP --epochs 50 --batch_size 8

# Train CNNSTRAP model
python train.py --model CNNSTRAP --epochs 50 --batch_size 16
```


## Training Configuration

The training script supports various configuration options:

### Command-Line Entry Points

After installing the package, you have access to several command-line tools:

- **`strap-train-strap`**: Quick STRAP training with optimized defaults
- **`strap-train-mae`**: Quick MAESTRAP training with optimized defaults  
- **`strap-train-cnn`**: Quick CNNSTRAP training with optimized defaults

### Model Parameters
- `--model`: Model type (STRAP, MAESTRAP, CNNSTRAP)
- `--embed_dim`: Embedding dimension (default: 768)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)

### Dataset Parameters
The project comes with a mock dataset, with completely random data, as a template to understand
how to format the inputs and targets.
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
- Masked CNN-based survival model using ResNet50 backbone
- Simpler architecture, faster training
- Good baseline model


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

