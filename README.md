# C++ Transformer Implementation

A complete C++ implementation of a Transformer neural network with training and inference capabilities.

## Features

- **Complete Transformer Architecture**: Multi-head attention, positional encoding, layer normalization, feed-forward networks
- **Training Pipeline**: Data loading, loss calculation, optimization, model checkpointing
- **Inference Engine**: Text generation with trained models
- **Tokenization**: Simple word-based tokenizer
- **Model Serialization**: Save and load trained models

## Quick Start

### 1. Build the Project

```bash
make clean && make
```

This builds all executables:
- `transformer_demo`: Basic demonstration
- `train_model`: Full training program
- `train_demo`: Simple training demo
- `inference`: Inference program

### 2. Generate Training Data

```bash
python3 generate_dataset.py
```

This downloads Shakespeare text and generates synthetic training data:
- `train_data.txt`: Training dataset (7200+ lines)
- `val_data.txt`: Validation dataset (800+ lines) 
- `test_prompts.txt`: Test prompts for inference

### 3. Train a Model

#### Option A: Quick Demo Training (Recommended)
```bash
./train_demo
```

#### Option B: Interactive Training Script
```bash
chmod +x run_training.sh
./run_training.sh
```

#### Option C: Manual Training
```bash
./train_model --epochs 5 --batch_size 4 --lr 0.001
```

### 4. Run Inference

#### Option A: Interactive Inference Script
```bash
chmod +x run_inference.sh
./run_inference.sh
```

#### Option B: Manual Inference
```bash
# Batch mode with test prompts
./inference --model demo_model.bin --prompts test_prompts.txt

# Interactive mode
./inference --model demo_model.bin --interactive
```

## Project Structure

```
├── transformer.h           # Core transformer architecture
├── training.h              # Training extensions and utilities
├── main.cpp               # Basic demo program
├── train.cpp              # Training program
├── train_demo.cpp         # Simple training demo
├── inference.cpp          # Inference program
├── training.cpp           # Training implementation
├── matrix_ops.cpp         # Matrix operations and activations
├── attention.cpp          # Attention mechanism
├── transformer_blocks.cpp # Transformer layers
├── tokenizer_embedding.cpp# Tokenizer and embeddings
├── transformer_model.cpp  # Main transformer model
├── generate_dataset.py    # Dataset generation script
├── run_training.sh        # Training helper script
├── run_inference.sh       # Inference helper script
├── Makefile              # Build configuration
└── README.md             # This file
```

## Architecture Details

### Model Configuration
- **Model Dimension**: 64-128 (configurable)
- **Attention Heads**: 2-4 (configurable)  
- **Layers**: 1-2 (configurable)
- **Feed-Forward Dimension**: 256-512 (configurable)
- **Vocabulary Size**: Dynamic based on training data

### Training Features
- Cross-entropy loss
- SGD optimizer with weight decay
- Gradient clipping
- Model checkpointing
- Validation monitoring

### Key Components

1. **Multi-Head Attention**: Scaled dot-product attention with multiple heads
2. **Positional Encoding**: Sinusoidal position embeddings
3. **Layer Normalization**: Pre-norm transformer architecture
4. **Feed-Forward Networks**: GELU activation functions
5. **Tokenization**: Word-based tokenizer with special tokens

## Usage Examples

### Training with Custom Parameters

```bash
./train_model --epochs 10 --batch_size 8 --lr 0.01 --model my_model.bin
```

### Inference Options

```bash
# Use specific model and prompts file
./inference --model my_model.bin --prompts custom_prompts.txt

# Interactive mode
./inference --model my_model.bin --interactive
```

### Building Specific Targets

```bash
make transformer_demo  # Build demo only
make train_model      # Build training program  
make inference        # Build inference program
make train_demo       # Build simple demo
```

## Makefile Targets

- `make all`: Build all programs
- `make clean`: Clean build files
- `make run`: Run the basic demo
- `make train`: Run training program
- `make infer`: Run inference program
- `make debug`: Build with debug flags
- `make release`: Build with optimization

## Model Files

- `demo_model.bin`: Model from simple training demo
- `model.bin`: Model from full training
- `model_epoch_N.bin`: Periodic training checkpoints

## Training Data

The dataset includes:
- **Shakespeare text**: Classic literature for language modeling
- **Synthetic stories**: Generated narrative patterns
- **Simple sentences**: Basic grammatical structures

Total: ~8000 training samples with variety in vocabulary and structure.

## Performance Notes

This is an educational implementation focused on clarity over performance. For production use, consider:
- GPU acceleration (CUDA/OpenCL)
- Optimized BLAS libraries
- More sophisticated optimization algorithms
- Proper gradient computation (currently simplified)

## Limitations

- Simplified backpropagation (educational version)
- CPU-only implementation
- Basic tokenization
- Small model sizes due to memory constraints

## Contributing

This is a demonstration project. For improvements:
1. Add proper gradient computation
2. Implement GPU support
3. Add more sophisticated tokenization
4. Include attention visualization
5. Add more training optimizations

## License

Educational/demonstration use. See individual components for specific licensing.