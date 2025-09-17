#!/bin/bash

echo "=== Transformer Training Pipeline ==="
echo

# Check if data files exist
if [[ ! -f "train_data.txt" || ! -f "val_data.txt" ]]; then
    echo "Training data not found. Generating dataset..."
    python3 generate_dataset.py
    echo
fi

echo "Available training options:"
echo "1. Quick demo training (safe, fast)"
echo "2. Full training (may be unstable due to simplified implementation)"
echo "3. Custom training with parameters"
echo

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "Running quick demo training..."
        echo
        ./train_demo
        ;;
    2)
        echo "Running full training (5 epochs, small batch size)..."
        echo
        ./train_model --epochs 5 --batch_size 4 --lr 0.001
        ;;
    3)
        echo "Custom training parameters:"
        read -p "Epochs (default: 5): " epochs
        read -p "Batch size (default: 4): " batch_size
        read -p "Learning rate (default: 0.001): " lr
        
        epochs=${epochs:-5}
        batch_size=${batch_size:-4}
        lr=${lr:-0.001}
        
        echo "Running training with epochs=$epochs, batch_size=$batch_size, lr=$lr"
        ./train_model --epochs $epochs --batch_size $batch_size --lr $lr
        ;;
    *)
        echo "Invalid option. Running demo training..."
        ./train_demo
        ;;
esac

echo
echo "Training completed!"

# Check if model was created
if [[ -f "demo_model.bin" ]]; then
    echo "Model saved as demo_model.bin"
    echo
    echo "You can now run inference with:"
    echo "  ./inference --model demo_model.bin --prompts test_prompts.txt"
    echo "  ./inference --model demo_model.bin --interactive"
elif [[ -f "model.bin" ]]; then
    echo "Model saved as model.bin"
    echo
    echo "You can now run inference with:"
    echo "  ./inference --model model.bin --prompts test_prompts.txt"
    echo "  ./inference --model model.bin --interactive"
fi