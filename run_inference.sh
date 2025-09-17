#!/bin/bash

echo "=== Transformer Inference Pipeline ==="
echo

# Check for available models
models=()
if [[ -f "demo_model.bin" ]]; then
    models+=("demo_model.bin")
fi
if [[ -f "model.bin" ]]; then
    models+=("model.bin")
fi

# Find other .bin files
for file in *.bin; do
    if [[ -f "$file" && "$file" != "demo_model.bin" && "$file" != "model.bin" ]]; then
        models+=("$file")
    fi
done

if [[ ${#models[@]} -eq 0 ]]; then
    echo "No trained models found (.bin files)."
    echo "Please run training first:"
    echo "  ./run_training.sh"
    echo "  or"
    echo "  ./train_demo"
    exit 1
fi

echo "Available models:"
for i in "${!models[@]}"; do
    echo "$((i+1)). ${models[i]}"
done
echo

read -p "Choose model (1-${#models[@]}): " model_choice

if [[ $model_choice -lt 1 || $model_choice -gt ${#models[@]} ]]; then
    echo "Invalid choice. Using first model: ${models[0]}"
    selected_model="${models[0]}"
else
    selected_model="${models[$((model_choice-1))]}"
fi

echo "Selected model: $selected_model"
echo

echo "Inference options:"
echo "1. Interactive mode (type prompts manually)"
echo "2. Batch mode with test prompts"
echo "3. Batch mode with custom prompts file"
echo

read -p "Choose option (1-3): " inference_choice

case $inference_choice in
    1)
        echo "Starting interactive mode..."
        echo "Type your prompts and press Enter. Type 'quit' to exit."
        echo
        ./inference --model "$selected_model" --interactive
        ;;
    2)
        if [[ -f "test_prompts.txt" ]]; then
            echo "Running batch inference with test_prompts.txt..."
            echo
            ./inference --model "$selected_model" --prompts test_prompts.txt
        else
            echo "test_prompts.txt not found. Please generate dataset first:"
            echo "  python3 generate_dataset.py"
            exit 1
        fi
        ;;
    3)
        read -p "Enter prompts file path: " prompts_file
        if [[ -f "$prompts_file" ]]; then
            echo "Running batch inference with $prompts_file..."
            echo
            ./inference --model "$selected_model" --prompts "$prompts_file"
        else
            echo "File not found: $prompts_file"
            exit 1
        fi
        ;;
    *)
        echo "Invalid option. Running with test prompts..."
        if [[ -f "test_prompts.txt" ]]; then
            ./inference --model "$selected_model" --prompts test_prompts.txt
        else
            echo "test_prompts.txt not found. Starting interactive mode..."
            ./inference --model "$selected_model" --interactive
        fi
        ;;
esac

echo
echo "Inference completed!"