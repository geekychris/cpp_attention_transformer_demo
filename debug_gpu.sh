#!/bin/bash

echo "Starting GPU debug test..."
echo "Running: ./train_model --epochs 1 --batch_size 1 --lr 0.001 --gpu"
echo ""

# Start the training in the background and capture its PID
./train_model --epochs 1 --batch_size 1 --lr 0.001 --gpu &
TRAIN_PID=$!

echo "Training started with PID: $TRAIN_PID"

# Monitor for 10 seconds
for i in {1..10}; do
    echo "Waiting... ($i/10 seconds)"
    sleep 1
    
    # Check if process is still running
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "Training process completed normally"
        wait $TRAIN_PID
        exit 0
    fi
done

echo "Process still running after 10 seconds, killing..."
kill -TERM $TRAIN_PID 2>/dev/null
sleep 2

# Force kill if still running
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Force killing..."
    kill -KILL $TRAIN_PID 2>/dev/null
fi

echo "Process terminated"