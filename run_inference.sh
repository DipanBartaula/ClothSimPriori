#!/bin/bash

echo "Starting inference run..."

# Check if model weights exist
if [ ! -f "trained_cloth_params.pth" ]; then
    echo "Error: trained_cloth_params.pth not found."
    echo "Please run ./run_training.sh first."
    exit 1
fi

# Run the inference script
python inference.py \
    --weights "trained_cloth_params.pth" \
    --image "data/input_person.jpg" \
    --output "output_simulation.mp4"\
    --prompt "a person jumping up and down"

echo "Inference complete. Video saved to output_simulation.mp4"