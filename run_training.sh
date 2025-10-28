#!/bin/bash

# Ensure wandb is logged in
# wandb login [your_api_key]

echo "Starting training run..."

# Create a dummy image for the placeholder to "load"
mkdir -p data
touch data/input_person.jpg

# Install requirements
pip install -r requirements.txt

# Run the training script
python train.py

echo "Training complete. Model saved to trained_cloth_params.pth"