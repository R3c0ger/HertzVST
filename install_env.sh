#!/bin/bash
# HertzVST Virtual Environment Installation Script

# Initialize conda (if not already)
eval "$(conda shell.bash hook)"

echo "Removing old environment (if exists)..."
conda env remove -n HertzVST -y

echo "Creating new environment..."
conda env create -f environment.yaml

echo "Activating environment and installing PyTorch CUDA version..."
conda activate HertzVST

# Install PyTorch CUDA version (using official index)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "Environment installation complete!"
echo ""
echo "Use the following commands to activate the environment:"
echo "  source ~/.bashrc  # if conda is not yet loaded"
echo "  conda activate HertzVST"