#!/bin/bash
set -e

echo "Updating package lists..."
sudo apt-get update -y

echo "Installing python3-venv and pip..."
sudo apt-get install -y python3-venv python3-pip

echo "Creating Python virtual environment in current directory (./venv)..."
python3 -m venv ./venv

echo "Upgrading pip inside virtual environment..."
./venv/bin/pip install --upgrade pip

echo "Installing PyTorch packages..."
./venv/bin/pip install torch torchvision torchaudio

echo "Setup complete!"