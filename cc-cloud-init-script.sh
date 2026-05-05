#!/bin/bash
set -e

echo "Updating package lists..."
apt-get update -y

echo "Installing python3-venv and pip..."
apt-get install -y python3-venv python3-pip

echo "Creating Python virtual environment at /opt/pyenv..."
python3 -m venv /opt/pyenv

echo "Upgrading pip inside virtual environment..."
/opt/pyenv/bin/pip install --upgrade pip

echo "Installing PyTorch packages..."
/opt/pyenv/bin/pip install torch torchvision torchaudio

echo "Setup complete!"