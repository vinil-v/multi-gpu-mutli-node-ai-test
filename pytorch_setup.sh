#!/bin/bash
# 1. Create a dedicated directory structure on your NFS mount
mkdir -p /shared/apps/pytorch_env

# 2. Navigate to the installation directory
cd /shared/apps

# 3. Create a clean, isolated Python 3 virtual environment
python3 -m venv pytorch_env

# 4. Activate the newly created environment
source pytorch_env/bin/activate

# 5. Upgrade core packaging tools inside the environment
pip install --upgrade pip setuptools wheel

# 6. Install PyTorch with native CUDA 12 support (optimized for modern GPU hardware)
# Note: For Ubuntu 24.04 and modern compute nodes, cuda-12.1 or higher is standard.
pip install torch torchvision torchaudio 

# 7. (Optional but highly recommended) Verify the local installation sees your GPUs
python3 -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

# 8. Deactivate the environment once setup is complete
deactivate