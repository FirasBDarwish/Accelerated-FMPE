#!/bin/bash

#SBATCH -J training_fmpe  # Job name
#SBATCH -o ./out_err/training_fmpe.out  # Output file
#SBATCH -e ./out_err/training_fmpe.err  # Error file
#SBATCH -t 48:00:00  # Maximum runtime
#SBATCH -p nvidia  # Partition name
#SBATCH --gres=gpu:2  # Request 2 GPUs
#SBATCH -n 1 # (1 task)
#SBATCH -N 1 # (1 node) 
#SBATCH -c 6 # (6 CPUS per task)
#SBATCH --mem=128G  # Memory allocation (128GB, adjust as needed)

# Load required modules
module purge

# Activate virtual environment (if needed)
conda activate fmpe

# --- Confirm the environment ---
echo ">>> Conda environment info:"
conda info | grep "active environment"
which python
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import torch; print('Torch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# Run the Python script
python -m train_all --root ./custom_training/training_fmpe --base_settings ./custom_training/settings.yaml # Train
