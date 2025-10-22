#!/bin/bash

#SBATCH -J evaluating_fmpe  # Job name
#SBATCH -o ./out_err/evaluating_fmpe.out  # Output file
#SBATCH -e ./out_err/evaluating_fmpe.err  # Error file
#SBATCH -t 48:00:00  # Maximum runtime
#SBATCH -n 1 # (1 task)
#SBATCH -N 1 # (1 node) 
#SBATCH -c 6 # (6 CPUS per task)
#SBATCH --mem=128G  # Memory allocation (128GB, adjust as needed)

# Load required modules
module purge

# Activate virtual environment (if needed)
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate fmpe

# --- Confirm the environment ---
echo ">>> Conda environment info:"
conda info | grep "active environment"
which python
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import torch; print('Torch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# Run the Python script
python -m eval \
  --settings ./custom_evaluation/1e5_fmpe/settings.yaml \
  --model    ./custom_evaluation/1e5_fmpe/best_model.pt \
  --dataset_dir ./custom_training/training_fmpe/two_moons/dataset \
  --out_dir   ./custom_evaluation/1e5_fmpe/result2\