# Accelerated Flow Matching Posterior Estimation (FMPE)

This repository extends the **Flow Matching for Scalable Simulation-Based Inference (FMPE)** framework introduced in the NeurIPS-2023 paper [*Flow Matching for Scalable Simulation-Based Inference*](https://neurips.cc/virtual/2023/poster/72395).
It provides code to train and evaluate **flow-matching posterior estimators** on simulation-based inference (SBI) benchmarks and related tasks.

The implementation builds on the [**dingo**](https://github.com/dingo-gw/dingo) package for core posterior modeling and flow-matching mechanics.

---

## 🧩 Environment Setup
Before training or evaluation, create a Conda environment named fmpe and install all dependencies.

1. Create and activate the environment
```bash
conda create -n fmpe python=3.10 -y
conda activate fmpe
```
3. Install dependencies from dingo/pyproject.toml
Make sure you have pip ≥ 23 and setuptools ≥ 65:
```bash
pip install --upgrade pip setuptools wheel
```
Then install Dingo and all required dependencies (both regular and dev):
```bash
# From within your project root
cd dingo
pip install -e ".[dev]"
```
- 💡 This will install all runtime dependencies plus development tools (testing, linting, etc.) as specified in dingo/pyproject.toml.
  
You can confirm successful setup with:
```bash
python -c "import torch; print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

---

## ⚙️ Training

### Using SLURM (HPC)

Submit the training job via the provided script:

```bash
sbatch scripts/training.sh
```

This script:

* Allocates 2 GPUs and 6 CPUs with 128 GB memory for up to 48 hours
* Activates a Conda environment (`fmpe`)
* Prints environment diagnostics (Python path, CUDA, Torch)
* Runs the full training sweep:

```bash
python -m train_all \
  --root ./custom_training/training_fmpe \
  --base_settings ./custom_training/settings.yaml
```

Results are written to:

```
custom_training/training_fmpe/
  └── <task_name>/
       ├── dataset/
       ├── N_<budget>/
       │    ├── configs/
       │    ├── runs/
       │    ├── sweep.csv
       │    └── best.yaml
```

---

### Running Locally (no SLURM)

You can run the same process manually on your workstation:

```bash
conda activate fmpe

python -m train_all \
  --root ./custom_training/training_fmpe \
  --base_settings ./custom_training/settings.yaml \
  --tasks two_moons \
  --budgets 100000
```

> 🧩 *Tip:* Use smaller budgets (`--budgets 1000`) or smaller batch sizes for local runs.

---

### Hyperparameter Configuration

Hyperparameters for training are **defined and swept automatically** inside `train_all.py`.

To modify or fix them manually:

* Edit the `SWEEP_SPACE` dictionary for each dataset size (budget)
* Adjust the search space of:

  * `batch_sizes`
  * `learning_rates`
  * `max_width_pows` (controls network width)
  * `num_blocks` (network depth)
  * `time_prior_alphas` (time prior exponent)

Each configuration is hashed (`cfg_<id>`) and trained in its own run directory.
Validation loss and hyperparameters are automatically logged to `sweep.csv`.

Example of a single configuration’s output:

```
N_100000/runs/cfg_a1b2c3d4/
  ├── best_model.pt
  ├── history.txt
  ├── train_results.json
  └── settings.yaml
```

---

## 📈 Evaluation

### Using SLURM (HPC)

Submit the evaluation job via:

```bash
sbatch scripts/evaluation.sh
```

This script runs:

```bash
python -m eval \
  --settings ./custom_evaluation/1e5_fmpe/settings.yaml \
  --model ./custom_evaluation/1e5_fmpe/best_model.pt \
  --dataset_dir ./custom_training/training_fmpe/two_moons/dataset \
  --out_dir ./custom_evaluation/1e5_fmpe/result2
```

It reuses the Conda environment (`fmpe`) and the same diagnostic checks as training.

---

### Running Locally

```bash
conda activate fmpe

python -m eval \
  --settings ./custom_evaluation/1e5_fmpe/settings.yaml \
  --model ./custom_evaluation/1e5_fmpe/best_model.pt \
  --dataset_dir ./custom_training/training_fmpe/two_moons/dataset \
  --out_dir ./custom_evaluation/1e5_fmpe/result2
```

Outputs include posterior estimates, evaluation metrics, and visualizations (if enabled).

---

## 📄 References

* Wildberger et al., *Flow Matching for Scalable Simulation-Based Inference*, NeurIPS 2023
  [[Paper]](https://openreview.net/forum?id=D2cS6SoYlP)
* Dax et al., *Real-Time Gravitational Wave Science with Neural Posterior Estimation*, *Phys. Rev. Lett.* 127 (2021)

---
