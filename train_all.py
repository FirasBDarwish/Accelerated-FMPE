import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from os.path import join
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np
import torch
import yaml

# ===== imports =====
import sbibm.tasks
from torch.utils.data import DataLoader, Subset
from typing import List

from dingo.core.posterior_models.build_model import (
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits

def parse_history_best_val(history_path: str) -> tuple[float, int, float]:
    """
    Parse history.txt (tab or space separated):
        epoch  train_loss  val_loss  lr
    Returns (best_val_loss, best_epoch, lr_at_best).
    Robust to scientific notation and mixed whitespace.
    """
    best_val = float("inf")
    best_epoch = -1
    lr_at_best = None
    if not os.path.exists(history_path):
        # if training crashed before logging
        return best_val, best_epoch, lr_at_best

    with open(history_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()  # tabs or spaces
            if len(parts) < 4:
                # tolerate odd lines
                continue
            try:
                epoch = int(float(parts[0]))
                train_loss = float(parts[1])
                val_loss = float(parts[2])
                lr = float(parts[3])
            except ValueError:
                continue

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                lr_at_best = lr

    return best_val, best_epoch, lr_at_best

# ====== bring in your dataset utilities from the training script ======
# If this file is separate, you can import them instead.
class SbiDataset(torch.utils.data.Dataset):
    def __init__(self, theta, x):
        super(SbiDataset, self).__init__()
        self.standardization = {
            "x": {"mean": torch.mean(x, dim=0), "std": torch.std(x, dim=0)},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")

    def standardize(self, sample, label, inverse=False):
        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]
        if not inverse:
            return (sample - mean) / std
        else:
            return sample * std + mean

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ------------------ spec: tasks, budgets, sweeps ------------------

SBIBM_TASKS = {
    # name used by sbibm -> (dim_x, conditioning_mode)
    # GLU on (t, θ) only for the two high-dim-x tasks
    # "gaussian_linear":                (2,   "concat"),
    # "gaussian_linear_uniform":        (2,   "concat"),
    # "gaussian_mixture":               (2,   "concat"),
    "two_moons":                      (2,   "concat"),
    # "slcp":                           (8,   "concat"),
    # "bernoulli_glm":                  (10,  "concat"),
    # "bernoulli_glm_raw":              (100, "glu"),
    # "slcp_distractors":               (100, "glu"),
    # "sir":                            (4,   "concat"),
    # "lotka_volterra":                 (5,   "concat"),
}

# BUDGETS = [1_000, 10_000, 100_000]
BUDGETS = [1_000, 100_000]

SWEEP_SPACE = {
    1_000: dict(
        batch_sizes=[8],             # 2^n, n in {2..5}
        max_width_pows=[5, 6],               # peak width = 2^n, n in {4..6}
        num_blocks=[10, 12],                    # 10..12
        learning_rates=[1e-3, 5e-4, 2e-4],
        time_prior_alphas=[-0.25, -0.5, 0, 1, 4],
    ),
    # 10_000: dict(
    #     batch_sizes=[16, 32, 64, 128],          # 2^n, n in {4..7}
    #     max_width_pows=[6, 7, 8],               # n in {6..8}
    #     num_blocks=[12, 14, 16],
    #     learning_rates=[1e-3, 5e-4, 2e-4, 1e-4],
    #     time_prior_alphas=[-0.25, -0.5, 0, 1, 4],
    # ),
    100_000: dict(
        # batch_sizes=[64, 128, 256, 512],        # 2^n, n in {6..9}
        batch_sizes=[1024],        # 2^n, n in {6..9}
        max_width_pows=[9, 10],              # n in {8..10}
        num_blocks=[16, 18],
        learning_rates=[5e-4, 2e-4, 1e-4],
        time_prior_alphas=[-0.25, -0.5, 0, 1, 4],
    ),
}

def diamond_hidden_dims(
    max_width_pow: int,
    num_blocks: int,
    min_pow: int = 4,
    peak_repeats: int | None = None,
) -> List[int]:
    """
    Build a diamond-shaped list of hidden dims with exactly num_blocks elements.
    - Widths are powers of two (2^k), k integer.
    - Ascend from 2^min_pow up to 2^max_width_pow, add a flat peak (peak_repeats),
      then descend.
    - If the result is too long, trim from the descending side starting just
      below the peak to preserve the 'fat top' look.

    Args:
        max_width_pow: exponent for the maximum width (e.g., 10 -> 1024).
        num_blocks: desired total number of layers.
        min_pow: exponent for the minimum width (default 4 -> 16).
        peak_repeats: how many times to repeat the peak width (default computed
                      to minimally reach num_blocks, but you can set it, e.g. 3).

    Returns:
        List[int] of length == num_blocks.
    """
    assert num_blocks >= 1, "num_blocks must be >= 1"
    assert max_width_pow >= min_pow, "max_width_pow must be >= min_pow"

    # Ascend powers
    up_pows = list(range(min_pow, max_width_pow + 1))   # e.g., [5,6,7,8,9,10]
    down_pows = up_pows[-2::-1]                         # mirror without peak
    base_pows = up_pows + down_pows                     # classic diamond
    base_len = len(base_pows)

    # Minimal peak repeats to *at least* reach num_blocks
    if peak_repeats is None:
        # base already has 1 peak; extra repeats needed:
        extra = max(0, num_blocks - base_len)
        peak_repeats = 1 + extra

    # Build with plateau at the peak
    max_pow = max_width_pow
    up_no_peak = up_pows[:-1]                           # everything before peak
    peak = [max_pow] * peak_repeats
    desc = down_pows[:]                                 # copy to trim if needed
    pows = up_no_peak + peak + desc

    # If too long, trim from the descending side starting just below the peak.
    # This removes the first element(s) after the plateau (e.g., 2^(max_pow-1), then 2^(max_pow-2), ...)
    overflow = len(pows) - num_blocks
    trim_idx = 0  # index into 'desc' to trim (0 = just below peak)
    while overflow > 0 and trim_idx < len(desc):
        # remove desc[trim_idx] from 'pows'
        to_remove = desc[trim_idx]
        # find first occurrence of that power after the plateau start
        # plateau ends at len(up_no_peak) + peak_repeats - 1
        plateau_end = len(up_no_peak) + peak_repeats
        # locate in pows[plateau_end:] and delete one
        for i in range(plateau_end, len(pows)):
            if pows[i] == to_remove:
                del pows[i]
                break
        overflow -= 1
        trim_idx += 1

    # If still too long (very edge cases), trim from the tail
    if len(pows) > num_blocks:
        pows = pows[:num_blocks]

    return [2 ** k for k in pows]

def conditioning_flags(mode: str) -> Dict[str, bool]:
    # GLU for (t, θ) when mode == "glu"; otherwise concat (t, θ, x)
    if mode == "glu":
        return {"context_with_glu": False, "theta_with_glu": True}
    return {"context_with_glu": False, "theta_with_glu": False}


# ------------------ IO helpers & bookkeeping ------------------

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def write_yaml(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cfg_id_for(hp: Dict[str, Any]) -> str:
    # stable short id from the hyperparameter dict
    blob = json.dumps(hp, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:8]


@dataclass
class RunRecord:
    task: str
    budget: int
    cfg_id: str
    run_dir: str
    val_loss: float
    hyperparams: Dict[str, Any]

    def as_row(self) -> Dict[str, Any]:
        row = {
            "task": self.task,
            "budget": self.budget,
            "cfg_id": self.cfg_id,
            "run_dir": self.run_dir,
            "val_loss": self.val_loss,
        }
        row.update({f"hp.{k}": v for k, v in self.hyperparams.items()})
        return row


def append_rows_csv(csv_path: str, rows: List[Dict[str, Any]]):
    new_file = not os.path.exists(csv_path)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------ dataset creation (one per task) ------------------

def ensure_max_dataset(task_name: str, ds_dir: str, max_samples: int, seed: int = 12345):
    """
    Create (or reuse) a fixed 100k-sample dataset for <task_name>.
    All budgets use subsets of this dataset so runs are comparable.
    """
    safe_makedirs(ds_dir)
    x_path, th_path, meta_path = join(ds_dir, "x.npy"), join(ds_dir, "theta.npy"), join(ds_dir, "meta.yaml")
    if os.path.exists(x_path) and os.path.exists(th_path) and os.path.exists(meta_path):
        return  # already there

    # seed everything for reproducibility
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()

    batch_size = 1024
    nr_batches = math.ceil(max_samples / batch_size)
    thetas = []
    xs = []
    for _ in range(nr_batches):
        theta_sample = prior(batch_size)
        x_sample = simulator(theta_sample)
        thetas.append(theta_sample)
        xs.append(x_sample)

    x = np.vstack(xs)[:max_samples]
    theta = np.vstack(thetas)[:max_samples]
    np.save(x_path, x)
    np.save(th_path, theta)
    write_yaml(meta_path, {"task": task_name, "num_samples": int(max_samples)})


def load_dataset_slice(ds_dir: str, n: int) -> SbiDataset:
    x = np.load(join(ds_dir, "x.npy"))[:n]
    theta = np.load(join(ds_dir, "theta.npy"))[:n]
    x = torch.tensor(x, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    return SbiDataset(theta, x)


# ------------------ training wrapper ------------------

def build_loaders(
    dataset: SbiDataset,
    train_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
):
    """
    Splits dataset into a train and validation set using a fixed random seed,
    and returns their respective DataLoaders.
    """
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use random_split for a robust, shuffled split
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one(run_dir: str, settings: Dict[str, Any], train_loader, val_loader, dataset: SbiDataset,) -> Tuple[float, str]:
    """
    Trains and returns (validation_loss, best_model_path).
    """
    # Autocomplete dims
    settings = dict(settings)  # shallow copy
    settings["training"]["device"] = pick_device()
    
    # --- THIS IS THE FIX ---
    # Access dimensions cleanly from the full dataset object
    # The .theta and .x attributes are standardized, but their shape is the same.
    settings["task"]["dim_theta"] = dataset.theta.shape[1]
    settings["task"]["dim_x"] = dataset.x.shape[1]
    
    # build model
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],
        context_dim=settings["task"]["dim_x"],
    )
    model = build_model_from_kwargs(
        settings={"train_settings": settings},
        device=settings["training"].get("device", "cpu"),
    )
    
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()

    # train
    runtime_limits = RuntimeLimits(epoch_start=0, max_epochs_total=settings["training"]["epochs"])
    model.train(
        train_loader,
        val_loader,
        train_dir=run_dir,
        runtime_limits=runtime_limits,
        early_stopping=True,
        use_wandb=False,
    )

    # after training, read history.txt for ranking
    history_path = join(run_dir, "history.txt")
    best_val, best_epoch, lr_at_best = parse_history_best_val(history_path)

    # load best (already produced by your trainer via early stopping)
    best_model_path = join(run_dir, "best_model.pt")

    # persist results for this config
    with open(join(run_dir, "train_results.json"), "w") as f:
        json.dump(
            {
                "best_val_loss": None if math.isinf(best_val) else float(best_val),
                "best_epoch": int(best_epoch),
                "lr_at_best": lr_at_best,
                "history_file": "history.txt",
            },
            f,
        )

    return (float(best_val) if not math.isinf(best_val) else 1e99), best_model_path

# ------------------ settings manipulation ------------------

def make_run_settings(base_settings: Dict[str, Any],
                      task_name: str,
                      budget: int,
                      conditioning_mode: str,
                      hp: Dict[str, Any]) -> Dict[str, Any]:
    """clone base settings.yaml and override only what we sweep."""
    cfg = yaml.safe_load(yaml.safe_dump(base_settings))  # deep copy via yaml

    # task overrides
    cfg["task"]["name"] = task_name
    cfg["task"]["num_train_samples"] = budget

    # training overrides
    cfg["training"]["batch_size"] = hp["batch_size"]
    cfg["training"]["optimizer"]["lr"] = hp["lr"]

    # model overrides
    pk = cfg["model"]["posterior_kwargs"]
    pk["hidden_dims"] = diamond_hidden_dims(
        max_width_pow=hp["max_width_pow"],
        num_blocks=hp["num_blocks"],
    )
    pk["number_of_blocks"] = hp["num_blocks"]
    pk["time_prior_exponent"] = hp["alpha"]

    # conditioning rule
    pk.update(conditioning_flags(conditioning_mode))

    return cfg


def iter_hyperparams_for_budget(budget: int) -> Iterable[Dict[str, Any]]:
    space = SWEEP_SPACE[budget]
    for bs in space["batch_sizes"]:
        for lr in space["learning_rates"]:
            for k in space["max_width_pows"]:
                for nb in space["num_blocks"]:
                    for alpha in space["time_prior_alphas"]:
                        yield {
                            "batch_size": bs,
                            "lr": lr,
                            "max_width_pow": k,
                            "num_blocks": nb,
                            "alpha": alpha,
                        }


# ------------------ main sweep orchestration ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory to store all sweeps")
    parser.add_argument("--base_settings", required=True, help="Path to the base settings.yaml")
    parser.add_argument("--tasks", nargs="*", default=list(SBIBM_TASKS.keys()),
                        help="Optional subset of sbibm task names to run")
    parser.add_argument("--budgets", nargs="*", type=int, default=BUDGETS,
                        help="Budgets to run (choose from 1000, 10000, 100000)")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    base_settings = read_yaml(args.base_settings)

    # sweep all tasks requested
    for task_name in args.tasks:
        assert task_name in SBIBM_TASKS, f"Unknown task: {task_name}"
        dim_x, cond_mode = SBIBM_TASKS[task_name]

        task_root = join(args.root, task_name)
        ds_dir = join(task_root, "dataset")
        safe_makedirs(task_root)

        # 1) ensure a fixed dataset (100k) for this task
        ensure_max_dataset(task_name, ds_dir, max_samples=100_000, seed=args.seed)

        # 2) run budgets
        for budget in sorted(set(args.budgets)):
            budget_root = join(task_root, f"N_{budget:06d}")
            runs_root = join(budget_root, "runs")
            cfgs_root = join(budget_root, "configs")
            for p in [budget_root, runs_root, cfgs_root]:
                safe_makedirs(p)

            # create dataset slice & loaders ONCE per budget; reused across configs
            dataset = load_dataset_slice(ds_dir, n=budget)
            # use the train_fraction from base settings so validation stays 5%
            train_fraction = base_settings["training"]["train_fraction"]
            # batch size differs per config; for the loader we’ll rebuild inside the loop

            sweep_csv = join(budget_root, "sweep.csv")
            all_rows: List[Dict[str, Any]] = []
            best_rec: RunRecord | None = None

            # iterate hyperparams
            for hp in iter_hyperparams_for_budget(budget):
                # create per-run settings
                run_settings = make_run_settings(base_settings, task_name, budget, cond_mode, hp)
                cfg_id = cfg_id_for(hp)

                run_dir = join(runs_root, f"cfg_{cfg_id}")
                safe_makedirs(run_dir)

                # write the per-run settings.yaml for traceability
                run_cfg_path = join(cfgs_root, f"settings_{cfg_id}.yaml")
                write_yaml(run_cfg_path, run_settings)

                # build loaders with hp["batch_size"] and the new seeded split
                train_loader, val_loader = build_loaders(
                    dataset=dataset,
                    train_fraction=train_fraction,
                    batch_size=hp["batch_size"],
                    num_workers=run_settings["training"]["num_workers"],
                    seed=args.seed,  # Pass the global seed for reproducibility
                )
                
                # train & score
                val_loss, _ = train_one(run_dir, run_settings, train_loader, val_loader, dataset)

                rec = RunRecord(
                    task=task_name,
                    budget=budget,
                    cfg_id=cfg_id,
                    run_dir=run_dir,
                    val_loss=val_loss,
                    hyperparams=hp,
                )
                all_rows.append(rec.as_row())

                # update best
                if (best_rec is None) or (val_loss < best_rec.val_loss):
                    best_rec = rec

                # append row immediately (robust to interruptions)
                append_rows_csv(sweep_csv, [rec.as_row()])

            # write best summary for this budget
            if best_rec is not None:
                write_yaml(join(budget_root, "best.yaml"), {
                    "task": best_rec.task,
                    "budget": best_rec.budget,
                    "cfg_id": best_rec.cfg_id,
                    "run_dir": best_rec.run_dir,
                    "val_loss": float(best_rec.val_loss),
                    "hyperparams": best_rec.hyperparams,
                })

    print("Sweep complete.")


if __name__ == "__main__":
    main()