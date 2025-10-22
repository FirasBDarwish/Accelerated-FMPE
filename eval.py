#!/usr/bin/env python3
"""
Evaluate a trained DINGO posterior model on SBIBM.

This script is designed to run *alongside* your training artifacts and produce
all the figures / CSVs you asked for (C2ST, KSD, MMD, PM error, PVR, median distance,
scatter plots and log-prob histograms; plus an optional validation loss pass).

Inputs:
  --settings <PATH>       Path to the settings.yaml used for the trained model.
  --model <PATH>          Path to the trained model file (best_model.pt).
  --dataset_dir <PATH>    Path to the dataset directory saved during training
                          (must contain x.npy, theta.npy, meta.yaml as produced by your training script).
  --out_dir <PATH>        Directory to write plots and CSVs (defaults to model's directory if omitted).
  --metrics <...>         Optional list of metrics to compute. Defaults to
                          c2st ksd mmd posterior_mean_error posterior_variance_ratio median_distance
  --max_obs N             Number of SBIBM observations to evaluate (default: 9 to match your example).
  --compute_val_loss      If set, also runs a validation pass and writes val_losses.csv.
  --device <cpu|cuda|mps> Force a device (defaults to auto-detect like your training code).

Example:
  python evaluate_sweep_best.py \
      --settings /path/to/.../settings_abcdef12.yaml \
      --model    /path/to/.../runs/cfg_abcdef12/best_model.pt \
      --dataset_dir /path/to/.../dataset \
      --out_dir  /path/to/.../runs/cfg_abcdef12 \
      --compute_val_loss
"""

from __future__ import annotations
import argparse, csv, math, os
from os.path import join, dirname
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch, yaml, pandas as pd, matplotlib.pyplot as plt
import sbibm.tasks
from sbibm.metrics import c2st, median_distance, posterior_variance_ratio, posterior_mean_error, mmd
from dingo.core.posterior_models.build_model import build_model_from_kwargs, autocomplete_model_kwargs
from dingo.core.posterior_models.base_model import test_epoch
from collections import OrderedDict

# ------------------ Dataset ------------------
class SbiDataset(torch.utils.data.Dataset):
    def __init__(self, theta: torch.Tensor, x: torch.Tensor):
        super().__init__()
        print(f"[Dataset] Initializing dataset with θ∈{theta.shape}, x∈{x.shape}")
        self.standardization = {
            "x": {"mean": torch.mean(x, dim=0), "std": torch.std(x, dim=0)},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")
        print(f"[Dataset] Standardization complete.")

    def standardize(self, sample: torch.Tensor, label: str, inverse: bool = False) -> torch.Tensor:
        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]
        mean = mean.to(device=sample.device, dtype=sample.dtype)
        std = std.to(device=sample.device, dtype=sample.dtype)
        if not inverse:
            return (sample - mean) / std
        else:
            return sample * std + mean

    def __len__(self): return len(self.theta)
    def __getitem__(self, idx): return self.theta[idx], self.x[idx]

def load_dataset_from_dir(dataset_dir: str) -> SbiDataset:
    print(f"[Load] Loading dataset from {dataset_dir}")
    x = np.load(join(dataset_dir, "x.npy"))
    theta = np.load(join(dataset_dir, "theta.npy"))
    x = torch.tensor(x, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    return SbiDataset(theta, x)

# ------------------ Utility ------------------
def pick_device(user_pref: str | None = None) -> str:
    if user_pref: return user_pref
    # if torch.cuda.is_available(): return "cuda"
    return "cpu"

def build_train_and_val_loaders(dataset, train_fraction, batch_size, num_workers, seed=42):
    print(f"[DataLoader] Building train/val loaders (train_fraction={train_fraction}, batch_size={batch_size})")
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, drop_last=False)
    print(f"[DataLoader] Train={train_size} samples, Val={val_size} samples.")
    return train_loader, val_loader

# ------------------ Plot helpers ------------------
def plot_posteriors_and_log_probs(reference_samples, posterior_samples,
                                  reference_log_probs, posterior_log_probs, out_dir: str):
    print(f"[Plot] Saving posterior/log-prob plots to {out_dir}")
    plt.hist(
        posterior_log_probs.cpu(),
        bins='fd',            
        density=True,         
        alpha=0.5,
        label="posterior log probs",
        color='C0'
    )
    
    plt.hist(
        reference_log_probs.cpu(),
        bins='fd',            
        density=True,
        alpha=0.5,
        label="reference log probs",
        color='C1'
    )
    plt.legend()
    plt.savefig(join(out_dir, "log_probs.png"))
    plt.clf()

    plt.scatter(posterior_samples[:, 0].cpu(), posterior_samples[:, 1].cpu(),
                s=0.5, alpha=0.2, label="flow matching")
    plt.scatter(reference_samples[:, 0].cpu(), reference_samples[:, 1].cpu(),
                s=0.5, alpha=0.2, label="reference")
    plt.legend()
    plt.savefig(join(out_dir, "posteriors.png"))
    plt.clf()

# ------------------ Core evaluation ------------------
def complete_model_evaluation(out_dir: str, settings: Dict, dataset: SbiDataset, model,
                              device: str, metrics: List[str], max_obs: int = 9,
                              save_samples: bool = True):
    print(f"[Eval] Starting model evaluation on device={device} using up to {max_obs} observations.")
    task = sbibm.get_task(settings["task"]["name"])
    max_batch_size = settings["task"].get("max_batch_size", 500)
    metrics_dict = {
        "c2st": c2st, "mmd": mmd,
        "posterior_mean_error": posterior_mean_error,
        "posterior_variance_ratio": posterior_variance_ratio,
        "median_distance": median_distance,
    }
    metrics = [m for m in metrics if m in metrics_dict]
    print(f"[Eval] Metrics to compute: {metrics}")
    result_list: List[Dict] = []

    for obs in range(1, max_obs + 1):
        print(f"\n[Eval] === Observation {obs}/{max_obs} ===")
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        num_samples = len(reference_samples)
        print(f"[Eval] Loaded {num_samples} reference samples.")
        reference_samples_standardized = dataset.standardize(reference_samples, label="theta")
        observation = dataset.standardize(task.get_observation(num_observation=obs), label="x").to(device)

        # log-probs for reference samples
        print("[Eval] Computing log-probs for reference samples...")
        reference_log_probs = []
        for i in range(math.ceil(num_samples / max_batch_size)):
            reference_batch = reference_samples_standardized[(i * max_batch_size):((i + 1) * max_batch_size)].to(device)
            lp = model.log_prob_batch(reference_batch, observation.repeat((len(reference_batch), 1))).detach()
            reference_log_probs.append(lp)
        reference_log_probs = torch.cat(reference_log_probs, dim=0)

        # generate posterior samples
        print("[Eval] Generating posterior samples from model...")
        posterior_samples, posterior_log_probs = [], []
        for _ in range(2 * num_samples // max_batch_size + 1):
            ps, lp = model.sample_and_log_prob_batch(observation.repeat((max_batch_size, 1)))
            posterior_samples.append(ps.detach())
            posterior_log_probs.append(lp.detach())
        posterior_samples = torch.cat(posterior_samples, dim=0)
        posterior_log_probs = torch.cat(posterior_log_probs, dim=0)

        print("[Eval] Unstandardizing samples...")
        posterior_samples = dataset.standardize(posterior_samples, label="theta", inverse=True)

        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples))
        out_ratio = (1 - torch.sum(prior_mask) / len(prior_mask)) * 100
        print(f"[Eval] {out_ratio:.2f}% of samples outside prior support. Discarding.")
        posterior_samples = posterior_samples[prior_mask]
        posterior_log_probs = posterior_log_probs[prior_mask]

        n = min(len(reference_samples), len(posterior_samples))
        if len(reference_samples) > len(posterior_samples):
            print("[Eval] Warning: fewer posterior samples than reference samples.")
        posterior_samples = posterior_samples[:n].detach()
        posterior_log_probs = posterior_log_probs[:n].detach()
        reference_samples = reference_samples[:n].detach()
        reference_log_probs = reference_log_probs[:n].detach()

        if obs == 1:
            plot_posteriors_and_log_probs(reference_samples, posterior_samples,
                                          reference_log_probs, posterior_log_probs, out_dir)

        print("[Eval] Computing metrics...")
        result = {"num_observation": obs}
        for m in metrics:
            score = metrics_dict[m](posterior_samples, reference_samples).item()
            result[m] = float(score)
            print(f"    {m}: {score:.4f}")
        result_list.append(result)

        if save_samples:
            dir_obs = join(out_dir, str(obs).zfill(2))
            Path(dir_obs).mkdir(exist_ok=True)
            print(f"[Eval] Saving samples to {dir_obs}")
            np.save(join(dir_obs, "samples.npy"), posterior_samples.cpu().numpy())
            np.save(join(dir_obs, "posterior_log_probs.npy"), posterior_log_probs.cpu().numpy())
            np.save(join(dir_obs, "reference_log_probs.npy"), reference_log_probs.cpu().numpy())

    if result_list:
        results_csv = join(out_dir, "results.csv")
        print(f"[Eval] Writing results to {results_csv}")
        with open(results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result_list[0].keys()))
            w.writeheader(); w.writerows(result_list)
    print("[Eval] Evaluation finished.")

# ------------------ Validation loss ------------------
def compute_validation_loss(model, val_loader, out_dir: str):
    print(f"[Val] Computing validation loss...")
    if hasattr(model, "time_prior_exponent"): model.time_prior_exponent = 0
    val_loss = test_epoch(model, val_loader)
    print(f"[Val] Mean val loss: {val_loss:.6f}")

    log_probs = []
    with torch.no_grad():
        model.network.eval()
        for batch_idx, data in enumerate(val_loader):
            data = [d.to(model.device, non_blocking=True) for d in data]
            log_probs.append(model.log_prob_batch(data[0], *data[1:]))
    log_probs = torch.cat(log_probs, dim=0)
    df = pd.DataFrame.from_records([{
        "val_loss": float(val_loss),
        "val_mean_log_likelihood": float(torch.mean(log_probs).item()),
        "val_median_log_likelihood": float(torch.median(log_probs).item()),
    }])
    out_csv = join(out_dir, "val_losses.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Val] Saved validation stats to {out_csv}")

# ------------------ CLI ------------------
def main():
    p = argparse.ArgumentParser(description="Verbose DINGO SBIBM evaluator.")
    default_metrics = ["c2st","ksd","mmd","posterior_mean_error","posterior_variance_ratio","median_distance"]
    p.add_argument("--settings", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--metrics", nargs="+", default=default_metrics)
    p.add_argument("--max_obs", type=int, default=9)
    p.add_argument("--compute_val_loss", action="store_true")
    p.add_argument("--device", default=None, choices=["cpu","cuda","mps"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = args.out_dir or dirname(args.model) or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[Main] Output directory: {out_dir}")

    with open(args.settings, "r") as f:
        settings = yaml.safe_load(f)
    device = pick_device(args.device)
    print(f"[Main] Using device: {device}")
    settings.setdefault("training", {})["device"] = device

    dataset = load_dataset_from_dir(args.dataset_dir)
    settings.setdefault("task", {})
    settings["task"]["dim_theta"] = int(dataset.theta.shape[1])
    settings["task"]["dim_x"] = int(dataset.x.shape[1])

    print(f"[Main] Building model from {args.model}")
    model = build_model_from_kwargs(filename=args.model, device=device)
    print("[Main] Model built successfully.")

    complete_model_evaluation(out_dir, settings, dataset, model, device,
                              metrics=args.metrics, max_obs=args.max_obs,
                              save_samples=True)

    if args.compute_val_loss:
        train_fraction = float(settings.get("training", {}).get("train_fraction", 0.95))
        batch_size = int(settings.get("training", {}).get("batch_size", 512))
        num_workers = int(settings.get("training", {}).get("num_workers", 0))
        _, val_loader = build_train_and_val_loaders(dataset, train_fraction, batch_size, num_workers, seed=args.seed)
        compute_validation_loss(model, val_loader, out_dir)

    print(f"[Main] ✅ Evaluation complete. Outputs written to: {out_dir}")

if __name__ == "__main__":
    main()