"""
Boundary-consistency fine-tuning for domain-decomposed APMGSRN (Seamless APMGSRN, part 2).

Copies an ensemble checkpoint to a new name, freezes every brick's adaptive feature grids
and grid transforms, and fine-tunes only the small MLP decoders with

    L = L_rec(overlap) + L_rec(interior anchor)
        + ramp * value_weight    * mean (f_i(x) - f_j(x))^2           / range^2
        + ramp * gradient_weight * mean (d_n f_i(x) - d_n f_j(x))^2   / range^2

for every face-adjacent pair (i, j) and points x in their shared ghost-cell overlap.

Example:
  python Code/finetune_boundaries.py --load_from Plume_APMGSRN_dd_g4 \
      --save_name Plume_APMGSRN_dd_g4_consistency --device cuda:0
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Models.options import load_options, save_options          # noqa: E402
from Models.models import load_model                            # noqa: E402
from Models.ensemble_utils import (parse_int_tuple, enumerate_face_adjacencies,  # noqa: E402
    intersect_extents_dhw, voxel_to_normalized)
from Other.utility_functions import str2bool, nc_to_tensor     # noqa: E402

project_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
data_folder = os.path.join(project_folder_path, "Data")
save_folder = os.path.join(project_folder_path, "SavedModels")


def clone_checkpoint(src_name, dst_name, overwrite=False):
    src = os.path.join(save_folder, src_name)
    dst = os.path.join(save_folder, dst_name)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Source ensemble {src} does not exist")
    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
        raise ValueError("Source and destination must differ; baseline checkpoints are never overwritten")
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"{dst} exists. Pass --overwrite true to replace it.")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    parent = load_options(dst)
    parent["save_name"] = dst_name
    save_options(parent, dst)
    for sub in sorted(os.listdir(dst)):
        sub_path = os.path.join(dst, sub)
        if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "options.json")):
            so = load_options(sub_path)
            so["save_name"] = f"{dst_name}/{sub}"
            save_options(so, sub_path)
    return dst


def load_full_volume(opt, device):
    o = dict(opt)
    o["extents"] = None
    d, _ = nc_to_tensor(os.path.join(data_folder, opt["data"]), o)
    return d.float().to(device)


def sample_ground_truth(volume, x_xyz):
    """Mirror Dataset.get_random_points: trilinear, align_corners=True, XYZ coordinates."""
    g = x_xyz.to(volume.device).reshape(1, 1, 1, -1, 3)
    v = F.grid_sample(volume, g, mode="bilinear", align_corners=True)
    return v.reshape(volume.shape[1], -1).transpose(0, 1)


def sample_box(extents_dhw, full_shape, n, gen, device):
    """Uniform continuous points inside end-exclusive voxel extents, returned in normalized XYZ."""
    cols = []
    for a in range(3):
        lo, hi = float(extents_dhw[2 * a]), float(extents_dhw[2 * a + 1] - 1)
        u = torch.rand(n, generator=gen) * (hi - lo) + lo
        cols.append(voxel_to_normalized(u, full_shape[a]))
    return torch.stack(cols[::-1], dim=-1).to(device)   # DHW -> XYZ


def build_pairs(ens, grid_dhw, full_shape):
    pairs = enumerate_face_adjacencies(grid_dhw)
    for p in pairs:
        p["overlap_dhw"] = list(intersect_extents_dhw(ens.support_extents_dhw[p["left"]],
                                                      ens.support_extents_dhw[p["right"]]))
        a = p["normal_axis_dhw"]
        p["delta"] = 2.0 / (full_shape[a] - 1)
        p["lo_xyz"] = voxel_to_normalized(float(p["overlap_dhw"][2 * a]), full_shape[a])
        p["hi_xyz"] = voxel_to_normalized(float(p["overlap_dhw"][2 * a + 1] - 1), full_shape[a])
    return pairs


def normal_derivative(ens, idx, x, axis_xyz, delta, lo, hi):
    xp, xm = x.clone(), x.clone()
    xp[:, axis_xyz] = (x[:, axis_xyz] + delta).clamp(lo, hi)
    xm[:, axis_xyz] = (x[:, axis_xyz] - delta).clamp(lo, hi)
    h = (xp[:, axis_xyz] - xm[:, axis_xyz]).clamp_min(1e-12).unsqueeze(-1)
    return (ens.forward_child(xp, idx) - ens.forward_child(xm, idx)) / h


def main():
    ap = argparse.ArgumentParser(description="Interface-consistency fine-tuning of ensemble decoders")
    ap.add_argument("--load_from", required=True)
    ap.add_argument("--save_name", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data_device", default="cpu")
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--points_per_iteration", type=int, default=4096, help="overlap points per pair")
    ap.add_argument("--interior_points", type=int, default=2048, help="anchor points per brick")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--value_weight", type=float, default=1.0)
    ap.add_argument("--gradient_weight", type=float, default=0.0)
    ap.add_argument("--warmup_iterations", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_minutes", type=float, default=20.0)
    ap.add_argument("--max_gpu_gb", type=float, default=3.7)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--overwrite", type=str2bool, default=False)
    args = vars(ap.parse_args())

    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    gen = torch.Generator().manual_seed(args["seed"])
    device = args["device"]
    use_cuda = "cuda" in device

    dst = clone_checkpoint(args["load_from"], args["save_name"], args["overwrite"])
    opt = load_options(dst)
    opt["device"] = device
    opt["data_device"] = args["data_device"]
    opt["ensemble_routing_mode"] = "hard"
    full_shape = [int(v) for v in opt["full_shape"]]
    grid_dhw = parse_int_tuple(opt["ensemble_grid"])

    ens = load_model(opt, device).to(device)
    ens.train(False)
    params = []
    for child in ens.models:
        for p in child.parameters():
            p.requires_grad_(False)
        for p in child.decoder.parameters():
            p.requires_grad_(True)
            params.append(p)
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in ens.parameters())
    print(f"Fine-tuning {n_trainable} decoder parameters of {n_total} total ({100*n_trainable/n_total:.2f}%)")
    optimizer = torch.optim.Adam(params, lr=args["lr"])

    volume = load_full_volume(opt, args["data_device"])
    data_min, data_max = float(volume.min()), float(volume.max())
    scale2 = (data_max - data_min) ** 2 + 1e-12
    pairs = build_pairs(ens, grid_dhw, full_shape)
    print(f"{len(pairs)} face-adjacent pairs; actual overlap {ens.actual_ghost_cells} cells")

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    log_rows = []
    rec_history = []
    stop_reason = "completed"
    t0 = time.time()
    it = 0
    for it in range(args["iterations"]):
        ramp = min(1.0, (it + 1) / max(1, args["warmup_iterations"]))
        optimizer.zero_grad(set_to_none=True)
        rec_terms, val_terms, grad_terms = [], [], []
        # Interior anchors keep each brick faithful away from the interfaces.
        if args["interior_points"] > 0:
            for i in range(len(ens.models)):
                x = sample_box(ens.support_extents_dhw[i], full_shape, args["interior_points"], gen, device)
                y = sample_ground_truth(volume, x).to(device)
                rec_terms.append(F.mse_loss(ens.forward_child(x, i), y))
        for p in pairs:
            x = sample_box(p["overlap_dhw"], full_shape, args["points_per_iteration"], gen, device)
            y = sample_ground_truth(volume, x).to(device)
            yi = ens.forward_child(x, p["left"])
            yj = ens.forward_child(x, p["right"])
            rec_terms.append(0.5 * (F.mse_loss(yi, y) + F.mse_loss(yj, y)))
            val_terms.append(F.mse_loss(yi, yj))
            if args["gradient_weight"] > 0:
                gi = normal_derivative(ens, p["left"], x, p["normal_axis_xyz"], p["delta"], p["lo_xyz"], p["hi_xyz"])
                gj = normal_derivative(ens, p["right"], x, p["normal_axis_xyz"], p["delta"], p["lo_xyz"], p["hi_xyz"])
                # Derivatives are per normalized unit; express per voxel to match value scale.
                grad_terms.append(F.mse_loss(gi * p["delta"], gj * p["delta"]))
        l_rec = torch.stack(rec_terms).mean() / scale2
        l_val = torch.stack(val_terms).mean() / scale2
        l_grad = torch.stack(grad_terms).mean() / scale2 if grad_terms else torch.zeros((), device=device)
        loss = l_rec + ramp * args["value_weight"] * l_val + ramp * args["gradient_weight"] * l_grad
        if not torch.isfinite(loss):
            stop_reason = f"non-finite loss at iteration {it}"
            break
        loss.backward()
        optimizer.step()

        rec_history.append(float(l_rec))
        elapsed = time.time() - t0
        if it % args["log_every"] == 0 or it == args["iterations"] - 1:
            row = {"iteration": it, "elapsed_s": round(elapsed, 3), "ramp": ramp,
                   "loss_total": float(loss), "loss_reconstruction": float(l_rec),
                   "loss_value": float(l_val), "loss_gradient": float(l_grad),
                   "lr": optimizer.param_groups[0]["lr"]}
            if use_cuda:
                row["gpu_alloc_gb"] = torch.cuda.memory_allocated(device) / 1024**3
                row["gpu_reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3
            log_rows.append(row)
            print(json.dumps(row))
        if elapsed / 60.0 > args["max_minutes"]:
            stop_reason = f"max_minutes reached at iteration {it}"
            break
        if use_cuda and torch.cuda.max_memory_allocated(device) / 1024**3 > args["max_gpu_gb"]:
            stop_reason = f"GPU memory guard at iteration {it}"
            break
        if len(rec_history) > 150:
            base = float(torch.tensor(rec_history[:100]).median())
            if float(l_rec) > 5 * base:
                stop_reason = f"reconstruction blow-up guard at iteration {it}"
                break

    elapsed = time.time() - t0
    print(f"Stopped: {stop_reason}. {elapsed:.1f}s")
    for i, child in enumerate(ens.models):
        sub = os.path.join(dst, ens.submodel_folders[i])
        torch.save({"state_dict": child.state_dict()}, os.path.join(sub, "model.ckpt.tar"), pickle_protocol=4)

    parent = load_options(dst)
    parent.update({
        "boundary_finetuned_from": args["load_from"],
        "boundary_consistency_weight": args["value_weight"],
        "boundary_gradient_weight": args["gradient_weight"],
        "boundary_finetune_iterations": args["iterations"],
        "boundary_finetune_iterations_completed": it + 1,
        "boundary_points_per_iteration": args["points_per_iteration"],
        "boundary_interior_points": args["interior_points"],
        "boundary_finetune_lr": args["lr"],
        "boundary_warmup_iterations": args["warmup_iterations"],
        "boundary_finetune_seed": args["seed"],
        "boundary_finetune_seconds": elapsed,
        "boundary_finetune_stop_reason": stop_reason,
        "boundary_trainable_parameters": n_trainable,
        "boundary_total_parameters": n_total,
    })
    save_options(parent, dst)
    summary = {"args": args, "stop_reason": stop_reason, "elapsed_seconds": elapsed,
               "iterations_completed": it + 1, "trainable_parameters": n_trainable,
               "total_parameters": n_total, "actual_ghost_cells": ens.actual_ghost_cells,
               "peak_gpu_gb": (torch.cuda.max_memory_allocated(device) / 1024**3) if use_cuda else 0.0,
               "log": log_rows}
    with open(os.path.join(dst, "boundary_finetune_log.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if log_rows:
        with open(os.path.join(dst, "boundary_finetune_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[-1].keys()))
            w.writeheader()
            for r in log_rows:
                w.writerow({k: r.get(k, "") for k in log_rows[-1].keys()})
    print(f"Saved fine-tuned ensemble to {dst}")


if __name__ == "__main__":
    main()
