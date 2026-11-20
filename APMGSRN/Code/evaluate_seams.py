"""
Seam metrics for domain-decomposed APMGSRN (Seamless APMGSRN, part 3).

Writes <output_dir>/metrics.json, pairs.csv, profile.csv and figures. Metrics:
  * global PSNR over the exact full voxel lattice
  * boundary-band PSNR (voxels within --boundary_band_cells of an internal split), per plane
  * child-pair disagreement |f_i - f_j| and |d_n f_i - d_n f_j| on each interface plane
    (property of the trained bricks; what fine-tuning reduces)
  * composite jumps of the routed field F across each plane (what a viewer sees;
    what blending removes): |F(p+e) - F(p-e)| and the one-voxel one-sided normal
    derivative jump, reported next to the same quantity measured on ground truth
  * throughput, peak GPU memory, fraction of points touching 1/2/4/8 bricks
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Models.options import load_options                          # noqa: E402
from Models.models import load_model                             # noqa: E402
from Models.ensemble_utils import (parse_int_tuple, enumerate_face_adjacencies,  # noqa: E402
    intersect_extents_dhw, voxel_to_normalized)
from Other.utility_functions import nc_to_tensor, make_coord_grid  # noqa: E402

project_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
data_folder = os.path.join(project_folder_path, "Data")
save_folder = os.path.join(project_folder_path, "SavedModels")


def sync(device):
    if "cuda" in str(device):
        torch.cuda.synchronize(device)


def stats(t):
    t = t.detach().float().flatten().cpu()
    return {"mean": float(t.mean()), "median": float(t.median()),
            "p95": float(torch.quantile(t, 0.95)) if t.numel() < 16_000_000 else float(t.sort().values[int(0.95 * (t.numel() - 1))]),
            "max": float(t.max())}


def psnr(mse, data_range):
    return float(20 * np.log10(data_range) - 10 * np.log10(max(mse, 1e-30)))


@torch.no_grad()
def query(model, x, device, batch=2**18):
    out = torch.empty((x.shape[0], 1), dtype=torch.float32)
    for s in range(0, x.shape[0], batch):
        out[s:s + batch] = model(x[s:s + batch].to(device)).float().cpu()
    return out


@torch.no_grad()
def reconstruct(model, full_shape, device):
    """Full DHW reconstruction on CPU, built slab by slab along D."""
    D, H, W = full_shape
    out = torch.empty((D, H, W), dtype=torch.float32)
    ys = voxel_to_normalized(torch.arange(H, dtype=torch.float32), H)
    xs = voxel_to_normalized(torch.arange(W, dtype=torch.float32), W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    hw = torch.stack([xx, yy, torch.zeros_like(xx)], -1).reshape(-1, 3)    # XYZ, z filled per slab
    slab = max(1, (2**20) // (H * W))
    for d0 in range(0, D, slab):
        d1 = min(D, d0 + slab)
        zs = voxel_to_normalized(torch.arange(d0, d1, dtype=torch.float32), D)
        x = hw.unsqueeze(0).repeat(d1 - d0, 1, 1)
        x[:, :, 2] = zs[:, None]
        out[d0:d1] = query(model, x.reshape(-1, 3), device).reshape(d1 - d0, H, W)
    return out


def band_masks(full_shape, grid_dhw, band):
    planes = []
    for a in range(3):
        for k in range(1, grid_dhw[a]):
            s = int(k * full_shape[a] / grid_dhw[a])
            m = torch.zeros(full_shape, dtype=torch.bool)
            sl = [slice(None)] * 3
            sl[a] = slice(max(0, s - band), min(full_shape[a], s + band))
            m[tuple(sl)] = True
            planes.append({"axis_dhw": a, "split_voxel": s, "mask": m})
    return planes


def plane_points(pair, ens, full_shape, n, gen):
    """Points on the interface plane (voxel s - 0.5, the hard-router split), tangentially inside the overlap."""
    a = pair["normal_axis_dhw"]
    ov = intersect_extents_dhw(ens.support_extents_dhw[pair["left"]], ens.support_extents_dhw[pair["right"]])
    s = ens.core_extents_dhw[pair["right"]][2 * a]
    cols = []
    for ax in range(3):
        if ax == a:
            u = torch.full((n,), s - 0.5)
        else:
            u = torch.rand(n, generator=gen) * (ov[2 * ax + 1] - 1 - ov[2 * ax]) + ov[2 * ax]
        cols.append(voxel_to_normalized(u, full_shape[ax]))
    return torch.stack(cols[::-1], -1), 2 - a


def gt_sample(volume, x):
    g = x.reshape(1, 1, 1, -1, 3)
    return F.grid_sample(volume, g, mode="bilinear", align_corners=True).reshape(-1, 1)


def shifted(x, axis_xyz, amount):
    y = x.clone()
    y[:, axis_xyz] += amount
    return y


@torch.no_grad()
def interface_metrics(ens, volume, full_shape, grid_dhw, device, n, gen, data_range):
    rows, all_val, all_grad, all_cval, all_cgrad, all_gtgrad = [], [], [], [], [], []
    for p in enumerate_face_adjacencies(grid_dhw):
        x, axis_xyz = plane_points(p, ens, full_shape, n, gen)
        h = 2.0 / (full_shape[p["normal_axis_dhw"]] - 1)   # one voxel in normalized units
        eps = 1e-3 * h
        xd = x.to(device)
        fi = ens.forward_child(xd, p["left"]).float().cpu()
        fj = ens.forward_child(xd, p["right"]).float().cpu()
        # Central differences (one voxel), stay inside both supports for g >= 1.
        gi = (ens.forward_child(shifted(xd, axis_xyz, h / 2), p["left"]) -
              ens.forward_child(shifted(xd, axis_xyz, -h / 2), p["left"])).float().cpu()
        gj = (ens.forward_child(shifted(xd, axis_xyz, h / 2), p["right"]) -
              ens.forward_child(shifted(xd, axis_xyz, -h / 2), p["right"])).float().cpu()
        val = (fi - fj).abs() / data_range
        grad = (gi - gj).abs() / data_range                       # per-voxel derivative units
        # Composite routed field.
        Fp = query(ens, shifted(x, axis_xyz, eps), device)
        Fm = query(ens, shifted(x, axis_xyz, -eps), device)
        Fp1 = query(ens, shifted(x, axis_xyz, h), device)
        Fm1 = query(ens, shifted(x, axis_xyz, -h), device)
        cval = (Fp - Fm).abs() / data_range
        cgrad = ((Fp1 - Fp) - (Fm - Fm1)).abs() / data_range
        Gp, Gm = gt_sample(volume, shifted(x, axis_xyz, eps)), gt_sample(volume, shifted(x, axis_xyz, -eps))
        Gp1, Gm1 = gt_sample(volume, shifted(x, axis_xyz, h)), gt_sample(volume, shifted(x, axis_xyz, -h))
        gtgrad = ((Gp1 - Gp) - (Gm - Gm1)).abs() / data_range
        all_val.append(val); all_grad.append(grad); all_cval.append(cval)
        all_cgrad.append(cgrad); all_gtgrad.append(gtgrad)
        row = {"left": p["left"], "right": p["right"], "normal_axis_dhw": p["normal_axis_dhw"],
               "split_voxel": int(ens.core_extents_dhw[p["right"]][2 * p["normal_axis_dhw"]])}
        for k, t in [("pair_value", val), ("pair_gradient", grad), ("composite_value", cval),
                     ("composite_gradient", cgrad), ("gt_gradient_change", gtgrad)]:
            s = stats(t)
            row[f"{k}_mean"] = s["mean"]; row[f"{k}_p95"] = s["p95"]; row[f"{k}_max"] = s["max"]
        rows.append(row)
    cat = lambda l: torch.cat(l)
    return rows, {"pair_value_jump": stats(cat(all_val)), "pair_gradient_jump": stats(cat(all_grad)),
                  "composite_value_jump": stats(cat(all_cval)),
                  "composite_gradient_jump": stats(cat(all_cgrad)),
                  "gt_gradient_change": stats(cat(all_gtgrad))}


@torch.no_grad()
def throughput(ens, device, n, repeats, warmup=5):
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.rand(n, 3, generator=gen) * 2 - 1).to(device)
    for _ in range(warmup):
        ens(x)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        ens(x)
    sync(device)
    dt = time.perf_counter() - t0
    w = ens.model_support_weights(x)
    active = (w > 0).sum(0)
    frac = {str(k): float((active == k).float().mean()) for k in (1, 2, 4, 8)}
    return n * repeats / dt, frac


def save_figures(out_dir, recon, gt, full_shape, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    D, H, W = full_shape
    h_idx = H // 2 - 2                      # D x W slice through the plume, crossing D and W splits
    err = (recon[:, h_idx, :] - gt[:, h_idx, :]).abs().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(9, 6))
    for a, img, title in [(ax[0], gt[:, h_idx, :].numpy(), "ground truth"),
                          (ax[1], recon[:, h_idx, :].numpy(), label),
                          (ax[2], err, "|error|")]:
        im = a.imshow(img, cmap="inferno" if title != "|error|" else "magma", aspect="auto",
                      vmax=None if title != "|error|" else max(1e-6, float(np.quantile(err, 0.999))))
        a.set_title(title); a.set_xlabel("W"); a.set_ylabel("D")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slice_error.png"), dpi=130)
    plt.close(fig)


@torch.no_grad()
def line_profile(ens, volume, full_shape, device, out_dir, gen):
    """Dense profile across the W split (model x axis) through the plume.

    The H coordinate sits two voxels off the H split; the D slice is the one where the ground
    truth varies most along the profile, so the profile crosses real structure.
    """
    D, H, W = full_shape
    s = W // 2
    h = H // 2 - 2
    gt = volume[0, 0]
    d = int(gt[:, h, s - 12:s + 12].std(dim=-1).argmax())
    vox = torch.linspace(s - 12, s + 12, 1201)
    x = torch.stack([voxel_to_normalized(vox, W),
                     torch.full_like(vox, voxel_to_normalized(float(h), H)),
                     torch.full_like(vox, voxel_to_normalized(float(d), D))], -1)
    f = query(ens, x, device)[:, 0]
    g = gt_sample(volume, x)[:, 0]
    with open(os.path.join(out_dir, "profile.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["voxel_w", "model", "ground_truth", "d_index", "h_index"])
        for a, b, c in zip(vox.tolist(), f.tolist(), g.tolist()):
            w.writerow([a, b, c, d, h])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_from", required=True)
    ap.add_argument("--routing_mode", default="hard", choices=["hard", "blend"])
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data_device", default="cpu")
    ap.add_argument("--boundary_band_cells", type=int, default=2)
    ap.add_argument("--face_samples", type=int, default=65536)
    ap.add_argument("--throughput_points", type=int, default=262144)
    ap.add_argument("--throughput_repeats", type=int, default=20)
    ap.add_argument("--output_root", default=os.path.join(project_folder_path, "Output", "SeamlessAPMGSRN"))
    ap.add_argument("--save_reconstruction", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = vars(ap.parse_args())
    t_start = time.time()
    run_id = args["run_id"] or f"{args['load_from']}_{args['routing_mode']}"
    out_dir = os.path.join(args["output_root"], "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    device = args["device"]
    torch.manual_seed(args["seed"])
    gen = torch.Generator().manual_seed(args["seed"])

    opt = load_options(os.path.join(save_folder, args["load_from"]))
    opt["device"] = device
    opt["data_device"] = args["data_device"]
    opt["ensemble_routing_mode"] = args["routing_mode"]
    ens = load_model(opt, device).to(device).eval()
    ens.set_routing_mode(args["routing_mode"])
    full_shape = [int(v) for v in opt["full_shape"]]
    grid_dhw = parse_int_tuple(opt["ensemble_grid"])
    o = dict(opt); o["extents"] = None
    volume = nc_to_tensor(os.path.join(data_folder, opt["data"]), o)[0].float()
    gt = volume[0, 0]
    data_range = float(gt.max() - gt.min())
    if "cuda" in device:
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    recon = reconstruct(ens, full_shape, device)
    recon_seconds = time.time() - t0
    se = (recon - gt) ** 2
    result = {"run_id": run_id, "checkpoint": args["load_from"], "routing_mode": args["routing_mode"],
              "configured_extra_ghost_cells": int(opt.get("ensemble_ghost_cells", 0)),
              "actual_ghost_cells": int(opt.get("ensemble_ghost_cells", 0)) + 1,
              "fine_tuned": "boundary_finetuned_from" in opt,
              "data_range": data_range,
              "global_psnr_db": psnr(float(se.mean()), data_range),
              "reconstruction_seconds": recon_seconds}
    planes = band_masks(full_shape, grid_dhw, args["boundary_band_cells"])
    union = torch.zeros(full_shape, dtype=torch.bool)
    per_plane = []
    for pl in planes:
        union |= pl["mask"]
        per_plane.append({"axis_dhw": pl["axis_dhw"], "split_voxel": pl["split_voxel"],
                          "psnr_db": psnr(float(se[pl["mask"]].mean()), data_range)})
    result["boundary_band_cells"] = args["boundary_band_cells"]
    result["boundary_psnr_db"] = psnr(float(se[union].mean()), data_range)
    result["interior_psnr_db"] = psnr(float(se[~union].mean()), data_range)
    result["boundary_psnr_per_plane"] = per_plane

    pair_rows, jumps = interface_metrics(ens, volume, full_shape, grid_dhw, device,
                                         args["face_samples"], gen, data_range)
    result.update(jumps)
    tp, frac = throughput(ens, device, args["throughput_points"], args["throughput_repeats"])
    result["throughput_points_per_second"] = tp
    result["active_model_fraction"] = frac
    result["peak_gpu_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3 if "cuda" in device else 0.0
    result["n_parameters"] = sum(p.numel() for p in ens.parameters())
    result["elapsed_seconds"] = time.time() - t_start

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)
    line_profile(ens, volume, full_shape, device, out_dir, gen)
    save_figures(out_dir, recon, gt, full_shape, run_id)
    if args["save_reconstruction"]:
        np.save(os.path.join(out_dir, "reconstruction.npy"), recon.numpy())
    print(json.dumps({k: v for k, v in result.items() if k != "boundary_psnr_per_plane"}, indent=2))


if __name__ == "__main__":
    main()
