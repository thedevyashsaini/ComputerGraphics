"""
Rendering comparison for Seamless APMGSRN (part 3).

Self-contained fixed-step emission-absorption ray marcher in PyTorch, so the raw data and
every model are rendered with an identical camera, transfer function and sample positions.

  python Code/compare_renderings.py --runs g1_hard=Plume_APMGSRN_dd_g1:hard \
      g4_blend=Plume_APMGSRN_dd_g4:blend --device cuda:0

Writes renders/<run>.png, renders/<run>_error.png, render_metrics.csv/json and a montage.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Models.options import load_options                     # noqa: E402
from Models.models import load_model                        # noqa: E402
from Other.utility_functions import nc_to_tensor, PSNR, ssim  # noqa: E402

project_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
data_folder = os.path.join(project_folder_path, "Data")
save_folder = os.path.join(project_folder_path, "SavedModels")
colormap_folder = os.path.join(project_folder_path, "Colormaps")


class TransferFunction:
    def __init__(self, name, vmin, vmax, device):
        with open(os.path.join(colormap_folder, name)) as f:
            c = json.load(f)[0]
        rgb = torch.tensor(c["RGBPoints"], dtype=torch.float32).reshape(-1, 4)
        self.cx = ((rgb[:, 0] - rgb[0, 0]) / (rgb[-1, 0] - rgb[0, 0])).to(device)
        self.cv = rgb[:, 1:].to(device)
        if "Points" in c:
            a = torch.tensor(c["Points"], dtype=torch.float32).reshape(-1, 4)
            self.ax = ((a[:, 0] - a[0, 0]) / (a[-1, 0] - a[0, 0])).to(device)
            self.av = a[:, 1].to(device)
        else:
            self.ax = torch.tensor([0.0, 1.0], device=device)
            self.av = torch.tensor([0.0, 1.0], device=device)
        self.vmin, self.vmax = vmin, vmax

    @staticmethod
    def _interp(x, xp, fp):
        i = torch.searchsorted(xp, x.clamp(0, 1)).clamp(1, xp.numel() - 1)
        x0, x1 = xp[i - 1], xp[i]
        t = ((x.clamp(0, 1) - x0) / (x1 - x0).clamp_min(1e-12)).unsqueeze(-1) if fp.dim() == 2 else \
            (x.clamp(0, 1) - x0) / (x1 - x0).clamp_min(1e-12)
        return fp[i - 1] + t * (fp[i] - fp[i - 1])

    def __call__(self, v):
        x = (v - self.vmin) / (self.vmax - self.vmin)
        return self._interp(x, self.cx, self.cv), self._interp(x, self.ax, self.av)


class GridField(torch.nn.Module):
    """Ground truth: trilinear sampling of the raw volume (same as training targets)."""
    def __init__(self, volume):
        super().__init__()
        self.volume = volume

    def forward(self, x):
        g = x.reshape(1, 1, 1, -1, 3)
        return F.grid_sample(self.volume, g, mode="bilinear", align_corners=True).reshape(-1, 1)


def camera_rays(H, W, full_shape, azi, polar, dist, fov_y, device):
    D, Hh, Ww = full_shape
    half = torch.tensor([Ww - 1, Hh - 1, D - 1], dtype=torch.float32) / 2
    half = half / half.max()                                   # world half extents (XYZ)
    az, el = math.radians(azi), math.radians(90 - polar)
    eye = torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]) * dist
    fwd = -eye / eye.norm()
    up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.linalg.cross(fwd, up); right /= right.norm()
    up = torch.linalg.cross(right, fwd)
    ty = math.tan(math.radians(fov_y) / 2)
    tx = ty * W / H
    j, i = torch.meshgrid(torch.linspace(1, -1, H), torch.linspace(-1, 1, W), indexing="ij")
    dirs = fwd + i[..., None] * tx * right + j[..., None] * ty * up
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return eye.to(device), dirs.reshape(-1, 3).to(device), half.to(device)


@torch.no_grad()
def render(field, tf, H, W, full_shape, args, device):
    eye, dirs, half = camera_rays(H, W, full_shape, args["azi"], args["polar"], args["dist"], args["fov"], device)
    inv = 1.0 / torch.where(dirs.abs() < 1e-9, torch.full_like(dirs, 1e-9), dirs)
    t0 = (-half - eye) * inv
    t1 = (half - eye) * inv
    tn = torch.minimum(t0, t1).max(-1).values.clamp_min(0)
    tf_ = torch.maximum(t0, t1).min(-1).values
    hit = tf_ > tn
    n_rays = dirs.shape[0]
    color = torch.zeros(n_rays, 3, device=device)
    trans = torch.ones(n_rays, 1, device=device)
    S = args["samples"]
    step_world = (2.0 * half.max() * math.sqrt(3)) / S
    idx = torch.nonzero(hit)[:, 0]
    for s in range(S):
        t = tn[idx] + (s + 0.5) * step_world
        live = t < tf_[idx]
        if not bool(live.any()):
            break
        ri = idx[live]
        p = eye + dirs[ri] * t[live, None]
        x = (p / half).clamp(-1, 1)                            # normalized XYZ
        v = field(x).float()[:, 0]
        rgb, op = tf(v)
        alpha = 1 - torch.exp(-op * args["density"] * step_world).unsqueeze(-1)
        color[ri] += trans[ri] * alpha * rgb
        trans[ri] *= (1 - alpha)
        idx = idx[(trans[idx, 0] > 1e-3) & (t < tf_[idx])] if s % 16 == 15 else idx
    img = color + trans * torch.tensor(args["background"], device=device)
    return img.reshape(H, W, 3).clamp(0, 1)


def to_png(img, path):
    import imageio.v2 as imageio
    imageio.imwrite(path, (img.cpu().numpy() * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run_id=checkpoint:hard|blend")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hw", default="384,192")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--azi", type=float, default=45)
    ap.add_argument("--polar", type=float, default=80)
    ap.add_argument("--dist", type=float, default=3.0)
    ap.add_argument("--fov", type=float, default=42)
    ap.add_argument("--density", type=float, default=60.0)
    ap.add_argument("--background", default="1,1,1")
    ap.add_argument("--colormap", default="Plume.json")
    ap.add_argument("--data", default="Plume.nc")
    ap.add_argument("--crop", default="0.40,0.60", help="vertical crop fraction around the D split")
    ap.add_argument("--output_root", default=os.path.join(project_folder_path, "Output", "SeamlessAPMGSRN"))
    ap.add_argument("--montage_runs", default=None,
                    help="comma-separated run_ids to show in montage.png (default: all)")
    args = vars(ap.parse_args())
    args["background"] = [float(v) for v in args["background"].split(",")]
    H, W = [int(v) for v in args["hw"].split(",")]
    device = args["device"]
    out = os.path.join(args["output_root"], "renders")
    os.makedirs(out, exist_ok=True)

    vol = nc_to_tensor(os.path.join(data_folder, args["data"]), {"extents": None})[0].float().to(device)
    full_shape = list(vol.shape[2:])
    tf = TransferFunction(args["colormap"], float(vol.min()), float(vol.max()), device)
    t = time.perf_counter()
    ref = render(GridField(vol), tf, H, W, full_shape, args, device)
    gt_time = time.perf_counter() - t
    to_png(ref, os.path.join(out, "ground_truth.png"))
    rows = [{"run_id": "ground_truth", "psnr_db": float("inf"), "ssim": 1.0, "frame_seconds": gt_time}]
    images = {"ground_truth": ref}
    for spec in args["runs"]:
        run_id, rest = spec.split("=")
        ckpt, mode = rest.split(":")
        opt = load_options(os.path.join(save_folder, ckpt))
        opt["device"] = device
        opt["ensemble_routing_mode"] = mode
        model = load_model(opt, device).to(device).eval()
        model.set_routing_mode(mode)
        render(model, tf, 32, 16, full_shape, args, device)     # warm-up
        if "cuda" in device:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        t = time.perf_counter()
        img = render(model, tf, H, W, full_shape, args, device)
        if "cuda" in device:
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t
        images[run_id] = img
        to_png(img, os.path.join(out, f"{run_id}.png"))
        err = (img - ref).abs()
        to_png((err * 8).clamp(0, 1), os.path.join(out, f"{run_id}_error_x8.png"))
        a = img.permute(2, 0, 1).unsqueeze(0); b = ref.permute(2, 0, 1).unsqueeze(0)
        rows.append({"run_id": run_id, "checkpoint": ckpt, "routing_mode": mode,
                     "psnr_db": float(PSNR(a, b)), "ssim": float(ssim(a, b)),
                     "frame_seconds": dt,
                     "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if "cuda" in device else 0.0})
        print(rows[-1])
        del model
    with open(os.path.join(args["output_root"], "render_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "checkpoint", "routing_mode", "psnr_db", "ssim",
                                          "frame_seconds", "peak_gpu_gb"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args["output_root"], "render_metrics.json"), "w") as f:
        json.dump({"settings": args, "rows": rows}, f, indent=2)
    if args["montage_runs"]:
        keep = ["ground_truth"] + [r for r in args["montage_runs"].split(",") if r in images]
        images = {k: images[k] for k in keep}
    make_montage(images, rows, args, out)


def make_montage(images, rows, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(images.keys())
    c0, c1 = [float(v) for v in args["crop"].split(",")]
    ref = images["ground_truth"].cpu().numpy()
    fig, ax = plt.subplots(3, len(names), figsize=(2.6 * len(names), 11),
                           gridspec_kw={"height_ratios": [2, 1, 1]})
    info = {r["run_id"]: r for r in rows}
    for k, n in enumerate(names):
        img = images[n].cpu().numpy()
        H = img.shape[0]
        r0, r1 = int(c0 * H), int(c1 * H)
        ax[0, k].imshow(img)
        ax[0, k].axhspan(r0, r1, fill=False, ec="red", lw=1)
        title = n if n == "ground_truth" else f"{n}\n{info[n]['psnr_db']:.2f} dB / {info[n]['ssim']:.4f}"
        ax[0, k].set_title(title, fontsize=8)
        ax[1, k].imshow(img[r0:r1])
        ax[2, k].imshow(np.clip(np.abs(img[r0:r1] - ref[r0:r1]) * 8, 0, 1))
        for a in ax[:, k]:
            a.set_xticks([]); a.set_yticks([])
    ax[1, 0].set_ylabel("zoom on split", fontsize=8)
    ax[2, 0].set_ylabel("|error| x8", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "montage.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
