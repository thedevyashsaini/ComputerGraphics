"""Aggregate the seed + capacity study into comparison tables and figures.

Reads <output_root>/study_runs.csv, runs/<run_id>/metrics.json and render_metrics.csv.
Writes comparison_*.md/csv, paired_tests.md and plots/*.png under <output_root>.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

project_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FULL = (512, 128, 128)
GRID = (2, 2, 2)

LABELS = {
    ("1", "hard"): "g1 hard (paper, min overlap)",
    ("2", "hard"): "g2 hard",
    ("4", "hard"): "g4 hard",
    ("16", "hard"): "g16 hard (paper, wide overlap)",
    ("4", "blend"): "g4 blend (no training)",
    ("4", "ft_hard"): "g4 consistency-FT, hard",
    ("4", "ft_blend"): "g4 consistency-FT + blend (OURS)",
    ("2", "blend"): "g2 blend (no training)",
    ("2", "ft_blend"): "g2 consistency-FT + blend (OURS)",
    ("16", "blend"): "g16 blend",
    ("1", "blend"): "g1 blend",
    ("2", "ft_hard"): "g2 consistency-FT, hard",
    ("4", "rft_hard"): "g4 recon-FT, hard (control)",
    ("4", "rft_blend"): "g4 recon-FT + blend (OURS, best variant)",
    ("16", "rft_hard"): "g16 + recon-FT (paper + equal extra training)",
    ("16", "rft_blend"): "g16 recon-FT + blend",
}

METRICS = [
    ("global_psnr_db", "Global PSNR (dB)", "max"),
    ("boundary_psnr_db", "Boundary PSNR (dB)", "max"),
    ("interior_psnr_db", "Interior PSNR (dB)", "max"),
    ("render_psnr_db", "Render PSNR (dB)", "max"),
    ("render_ssim", "Render SSIM", "max"),
    ("composite_value_jump_mean", "Value jump (mean)", "min"),
    ("composite_value_jump_max", "Value jump (max)", "min"),
    ("composite_gradient_jump_mean", "Gradient jump (mean)", "min"),
    ("throughput_mpts_s", "Query throughput, median of 7 (Mpts/s)", "max"),
    ("frame_seconds", "Frame time (s)", "min"),
]


def overlap_factor(ghost):
    """Stored voxels / volume voxels for a 2x2x2 split with `ghost` actual overlap cells."""
    f = 1.0
    for n, g in zip(FULL, GRID):
        half = n // g
        f *= (2 * min(n, half + ghost)) / n
    return f


def load_bench(root):
    """Median query throughput (Mpts/s) per model from benchmark_throughput.py (interleaved rounds)."""
    bench = {}
    bp = os.path.join(root, "throughput_benchmark.csv")
    if os.path.exists(bp):
        with open(bp) as fh:
            for r in csv.DictReader(fh):
                bench[r["model"]] = float(r["median_mpts_s"])
    return bench


def load(root):
    bench = load_bench(root)
    with open(os.path.join(root, "study_runs.csv")) as fh:
        spec = list(csv.DictReader(fh))
    render = {}
    rp = os.path.join(root, "render_metrics.csv")
    if os.path.exists(rp):
        with open(rp) as fh:
            for r in csv.DictReader(fh):
                render[r["run_id"]] = r
    rows = []
    seen = set()
    for s in spec:
        if s["run_id"] in seen:
            continue
        seen.add(s["run_id"])
        mp = os.path.join(root, "runs", s["run_id"], "metrics.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as fh:
            m = json.load(fh)
        r = dict(s)
        for nest in ("pair_value_jump", "pair_gradient_jump", "composite_value_jump",
                     "composite_gradient_jump", "gt_gradient_change"):
            for stat_name, val in (m.get(nest) or {}).items():
                m[f"{nest}_{stat_name}"] = val
        for k in ("global_psnr_db", "boundary_psnr_db", "interior_psnr_db",
                  "composite_value_jump_mean", "composite_value_jump_max",
                  "composite_gradient_jump_mean", "pair_value_jump_mean", "gt_gradient_change_mean",
                  "n_parameters", "peak_gpu_gb"):
            if k in m:
                r[k] = float(m[k])
        # Per-eval throughput is a single noisy timing; keep it for reference only.
        r["throughput_per_eval_mpts_s"] = float(m["throughput_points_per_second"]) / 1e6
        bkey = f"g{s['ghost']}_f{s['feat']}_{s['variant']}"
        if bkey in bench:
            r["throughput_mpts_s"] = bench[bkey]
        if s["run_id"] in render:
            r["render_psnr_db"] = float(render[s["run_id"]]["psnr_db"])
            r["render_ssim"] = float(render[s["run_id"]]["ssim"])
            r["frame_seconds"] = float(render[s["run_id"]]["frame_seconds"])
        r["overlap_factor"] = overlap_factor(int(s["ghost"]))
        rows.append(r)
    return rows


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["ghost"], r["feat"], r["variant"])].append(r)
    return g


def stat(rs, key):
    v = np.array([r[key] for r in rs if key in r], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def fmt(m, s, n, key):
    if math.isnan(m):
        return "-"
    small = "jump" in key
    f = (lambda x: f"{x:.2e}") if small else ((lambda x: f"{x:.4f}") if "ssim" in key else (lambda x: f"{x:.2f}"))
    return f"{f(m)} +/- {f(s)}" if n > 1 and "throughput" not in key else f(m)


def label(ghost, feat, variant):
    base = LABELS.get((ghost, variant), f"g{ghost} {variant}")
    return base if feat == "24" else f"{base} [{feat}^3 grids]"


MAIN = [("1", "24", "hard"), ("2", "24", "hard"), ("4", "24", "hard"), ("16", "24", "hard"),
        ("16", "27", "hard"), ("16", "32", "hard"), ("16", "24", "rft_hard"), ("16", "24", "blend"), ("4", "24", "blend"), ("4", "24", "rft_hard"), ("4", "24", "rft_blend"), ("4", "24", "ft_hard"),
        ("2", "24", "ft_blend"), ("4", "24", "ft_blend")]


def write_main_table(root, g):
    keys = ["global_psnr_db", "boundary_psnr_db", "interior_psnr_db", "render_psnr_db", "render_ssim",
            "composite_value_jump_mean", "composite_value_jump_max", "composite_gradient_jump_mean",
            "throughput_mpts_s", "frame_seconds"]
    head = ["Method", "Seeds", "Params", "Stored voxels", *[dict((k, l) for k, l, _ in METRICS)[k] for k in keys]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    csv_rows = []
    for key in MAIN:
        rs = g.get(key)
        if not rs:
            continue
        cells = [label(*key), str(len(rs)), f"{int(rs[0]['n_parameters']):,}",
                 f"{rs[0]['overlap_factor']:.2f}x"]
        row = {"method": label(*key), "ghost": key[0], "feat": key[1], "variant": key[2], "seeds": len(rs),
               "n_parameters": int(rs[0]["n_parameters"]), "stored_voxel_factor": rs[0]["overlap_factor"]}
        for k in keys:
            m, s, n = stat(rs, k)
            cells.append(fmt(m, s, n, k))
            row[k + "_mean"], row[k + "_std"] = m, s
        lines.append("| " + " | ".join(cells) + " |")
        csv_rows.append(row)
    with open(os.path.join(root, "comparison_main.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(root, "comparison_main.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    return lines


def paired(root, rows):
    by = {(r["ghost"], r["feat"], r["variant"], r["seed"]): r for r in rows}
    comps = [(("4", "24", "ft_blend"), ("16", "24", "hard")),
             (("4", "24", "ft_blend"), ("16", "27", "hard")),
             (("4", "24", "ft_blend"), ("4", "24", "hard")),
             (("4", "24", "ft_blend"), ("1", "24", "hard")),
             (("4", "24", "blend"), ("4", "24", "hard")),
             (("4", "24", "ft_blend"), ("4", "24", "blend")),
             (("4", "24", "ft_blend"), ("4", "24", "rft_blend")),
             (("4", "24", "ft_blend"), ("16", "24", "rft_hard")),
             (("4", "24", "rft_blend"), ("16", "24", "rft_hard")),
             (("4", "24", "blend"), ("16", "24", "hard")),
             (("4", "24", "rft_hard"), ("4", "24", "hard")),
             (("4", "24", "ft_blend"), ("16", "32", "hard")),
             (("2", "24", "ft_blend"), ("16", "24", "hard")),
             (("2", "24", "ft_blend"), ("4", "24", "ft_blend"))]
    keys = ["global_psnr_db", "boundary_psnr_db", "render_psnr_db", "composite_value_jump_mean",
            "composite_gradient_jump_mean"]
    lines = ["| Ours / candidate | Baseline | Metric | Mean diff | Std | Per-seed diffs | Wins |",
             "|---|---|---|---|---|---|---|"]
    for a, b in comps:
        seeds = sorted({k[3] for k in by if k[:3] == a} & {k[3] for k in by if k[:3] == b})
        if not seeds:
            continue
        for k in keys:
            if any(k not in by[a + (s,)] or k not in by[b + (s,)] for s in seeds):
                continue
            if "jump" in k:   # report as reduction factor baseline/ours
                d = np.array([by[b + (s,)][k] / max(by[a + (s,)][k], 1e-30) for s in seeds])
                win = int((d > 1).sum())
                per = ", ".join(f"{x:.0f}x" if x > 10 else f"{x:.2f}x" for x in d)
                mean, std = f"{d.mean():.3g}x lower", f"{d.std(ddof=1):.3g}" if len(d) > 1 else "-"
            else:
                d = np.array([by[a + (s,)][k] - by[b + (s,)][k] for s in seeds])
                win = int((d > 0).sum())
                per = ", ".join(f"{x:+.2f}" for x in d)
                mean, std = f"{d.mean():+.3f} dB", f"{d.std(ddof=1):.3f}" if len(d) > 1 else "-"
            lines.append(f"| {label(*a)} | {label(*b)} | {k} | {mean} | {std} | {per} | {win}/{len(seeds)} |")
    with open(os.path.join(root, "paired_tests.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def capacity_table(root, g):
    lines = ["| Grid | Params | g4 hard | g4 OURS | g16 hard | OURS - g16 (dB) | g4 hard jump | OURS jump | g16 jump |",
             "|---|---|---|---|---|---|---|---|---|"]
    for feat in ["16", "24", "27", "32"]:
        a, o, b = g.get(("4", feat, "hard")), g.get(("4", feat, "ft_blend")), g.get(("16", feat, "hard"))
        if not (a or o or b):
            continue
        p = int((a or o or b)[0]["n_parameters"])
        cell = lambda rs, k: fmt(*stat(rs, k), k) if rs else "-"
        diff = f"{stat(o, 'global_psnr_db')[0] - stat(b, 'global_psnr_db')[0]:+.2f}" if o and b else "-"
        lines.append(f"| {feat}^3 | {p:,} | {cell(a, 'global_psnr_db')} | {cell(o, 'global_psnr_db')} | "
                     f"{cell(b, 'global_psnr_db')} | {diff} | {cell(a, 'composite_value_jump_mean')} | "
                     f"{cell(o, 'composite_value_jump_mean')} | {cell(b, 'composite_value_jump_mean')} |")
    with open(os.path.join(root, "comparison_capacity.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def train_minutes(checkpoint):
    """Sum of per-brick training wall time parsed from each brick's train_log.txt."""
    import re
    base = os.path.join(project_folder_path, "SavedModels", checkpoint.replace("_consistency", "").replace("_reconft", ""))
    total, n = 0.0, 0
    if not os.path.isdir(base):
        return float("nan")
    for d in os.listdir(base):
        lp = os.path.join(base, d, "train_log.txt")
        if not os.path.exists(lp):
            continue
        with open(lp, errors="ignore") as fh:
            hits = re.findall(r"completed training after (\d+)m ([\d.]+)s", fh.read())
        if hits:
            m, s = hits[-1]
            total += int(m) * 60 + float(s); n += 1
    return total / 60 if n == 8 else float("nan")


def finetune_minutes(checkpoint):
    lp = os.path.join(project_folder_path, "SavedModels", checkpoint, "boundary_finetune_log.json")
    if not os.path.exists(lp):
        return 0.0
    with open(lp) as fh:
        return float(json.load(fh)["elapsed_seconds"]) / 60


def write_cost_table(root, g):
    bench = {}
    bp = os.path.join(root, "throughput_benchmark.csv")
    if os.path.exists(bp):
        with open(bp) as fh:
            for r in csv.DictReader(fh):
                bench[r["model"]] = r
    head = ["Method", "Params", "Size (MB, fp32)", "Stored voxels", "Train (min, 8 bricks)", "Fine-tune (min)",
            "Query speed, median (Mpts/s)", "Frame time (s)", "Peak GPU (GB)", "Global PSNR", "Value jump"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in MAIN:
        rs = g.get(key)
        if not rs:
            continue
        r0 = rs[0]
        tm = np.nanmean([train_minutes(r["checkpoint"]) for r in rs])
        fm = np.mean([finetune_minutes(r["checkpoint"]) for r in rs])
        bname = f"g{key[0]}_f{key[1]}_{key[2]}"
        speed = f"{float(bench[bname]['median_mpts_s']):.1f}" if bname in bench else "-"
        lines.append("| " + " | ".join([
            label(*key), f"{int(r0['n_parameters']):,}", f"{r0['n_parameters'] * 4 / 2**20:.2f}",
            f"{r0['overlap_factor']:.2f}x", f"{tm:.1f}", f"{fm:.1f}" if fm else "0", speed,
            fmt(*stat(rs, "frame_seconds"), "frame_seconds"), f"{r0.get('peak_gpu_gb', float('nan')):.3f}",
            fmt(*stat(rs, "global_psnr_db"), "global_psnr_db"),
            fmt(*stat(rs, "composite_value_jump_mean"), "composite_value_jump_mean")]) + " |")
    with open(os.path.join(root, "comparison_costs.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

def plots(root, rows, g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(root, "plots")
    os.makedirs(out, exist_ok=True)
    GREY, BLUE, GREEN, RED, ORANGE = "#8a8a8a", "#3b75af", "#23a974", "#c8453b", "#e39b2d"

    def colour(key):
        if key[2] == "ft_blend":
            return GREEN
        if key[0] == "16" and key[2] == "rft_hard":
            return RED
        if key[2] == "rft_blend":
            return "#137a52"
        if key[2] == "rft_hard":
            return ORANGE
        if key[2] in ("blend", "ft_hard"):
            return BLUE
        return RED if key[0] == "16" else GREY

    # 1. Main bars with seed error bars
    keys = [k for k in MAIN if k in g]
    panels = [("global_psnr_db", "Global PSNR (dB)"), ("boundary_psnr_db", "Boundary-band PSNR (dB)"),
              ("render_psnr_db", "Rendering PSNR (dB)"), ("composite_value_jump_mean", "Value jump across interfaces (log)"),
              ("composite_gradient_jump_mean", "Normal-gradient jump"), ("throughput_mpts_s", "Query throughput, median of 7 rounds (Mpts/s)")]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (k, t) in zip(axs.flat, panels):
        ms = [stat(g[c], k) for c in keys]
        x = np.arange(len(keys))
        if all(math.isnan(m[0]) for m in ms) or k == "throughput_mpts_s":
            if k == "throughput_mpts_s":
                ax.bar(x, [m[0] for m in ms], color=[colour(c) for c in keys])
                ax.set_xticks(x); ax.set_xticklabels([label(*c).replace(" (", "\n(") for c in keys], rotation=60, ha="right", fontsize=7)
                ax.set_title(t, fontsize=10)
                continue
            ax.set_title(t + " (not available yet)", fontsize=10); ax.axis("off")
            continue
        ax.bar(x, [m[0] for m in ms], yerr=[m[1] for m in ms], color=[colour(c) for c in keys], capsize=3)
        for c_i, c in enumerate(keys):
            vals = [r[k] for r in g[c] if k in r]
            ax.scatter([c_i] * len(vals), vals, s=8, color="k", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([label(*c).replace(" (", "\n(") for c in keys], rotation=60, ha="right", fontsize=7)
        ax.set_title(t, fontsize=10)
        if "jump" in k and "value" in t.lower():
            ax.set_yscale("log")
        elif "psnr" in k:
            lo = min(m[0] - m[1] for m in ms if not math.isnan(m[0]))
            hi = max(m[0] + m[1] for m in ms if not math.isnan(m[0]))
            ax.set_ylim(lo - 0.6, hi + 0.4)
        if k == "composite_gradient_jump_mean":
            gt = np.mean([r["gt_gradient_change_mean"] for r in rows if r["ghost"] == "4" and "gt_gradient_change_mean" in r])
            ax.axhline(gt, ls="--", color="k", lw=1, label="ground truth")
            ax.legend(fontsize=7)
            lo = min(m[0] - m[1] for m in ms if not math.isnan(m[0])); ax.set_ylim(min(lo, gt) * 0.9, None)
    fig.suptitle("Plume 2x2x2: mean +/- std over seeds (dots = individual seeds). Red = paper g16, grey = paper hard routing, "
                 "blue = partial delta, green = ours", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "main_comparison.png"), dpi=140); plt.close(fig)

    # 2. Capacity curve: global PSNR vs parameters
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    series = [(("4", "hard"), "g4 hard (paper routing)", GREY, "o"), (("4", "ft_blend"), "g4 consistency-FT + blend (OURS)", GREEN, "s"),
              (("4", "rft_blend"), "g4 recon-FT + blend (OURS, best)", "#137a52", "*"),
              (("16", "hard"), "g16 hard (paper default overlap)", RED, "^")]
    for (gh, var), name, col, mk in series:
        pts = sorted({r["feat"] for r in rows if r["ghost"] == gh and r["variant"] == var}, key=int)
        if not pts:
            continue
        for ax, k in zip(axs, ["global_psnr_db", "boundary_psnr_db"]):
            xs, ys, es = [], [], []
            for f in pts:
                m, s, n = stat(g[(gh, f, var)], k)
                xs.append(g[(gh, f, var)][0]["n_parameters"] / 1e6); ys.append(m); es.append(s)
            ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, capsize=3, label=name)
            for xx, yy, f in zip(xs, ys, pts):
                ax.annotate(f"{f}^3", (xx, yy), textcoords="offset points", xytext=(4, -10), fontsize=7, color=col)
    for ax, t in zip(axs, ["Global PSNR vs model size", "Boundary-band PSNR vs model size"]):
        ax.set_xlabel("total parameters (millions, all 8 bricks)"); ax.set_ylabel("dB"); ax.set_title(t); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "capacity_curve.png"), dpi=140); plt.close(fig)

    # 3. Ghost-cell curve at 24^3
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for var, name, col, mk in [("hard", "hard routing (paper)", GREY, "o"), ("blend", "blend only", BLUE, "D"),
                               ("ft_blend", "consistency-FT + blend (OURS)", GREEN, "s")]:
        gs = sorted({r["ghost"] for r in rows if r["feat"] == "24" and r["variant"] == var}, key=int)
        if not gs:
            continue
        for ax, k in zip(axs, ["global_psnr_db", "boundary_psnr_db", "composite_value_jump_mean"]):
            ms = [stat(g[(gh, "24", var)], k) for gh in gs]
            ax.errorbar([int(x) for x in gs], [m[0] for m in ms], yerr=[m[1] for m in ms], color=col, marker=mk, capsize=3, label=name)
    for ax, t in zip(axs, ["Global PSNR (dB)", "Boundary-band PSNR (dB)", "Value jump (log)"]):
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 16]); ax.set_xticklabels(["1", "2", "4", "16"])
        ax.set_xlim(0.8, 20); ax.set_xlabel("actual ghost cells"); ax.set_title(t); ax.grid(alpha=.3)
    jumps = [r["composite_value_jump_mean"] for r in rows if r["feat"] == "24"]
    axs[2].set_yscale("log"); axs[2].set_ylim(min(jumps) / 3, max(jumps) * 3); axs[0].legend(fontsize=8)
    fig.suptitle("Effect of overlap width at equal model size (24^3 grids, mean +/- std over seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "ghost_cell_curve.png"), dpi=140); plt.close(fig)

    # 4. Trade-off scatter: quality vs cost
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for c in keys:
        rs = g[c]
        for ax, (xk, xl) in zip(axs, [("n_parameters", "parameters (M)"), ("throughput_mpts_s", "query throughput, median of 7 (Mpts/s)"),
                                      ("composite_value_jump_mean", "value jump (log)")]):
            mx = stat(rs, xk)[0] / (1e6 if xk == "n_parameters" else 1)
            my, sy, _ = stat(rs, "global_psnr_db")
            ax.errorbar(mx, my, yerr=sy, marker="o", color=colour(c), capsize=3)
            ax.annotate(label(*c).split(" (")[0] + ("" if c[1] == "24" else f" {c[1]}^3"), (mx, my), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax.set_xlabel(xl); ax.set_ylabel("global PSNR (dB)"); ax.grid(alpha=.3)
    axs[2].set_xscale("log")
    axs[0].set_title("Accuracy vs storage"); axs[1].set_title("Accuracy vs inference speed (up-right is best)")
    axs[2].set_title("Accuracy vs seam discontinuity (up-left is best)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "tradeoffs.png"), dpi=140); plt.close(fig)

    # 5. Per-seed paired lines
    seeds = sorted({r["seed"] for r in rows if r["feat"] == "24" and r["ghost"] in ("4", "16")}, key=int)
    seq = [("16", "24", "hard"), ("16", "27", "hard"), ("4", "24", "hard"), ("4", "24", "blend"), ("4", "24", "ft_blend")]
    seq = [c for c in seq if c in g]
    by = {(r["ghost"], r["feat"], r["variant"], r["seed"]): r for r in rows}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, k in zip(axs, ["global_psnr_db", "boundary_psnr_db"]):
        for s in seeds:
            ys = [by.get(c + (s,), {}).get(k, np.nan) for c in seq]
            ax.plot(range(len(seq)), ys, marker="o", label=f"seed {s}")
        ax.set_xticks(range(len(seq))); ax.set_xticklabels([label(*c).replace(" (", "\n(") for c in seq], fontsize=7)
        ax.set_title(k.replace("_", " ")); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Same seed, different method: every line should rise left to right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "per_seed_paired.png"), dpi=140); plt.close(fig)

    # 6. Headline: paper vs ours, with model size in the tick labels
    head = [(("16", "24", "hard"), "Paper g16\n(same size)\n"), (("16", "32", "hard"), "Paper g16\n32^3 grids\n(2.3x params)"),
            (("4", "24", "hard"), "g4 hard"), (("4", "24", "blend"), "g4 + blend\n(no training)"),
            (("4", "24", "ft_blend"), "OURS\nconsistency-FT\n+ blend"), (("4", "24", "rft_blend"), "OURS\nrecon-FT\n+ blend")]
    head = [(c, n) for c, n in head if c in g]
    cols = [RED, "#e8837b", GREY, BLUE, GREEN, "#137a52"][:len(head)]
    hp = [("global_psnr_db", "Global PSNR (dB)"), ("boundary_psnr_db", "Boundary-band PSNR (dB)"),
          ("render_psnr_db", "Rendering PSNR (dB)"), ("composite_value_jump_mean", "Value jump at seams (log, lower=better)")]
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.8))
    for ax, (k, t) in zip(axs, hp):
        ms = [stat(g[c], k) for c, _ in head]
        x = np.arange(len(head))
        ax.bar(x, [m[0] for m in ms], yerr=[m[1] for m in ms], color=cols, capsize=3)
        for i, (c, _) in enumerate(head):
            vals = [r[k] for r in g[c] if k in r]
            ax.scatter([i] * len(vals), vals, s=10, color="k", zorder=3)
        names = [f"{n}\n{g[c][0]['n_parameters'] / 1e6:.2f}M, n={len(g[c])}" for c, n in head]
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
        ax.set_title(t, fontsize=10); ax.grid(axis="y", alpha=.3)
        if "jump" in k:
            ax.set_yscale("log")
        else:
            lo = min(m[0] - m[1] for m in ms); hi = max(m[0] + m[1] for m in ms)
            ax.set_ylim(lo - 0.5, hi + 0.3)
            for i, m in enumerate(ms):
                ax.text(i, m[0] + m[1] + 0.05, f"{m[0]:.2f}", ha="center", fontsize=7)
    fig.suptitle("Plume 2x2x2 - paper (red) vs our seamless delta (green). Bars = mean over seeds, dots = individual seeds", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "headline.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=os.path.join(project_folder_path, "Output", "SeedCapacityStudy"))
    root = ap.parse_args().output_root
    rows = load(root)
    with open(os.path.join(root, "all_runs.csv"), "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    g = group(rows)
    lines = write_main_table(root, g)
    paired(root, rows)
    capacity_table(root, g)
    write_cost_table(root, g)
    plots(root, rows, g)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
