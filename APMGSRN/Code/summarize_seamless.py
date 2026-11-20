"""Collect Output/SeamlessAPMGSRN/runs/*/metrics.json into metrics.csv/json and plots."""
import argparse
import csv
import json
import os

project_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ORDER = ["g1_hard", "g4_hard", "g4_blend", "g4_consistency_hard", "g4_consistency_blend",
         "g16_hard", "g16_blend"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=os.path.join(project_folder_path, "Output", "SeamlessAPMGSRN"))
    args = ap.parse_args()
    root = args.output_root
    runs = {}
    for name in sorted(os.listdir(os.path.join(root, "runs"))):
        p = os.path.join(root, "runs", name, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                runs[name] = json.load(f)
    names = [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER]
    render = {}
    rp = os.path.join(root, "render_metrics.json")
    if os.path.exists(rp):
        with open(rp) as f:
            render = {r["run_id"]: r for r in json.load(f)["rows"]}
    rows = []
    for n in names:
        m = runs[n]
        r = {"run_id": n, "checkpoint": m["checkpoint"], "routing": m["routing_mode"],
             "actual_ghost_cells": m["actual_ghost_cells"], "fine_tuned": m["fine_tuned"],
             "global_psnr_db": m["global_psnr_db"], "boundary_psnr_db": m["boundary_psnr_db"],
             "interior_psnr_db": m["interior_psnr_db"],
             "pair_value_jump_mean": m["pair_value_jump"]["mean"],
             "pair_value_jump_p95": m["pair_value_jump"]["p95"],
             "composite_value_jump_mean": m["composite_value_jump"]["mean"],
             "composite_value_jump_p95": m["composite_value_jump"]["p95"],
             "composite_value_jump_max": m["composite_value_jump"]["max"],
             "pair_gradient_jump_mean": m["pair_gradient_jump"]["mean"],
             "composite_gradient_jump_mean": m["composite_gradient_jump"]["mean"],
             "composite_gradient_jump_p95": m["composite_gradient_jump"]["p95"],
             "gt_gradient_change_mean": m["gt_gradient_change"]["mean"],
             "throughput_mpts_s": m["throughput_points_per_second"] / 1e6,
             "peak_gpu_gb": m["peak_gpu_gb"], "n_parameters": m["n_parameters"]}
        if n in render:
            r["render_psnr_db"] = render[n]["psnr_db"]
            r["render_ssim"] = render[n]["ssim"]
            r["frame_seconds"] = render[n]["frame_seconds"]
        rows.append(r)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(root, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(root, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(root, "metrics.md"), "w") as f:
        cols = ["run_id", "global_psnr_db", "boundary_psnr_db", "composite_value_jump_mean",
                "composite_gradient_jump_mean", "pair_value_jump_mean", "render_psnr_db", "render_ssim",
                "throughput_mpts_s"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                vals.append(f"{v:.4g}" if isinstance(v, float) else str(v))
            f.write("| " + " | ".join(vals) + " |\n")
    print(open(os.path.join(root, "metrics.md")).read())
    plots(root, rows, names)


def plots(root, rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(root, "plots")
    os.makedirs(out, exist_ok=True)
    labels = [r["run_id"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, key, title, log in [(ax[0], "boundary_psnr_db", "Boundary-band PSNR (dB, higher better)", False),
                               (ax[1], "composite_value_jump_mean", "Mean value jump across interfaces (lower better)", True),
                               (ax[2], "composite_gradient_jump_mean", "Mean normal-derivative jump (lower better)", True)]:
        vals = [max(r[key], 1e-9) for r in rows]
        a.bar(range(len(vals)), vals, color=["#888" if "hard" in l else "#2a7" for l in labels])
        a.set_xticks(range(len(vals))); a.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        a.set_title(title, fontsize=9)
        if log:
            a.set_yscale("log")
        if key == "composite_gradient_jump_mean" and rows:
            a.axhline(rows[0]["gt_gradient_change_mean"], color="k", ls="--", lw=1, label="ground truth")
            a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "seam_metrics.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    gt_done = False
    for n in names:
        p = os.path.join(root, "runs", n, "profile.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            data = list(csv.DictReader(f))
        x = [float(d["voxel_w"]) for d in data]
        if not gt_done:
            ax.plot(x, [float(d["ground_truth"]) for d in data], "k-", lw=2, label="ground truth")
            gt_done = True
        ax.plot(x, [float(d["model"]) for d in data], lw=1, label=n)
    ax.axvline(63.5, color="r", ls=":", lw=1, label="brick split")
    ax.set_xlabel("width voxel index (W axis)"); ax.set_ylabel("scalar value")
    ax.set_title("Line profile across the width split, through the plume")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "line_profile.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
