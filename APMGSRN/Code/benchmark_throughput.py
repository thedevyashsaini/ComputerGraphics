"""Fair inference-speed benchmark: all models loaded once, measured in interleaved rounds, median reported.

Usage: python Code/benchmark_throughput.py --models name=checkpoint:mode ... --rounds 7
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Models.options import load_options          # noqa: E402
from Models.models import load_model             # noqa: E402

project_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
save_folder = os.path.join(project_folder_path, "SavedModels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--points", type=int, default=262144)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--output", default=os.path.join(project_folder_path, "Output", "SeedCapacityStudy", "throughput_benchmark.csv"))
    a = ap.parse_args()
    dev = a.device
    models = {}
    for spec in a.models:
        name, rest = spec.split("=")
        ckpt, mode = rest.split(":")
        opt = load_options(os.path.join(save_folder, ckpt))
        opt["device"] = dev; opt["data_device"] = dev; opt["ensemble_routing_mode"] = mode
        m = load_model(opt, dev).to(dev).eval()
        m.set_routing_mode(mode)
        models[name] = m
    x = (torch.rand(a.points, 3, generator=torch.Generator().manual_seed(0)) * 2 - 1).to(dev)
    times = {n: [] for n in models}
    with torch.no_grad():
        for _ in range(3):                      # warm up and lift GPU clocks
            for m in models.values():
                m(x)
        for r in range(a.rounds):
            order = list(models) if r % 2 == 0 else list(reversed(list(models)))
            for n in order:
                torch.cuda.synchronize(dev); t = time.perf_counter()
                for _ in range(a.repeats):
                    models[n](x)
                torch.cuda.synchronize(dev)
                times[n].append(a.points * a.repeats / (time.perf_counter() - t) / 1e6)
    rows = []
    for n in models:
        v = times[n]
        rows.append({"model": n, "median_mpts_s": statistics.median(v), "min_mpts_s": min(v), "max_mpts_s": max(v),
                     "rounds": len(v)})
        print(f"{n:28s} median {statistics.median(v):7.2f} Mpts/s  (min {min(v):.2f}, max {max(v):.2f})")
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
