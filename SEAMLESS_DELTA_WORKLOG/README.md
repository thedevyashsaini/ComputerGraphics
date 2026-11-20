# Seamless APMGSRN: Project Worklog

This is the single, self-contained record of our post-midsem delta for the Computer Graphics and Multimedia course project. It is kept outside the code repo on purpose, so this one file plus the `results/` folder next to it tells the whole story without needing the git history.

Last updated: 2026-11-12 (run 2: three seeds, capacity sweep, fine-tune controls)

## 1. What we are doing

The base paper is APMGSRN (arXiv 2308.02494). Its domain-decomposed mode trains one small network per brick of the volume. At query time a hard router picks exactly one brick per point, so neighbouring bricks disagree right at the brick boundary and leave visible seams. The paper's fix is a wide overlap of extra "ghost cells" (up to 16), which wastes model capacity on redundant overlap and lowers PSNR.

Our delta, "Seamless APMGSRN", has three parts:

1. **Overlap blending.** Inside the shared ghost-cell overlap, the output is a smoothstep partition-of-unity mix of the neighbouring bricks rather than a hard switch. The weights add up to exactly 1 and are centred on the original split, so the field is continuous by construction. This works on existing checkpoints with no retraining.
2. **Interface-consistency fine-tuning.** Only the decoder MLPs (about 4% of parameters) get a short fine-tune with a loss that pulls adjacent bricks toward the same value, and optionally the same gradient, inside the overlap. Reconstruction anchors keep accuracy intact.
3. **Seam-aware evaluation.** Global, boundary-band, and interior PSNR, value and gradient jumps across every interface, rendering PSNR/SSIM from a fixed camera and transfer function, plus throughput and memory.

The claim we are testing: with only 4 ghost cells, blending plus fine-tuning should match or beat the paper's 16-ghost-cell setup on seam quality and accuracy.

**Bottom line after run 2 (section 5b).** The claim holds over 3 seeds: g4 + fine-tune + blend beats the paper's g16 by +2.0 dB global PSNR, +2.5 dB boundary PSNR and +2.1 dB render PSNR, with about 580x smaller seam jumps, at the same model size, winning on 3 of 3 seeds. Two things changed the story, though:

1. A reconstruction-only fine-tune control (same 1,500 decoder steps, consistency weights set to 0) does slightly *better* than our consistency fine-tune. So the fine-tune gain comes from extra decoder training, not from the consistency loss.
2. The paper's g16 catches up on global PSNR only when given 2.3x the parameters (32^3 grids), and it still has 444x larger seam jumps.

The defensible contribution is **overlap blending + narrow overlap + a short decoder fine-tune**. The consistency loss is an ablation with a marginal effect.

## 2. Where everything lives

| What | Path |
|---|---|
| This worklog | `Computer_Graphics/SEAMLESS_DELTA_WORKLOG/README.md` |
| Frozen results for run 1 | `Computer_Graphics/SEAMLESS_DELTA_WORKLOG/results/2026-11-12_run1/` |
| Frozen results for run 2 (seeds, capacity, controls) | `Computer_Graphics/SEAMLESS_DELTA_WORKLOG/results/2026-11-12_run2_seed_capacity/` |
| Run-2 live output (per-run metrics, renders) | `APMGSRN/Output/SeedCapacityStudy/` |
| Run-2 runner / aggregator / speed benchmark | `APMGSRN/run_seed_capacity_study.ps1`, `Code/aggregate_study.py`, `Code/benchmark_throughput.py` |
| Paper PDF and text | `Computer_Graphics/paper/` |
| Original study notes (pre-delta) | `Computer_Graphics/PROJECT_STUDY.md` |
| Research proposal | `Computer_Graphics/DELTA_PROPOSAL.md` |
| Junior-engineer runbook (the plan we implemented) | `Computer_Graphics/IMPLEMENTATION_PLAN_SEAMLESS_APMGSRN.md` |
| Code repo | `Computer_Graphics/APMGSRN` on branch `seamless-apmgsrn` |
| Repo-level code doc | `APMGSRN/SEAMLESS_APMGSRN.md` |
| Live pipeline output (overwritten each run) | `APMGSRN/Output/SeamlessAPMGSRN/` |
| Checkpoints | `APMGSRN/SavedModels/Plume_APMGSRN_dd_{g1,g4,g16,g4_consistency}` |
| Python environment | `D:\Computer_Graphics_env\.venv` (PyTorch 2.11 + CUDA 12.8) |

## 3. Code changes (branch `seamless-apmgsrn`)

| Commit | Content |
|---|---|
| `fd02e50` | Core delta: geometry utilities, blend router, fine-tuner, seam evaluator, renderer, summarizer, configs, tests |
| `8f0cde9` | Line profile and slices now pass through the plume; `start_jobs.py --skip_existing` resumes a crashed queue |
| `89027b1` | Vectorized blend weights (one broadcast for all bricks), optional exact core/overlap split, equivalence test |
| `15b80ad` | `train.py --seed`, `compare_renderings.py --montage_runs`, study runner and aggregator |
| `74443a3` | Benchmark-median throughput in all tables and plots, headline figure, cost table, recon-only controls, fair throughput benchmark, study job settings |

Diff from upstream: 21 files changed, about 1,865 insertions. Key files:

| File | Role |
|---|---|
| `Code/Models/ensemble_utils.py` | All decomposition geometry, DHW/XYZ conversion, face adjacency, smoothstep weights |
| `Code/Models/ensemble_SRN.py` | `forward_hard` (original, bit-for-bit unchanged), `forward_blend`, `forward_child` |
| `Code/finetune_boundaries.py` | Decoder-only interface-consistency fine-tune with time, memory, and divergence guards |
| `Code/evaluate_seams.py` | All seam metrics, profiles, slice error images |
| `Code/compare_renderings.py` | Self-contained volume renderer (no nerfacc), PSNR/SSIM, montage |
| `Code/summarize_seamless.py` | Tables and plots |
| `run_seamless_pipeline.ps1` | Fine-tune, evaluate all variants, render, summarize |
| `Tests/` | 19 CPU unit tests |

Two design details are worth stating in the presentation:

1. **Ghost-cell convention.** The repo uses `actual = ensemble_ghost_cells + 1`, so configs `0/3/15` mean an actual overlap of 1/4/16 cells (`g1/g4/g16`).
2. **Blending weights.** Both neighbours ramp across the same full overlap in opposite directions, and `smoothstep(t) + smoothstep(1 - t) = 1` makes the result an exact partition of unity centred at the original split. Unit tests check this, along with 1, 2, 4, and 8 simultaneously active bricks (face, edge, and corner) and finite autograd gradients.

## 4. Experimental setup (run 1)

1. Dataset: Plume, `512 x 128 x 128` (DHW), `APMGSRN/Data/Plume.nc`.
2. Decomposition: `2 x 2 x 2`, 8 bricks, identical per-brick architecture in all conditions (8 grids, 24^3 features, 2 x 64 decoder, 923,136 parameters total).
3. Training: 5,000 iterations per brick, 32,768 points per iteration, lr 0.01, single seed. Nothing hit the 4-minute per-brick cap.
4. Fine-tune (g4 only): 1,500 iterations, 4,096 interface points + 2,048 interior anchors, lr 5e-4, value weight 1.0, gradient weight 0.05, 250 warmup iterations, seed 42. Took 167 s and 0.21 GB peak GPU memory.
5. Evaluation: exact full-volume PSNR, a 2-cell boundary band, 65,536 face samples per interface, throughput on 262,144 random points x 20 repeats.
6. Rendering: 512 x 256, 768 samples per ray, same camera and transfer function for the raw volume and every model.

## 5. Results (run 1)

Value jump means the mean |left minus right| of the routed output across each interface. Lower is better for jumps; higher is better for PSNR and SSIM.

| Variant | Global PSNR | Boundary PSNR | Value jump | Gradient jump | Render PSNR | Render SSIM |
|---|---|---|---|---|---|---|
| g1 hard | 50.92 | 47.37 | 2.64e-3 | 2.89e-3 | 43.11 | 0.9919 |
| g4 hard | 50.25 | 46.63 | 2.48e-3 | 2.59e-3 | 42.63 | 0.9911 |
| g4 blend | 50.43 | 47.79 | 5.86e-6 | 2.23e-3 | 42.79 | 0.9915 |
| g4 consistency, hard | 50.40 | 46.97 | 1.99e-3 | 2.53e-3 | 43.03 | 0.9915 |
| **g4 consistency, blend (ours)** | **50.53** | **47.86** | **5.76e-6** | **2.20e-3** | **43.16** | **0.9918** |
| g16 hard (paper-style) | 48.47 | 45.45 | 3.26e-3 | 2.92e-3 | 40.97 | 0.9876 |
| g16 blend | 49.19 | 47.19 | 6.44e-6 | 2.40e-3 | 41.77 | 0.9894 |

The ground-truth gradient change across the same interfaces is 1.75e-3 for the g4 geometry, which is the floor a perfect model would reach on the gradient-jump metric.

Headline numbers:

1. **Blending removes the value discontinuity.** g4 goes from 2.48e-3 to 5.86e-6, about 420x lower, on the same checkpoint with zero training. Worst-case jump drops from 0.135 to 0.00044.
2. **Ours vs the paper-style 16-cell overlap.** g4 consistency + blend beats g16 hard by +2.06 dB global PSNR, +2.41 dB boundary PSNR, +2.19 dB render PSNR, SSIM 0.9918 vs 0.9876, and a 566x lower value jump. It also uses 4 overlap cells instead of 16.
3. **Fine-tuning helps on its own.** Pairwise brick disagreement drops 19% (2.51e-3 to 2.02e-3), and global PSNR rises 0.15 dB rather than falling. In hard mode alone it adds +0.40 dB render PSNR.
4. **Gradient jumps improve but are not solved.** We get 15% lower than g4 hard (2.59e-3 to 2.20e-3), still about 26% above the ground-truth floor.
5. **Versus a single 1-cell overlap.** Ours is 0.39 dB lower in global PSNR but 0.49 dB higher at the boundary, slightly higher in render PSNR, and has a 460x lower jump.

Overhead:

| Measure | Hard | Blend |
|---|---|---|
| Raw query throughput (Mpts/s, RTX 3050) | about 26-28 | about 12-16 |
| Render frame time, g4 (s) | 1.31 | 1.28-1.35 |
| Peak GPU memory (GB) | 0.181 | 0.182 |

Raw point queries get about 2x slower, because roughly 12% of uniform points fall in an overlap and need 2 to 8 brick evaluations. Render frame time for g4 was not measurably affected, and memory is unchanged. g16 blend frames are slower (2.6 s) because its overlaps cover far more of the volume. Throughput on this laptop is noisy between runs. In run 1, g4 consistency hard measured 12.4 Mpts/s even though its architecture and routing are identical to g4 hard, which shows the noise is large. Report throughput as a median over several repeated runs before quoting exact overhead numbers.

> **Superseded in run 2.** The "about 2x slower" figure above is wrong. It came from single noisy timings. The fair benchmark in run 2 loads every model once, then measures 262,144 points x 10 repeats in 7 interleaved rounds and reports the median. With that method blending is about **22% slower** (hard 11.5-12.2 Mpts/s, blend 8.7-9.6 Mpts/s). See section 5b.

Figures in `results/2026-11-12_run1/figures/`:

1. `montage.png`: renders of all seven variants against ground truth, a zoom on the split, and 8x error images.
2. `seam_metrics.png`: boundary PSNR, value jump (log scale), and gradient jump with the ground-truth line.
3. `line_profile.png`: dense profile across the width split, through the plume.
4. `slice_error_g4_hard.png` and `slice_error_g4_consistency_blend.png`: before and after error slices.

## 5b. Results (run 2: three seeds, capacity sweep, fine-tune controls)

### What changed vs run 1

1. **Seeds.** 42, 1 and 2 for the key conditions (`train.py --seed`). Seed 42 at 24^3 reuses the run-1 checkpoints.
2. **Same model size everywhere** for the main comparison: 8 grids, 24^3 features, 2 x 64 decoder, 923,136 parameters, 3.52 MB fp32 for every 24^3 row. Capacity was then varied on purpose for g16 only: 27^3 (1.30M params) and 32^3 (2.14M params, 2.3x).
3. **Controls.** `reconft` means a decoder-only fine-tune with the same 1,500 iterations and anchors, but with value weight 0 and gradient weight 0, so it is just extra reconstruction training. It was run for g4 (seeds 2 and 42) and g16 (seed 42).
4. **Fair speed benchmark.** Described in the note in section 5.
5. **GPU budget.** Ran out before these could run: g16 27^3 seed 1, g4 control seed 1, g1 seeds 1/2, g2, and the 16^3 conditions.

### Main table (Plume 2x2x2; mean +/- std over seeds; n = number of seeds)

| Method | n | Params | Global PSNR | Boundary PSNR | Render PSNR | SSIM | Value jump | Grad jump | Mpts/s |
|---|---|---|---|---|---|---|---|---|---|
| g1 hard (paper, minimum overlap) | 1 | 0.92M | 50.92 | 47.37 | 43.11 | 0.9919 | 2.64e-3 | 2.89e-3 | 11.5 |
| g4 hard | 3 | 0.92M | 50.16 +/- 0.08 | 46.56 +/- 0.17 | 42.23 +/- 0.85 | 0.9907 | 2.69e-3 | 2.50e-3 | 11.5 |
| **g16 hard (paper default)** | 3 | 0.92M | 48.66 +/- 0.43 | 45.48 +/- 0.38 | 41.25 +/- 0.54 | 0.9883 | 3.30e-3 | 2.93e-3 | 11.9 |
| g16 hard, 27^3 grids | 1 | 1.30M | 49.57 | 46.53 | 42.23 | 0.9902 | 2.75e-3 | 3.05e-3 | 11.7 |
| g16 hard, 32^3 grids | 1 | 2.14M | 50.62 | 47.45 | 43.57 | 0.9924 | 2.56e-3 | 2.99e-3 | 11.8 |
| g16 + recon-FT (equal extra training) | 1 | 0.92M | 49.04 | 46.09 | 42.08 | 0.9893 | 2.56e-3 | 2.92e-3 | 11.5 |
| g16 blend | 3 | 0.92M | 49.32 +/- 0.36 | 47.17 +/- 0.16 | 42.06 +/- 0.45 | 0.9900 | 6.34e-6 | 2.43e-3 | 8.7 |
| g4 blend (no training) | 3 | 0.92M | 50.35 +/- 0.07 | 47.80 +/- 0.01 | 42.42 +/- 0.82 | 0.9911 | 5.81e-6 | 2.15e-3 | 9.2 |
| g4 consistency-FT, hard | 3 | 0.92M | 50.53 +/- 0.31 | 47.17 +/- 0.26 | 43.19 +/- 0.16 | 0.9918 | 1.97e-3 | 2.45e-3 | 12.1 |
| **g4 consistency-FT + blend (OURS)** | 3 | 0.92M | 50.66 +/- 0.31 | 48.00 +/- 0.21 | 43.33 +/- 0.17 | 0.9921 | 5.69e-6 | 2.12e-3 | 9.1 |
| g4 recon-FT, hard (control) | 2 | 0.92M | 50.83 +/- 0.31 | 47.45 +/- 0.33 | 43.42 +/- 0.18 | 0.9921 | 2.00e-3 | 2.47e-3 | 12.2 |
| **g4 recon-FT + blend (OURS, best)** | 2 | 0.92M | 50.98 +/- 0.29 | 48.43 +/- 0.25 | 43.58 +/- 0.19 | 0.9924 | 5.71e-6 | 2.14e-3 | 9.6 |

Ground-truth gradient change across the g4 interfaces is 1.75e-3, which is the floor for the gradient-jump column.

### Paired differences (same seed, ours minus baseline)

| Comparison | Global | Boundary | Render | Value jump | Grad jump | Seeds won |
|---|---|---|---|---|---|---|
| OURS vs g16 hard (paper) | +2.00 +/- 0.12 dB | +2.51 dB | +2.08 dB | 579x lower | 1.38x lower | 3/3 on all |
| g4 blend alone vs g16 hard | +1.69 dB | +2.32 dB | +1.16 dB | 567x lower | 1.36x lower | 3/3 (render 2/3) |
| OURS vs g4 hard | +0.51 dB | +1.44 dB | +1.10 dB | 474x lower | 1.18x lower | 3/3 on all |
| Blend alone vs g4 hard | +0.20 +/- 0.03 dB | +1.24 dB | +0.19 dB | 463x lower | 1.16x lower | 3/3 on all |
| OURS vs g16 + recon-FT (fair extra training) | +1.50 dB | +1.77 dB | +1.08 dB | 445x lower | 1.32x lower | 1/1 |
| recon-FT + blend vs g16 + recon-FT | +1.73 dB | +2.17 dB | +1.36 dB | 442x lower | 1.31x lower | 1/1 |
| OURS vs g16 32^3 (2.3x params) | -0.09 dB | +0.40 dB | -0.41 dB | 444x lower | 1.36x lower | 1/1 |
| **OURS (consistency-FT) vs recon-FT control** | **-0.21 dB** | **-0.38 dB** | **-0.25 dB** | 1.01x | 1.01x | **0/2** |

### Capacity: how much bigger must g16 get to catch up?

| g16 grids | Params | g16 global PSNR | OURS (24^3) minus g16 | g16 value jump vs OURS |
|---|---|---|---|---|
| 24^3 | 0.92M | 48.66 | +2.00 dB | 580x larger |
| 27^3 | 1.30M | 49.57 | +0.97 dB | 478x larger |
| 32^3 | 2.14M | 50.62 | +0.04 dB (tie) | 444x larger |

g16 needs about 2.3x the parameters (8.15 MB vs 3.52 MB) to tie our global PSNR. Even then we are +0.40 dB better in the boundary band and about 444x smoother at the seam, but g16 32^3 has slightly higher render PSNR (43.57 vs 43.33). The recon-FT + blend variant ties it at 43.58.

### The negative finding (say it out loud)

The consistency loss does what it is designed to do, but by very little. Against the control, it lowers pairwise brick disagreement by only about 3.5% and the gradient jump by about 1%, and it loses 0.2-0.4 dB of accuracy on 2 of 2 seeds. Run 1's "fine-tuning helps" result was real, but the help comes from 1,500 extra decoder steps, not from the consistency term. Once blending is on, the value seam is already at the numerical floor (about 6e-6), so there is nothing left for a value-consistency loss to fix. Recommended framing: **blend + any short decoder fine-tune**, with consistency vs reconstruction-only as an ablation.

### Costs and trade-offs

| Measure | Hard routing (paper) | Blend (ours) | Note |
|---|---|---|---|
| Query throughput, median of 7 | 11.5-12.2 Mpts/s | 8.7-9.6 Mpts/s | about 22% slower; overlap points evaluate 2-8 bricks |
| Render frame time (512x256, 768 spp) | 3.5-4.3 s | 3.5-4.3 s | no consistent penalty |
| Peak GPU memory (eval) | 0.181 GB | 0.182 GB | unchanged |
| Model size | 3.52 MB | 3.52 MB | blend adds no parameters |
| Stored voxels (overlap redundancy) | g4 1.15x, g16 1.66x | same | g4 stores 30% fewer training voxels than g16 |
| Training | about 8 min / 8 bricks | +0 (blend) or +4-7 min (fine-tune) | blend works on existing checkpoints |

Note: training wall time depends on GPU contention. The run-1 g16 bricks took about 25 s each, while the run-2 bricks with identical settings took about 60 s, so wall time is not a meaningful per-method cost here.

### Figures (`results/2026-11-12_run2_seed_capacity/plots/`)

1. `headline.png`: paper g16 (same size and 2.3x size) vs g4 hard vs blend vs both "ours" variants, with seed dots. **Use this one in the talk.**
2. `main_comparison.png`: every condition, six panels, including the gradient-jump floor and the benchmark throughput.
3. `capacity_curve.png`: g16 PSNR vs parameters against our single 0.92M point.
4. `tradeoffs.png`: accuracy vs storage, speed, and seam jump.
5. `ghost_cell_curve.png`: g1/g4/g16 at equal size for hard, blend and ours.
6. `per_seed_paired.png`: same-seed lines from g16 to ours.
7. `render_montage.png`: seed-42 renders, a zoom on the split, and 8x error images (the g16 seam is visible as a vertical streak).

Tables: `comparison_main.md/.csv`, `paired_tests.md`, `comparison_capacity.md`, `comparison_costs.md`, `throughput_benchmark.csv`, `all_runs.csv`.

## 6. Honest caveats (say these before the professor asks)

1. **Seeds (updated in run 2).** The key conditions now have 3 seeds and the big effects hold on every seed. The g1, g16 27^3/32^3, and g16 control rows are still single-seed, and the g4 control has 2 seeds. One dataset only (Plume).
2. **g16 capacity (tested in run 2).** At equal size g16 loses by 2 dB. With 2.3x parameters it ties our global PSNR and slightly beats us in render PSNR, but it still has about 444x larger seam jumps. So "16 to 4 cells" is a quality claim at equal size, and a seam-smoothness claim at any size we tested.
3. **The consistency loss is not the source of the gain (run 2).** The reconstruction-only control matches or beats it. Present it as an ablation.
4. **g1 hard has higher global PSNR than our consistency variant** (50.92 vs 50.66, single seed), but it has 460x larger seam jumps and lower boundary PSNR. The recon-FT + blend variant (50.98) edges past it.
5. **Render PSNR is noisy** (g4 hard ranges 41.3-42.8 across seeds). Rely on volume PSNR for small differences.
6. **The value jump near 6e-6 is not exactly zero.** The blended field is continuous, but the metric samples finitely on both sides of each interface, so a small residual from field slope remains. Treat it as "at numerical floor", not a perfect zero.
7. **Gradient jumps measure C1 smoothness only approximately.** They use finite differences, and part of what remains is each brick's own gradient noise rather than an interface effect.
8. **Our renderer is not the paper's renderer.** Render PSNR is comparable between our variants, not to the paper's numbers.

## 7. Compute budget used

| Step | GPU time |
|---|---|
| Train g1 / g4 / g16 (8 bricks each) | 8.8 / 8.1 / 3.6 min |
| Fine-tune g4 | 2.8 min |
| Evaluate 7 variants + render + summarize (x3 pipeline runs) | about 3 min per run |
| Run-1 total | about 35 minutes |
| Run 2: 11 new 8-brick trainings (seeds, 27^3, 32^3) | about 90 min (bricks ran slower under contention) |
| Run 2: 5 fine-tunes (2 consistency, 3 recon-only) + evaluation + renders + benchmark | about 5 min plus evaluation (CPU/GPU light) |
| **Grand total** | **about 2 h 10 min** of GPU time. The budget is spent; do not start more training without asking. |

## 8. How to reproduce

Run from `APMGSRN`:

```powershell
$py = 'D:\Computer_Graphics_env\.venv\Scripts\python.exe'
& $py -m pytest -q Tests
foreach ($g in 'g1','g4','g16') {
  & $py Code/start_jobs.py --settings "train_plume_dd_$g.json" --devices cuda:0 --data_devices same --skip_existing true
}
./run_seamless_pipeline.ps1                # fine-tune + evaluate + render + summarize
./run_seamless_pipeline.ps1 -SkipFinetune  # re-evaluate only
```

Then copy `Output/SeamlessAPMGSRN/` into a new dated folder under `results/` and add a log entry below.

Run 2 (seeds + capacity + controls). The runner is resumable and skips any step whose output already exists:

```powershell
./run_seed_capacity_study.ps1 -TrainDeadline 14:00      # train/fine-tune/evaluate/render in priority order
./run_seed_capacity_study.ps1 -SkipTraining              # re-evaluate and re-render only
& $py Code/aggregate_study.py --output_root Output/SeedCapacityStudy   # tables + plots (CPU only, seconds)
```

Machine pitfalls:

1. The C: drive is nearly full. Keep environments and large outputs on D:.
2. The page file is small. Run one PyTorch training process at a time and always pass `--data_devices same`. Using `cpu` spawns 4 DataLoader workers and triggers `WinError 1455`.
3. Launch long jobs as detached processes with output going to a log file, so they survive a closed terminal or agent session. Use `--skip_existing true` to resume an interrupted queue.

## 9. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-12 | Delta = blending + interface fine-tune + seam metrics | Fits about 1 week, under 2 GPU-hours, builds visibly on the paper's own ghost-cell limitation |
| 2026-11-12 | Plume, 2x2x2, conditions g1/g4/g16 | Smallest setup that reproduces the paper's ghost-cell trade-off on a 4 GB GPU |
| 2026-11-12 | Hard router kept bit-for-bit, blend opt-in | Baseline numbers stay valid; a unit test checks equivalence |
| 2026-11-12 | Ramps span the full overlap from both sides | Exact partition of unity centred on the original split |
| 2026-11-12 | Fine-tune decoders only, features frozen | Cheap (37k parameters), stable, cannot destroy the learned grids |
| 2026-11-12 | Own renderer instead of nerfacc | nerfacc does not build on this Windows/PyTorch setup; identical camera for all variants |
| 2026-11-12 | Vectorized full blend as default; core/overlap split kept as an option | The split was exact but slower on this GPU (13.7 vs 15.4 Mpts/s) |
| 2026-11-12 | Run 2: 3 seeds for g4/g16 core conditions; capacity sweep for g16 only | Answers "is it seed noise?" and "is g16 just starved of capacity?" within the remaining GPU budget |
| 2026-11-12 | Added reconstruction-only fine-tune control | Without it the fine-tune gain could not be attributed to the consistency loss. It turned out not to be. |
| 2026-11-12 | Throughput reported as median of 7 interleaved rounds | Single timings varied by 2x for identical models |
| 2026-11-12 | Headline variant = blend + any decoder fine-tune; consistency loss = ablation | The control result (0/2 seeds for consistency vs recon-only) |
| 2026-11-12 | Stopped training at about 2 h 10 min total GPU | The user's GPU cap. Remaining conditions are listed as next steps. |

## 10. Log

### 2026-11-12

1. Implemented the full delta (commit `fd02e50`). 18 unit tests passed, smoke tests passed on GPU and CPU.
2. Trained g1, g4, g16. The first queue died after 7 of 8 g4 bricks when the agent session reset. Added `--skip_existing`, retrained only the missing brick, then trained g16 (commit `8f0cde9`).
3. Fine-tuned g4 with value + gradient consistency: 1,500 iterations, no divergence, reconstruction loss held at about 1.0e-5.
4. Ran the full pipeline. All 7 variants were evaluated and rendered. Results are in section 5.
5. Vectorized the blend weights and added an exact core/overlap split with an equivalence test (19 tests now). Re-ran the pipeline; quality numbers were identical (commit `89027b1`).
6. Created this worklog and froze run-1 results into `results/2026-11-12_run1/`.

### 2026-11-12 (run 2: seeds, capacity, controls)

1. Added `train.py --seed`, a montage selector, a resumable study runner, and an aggregator (commit `15b80ad`).
2. Trained g4 and g16 at seeds 1 and 2, g16 at 27^3 and 32^3, g1 seed 42 (reused), plus consistency fine-tunes for g4 seeds 1/2. Every 24^3 model was checked to have 923,136 parameters.
3. Added reconstruction-only fine-tune controls (g4 seeds 2/42, g16 seed 42). The control beat the consistency loss by 0.21 dB global on 2 of 2 seeds. Recorded as a negative finding and reframed the contribution.
4. Wrote `benchmark_throughput.py`. The fair median benchmark shows blending is about 22% slower, not 2x. Corrected section 5.
5. Fixed the aggregator: benchmark-median throughput in all tables and plots, `_reconft` train-time lookup, shorter labels, a headline figure, and correct tradeoff labels (commit `74443a3`). 19 tests pass.
6. Froze everything into `results/2026-11-12_run2_seed_capacity/`.

## 11. Next steps

Done in run 2: extra seeds for the core conditions; a fine-tune ablation, done as recon-only vs consistency, which is the more informative control.

Remaining, only if more GPU time is approved (in priority order):

1. A third seed for the g4 recon-FT control, and seeds 1/2 for g16 32^3 (about 40 GPU-minutes). This would turn the "2.3x params to tie" and control results from single or 2-seed results into 3-seed ones.
2. **g2 condition** (config `ensemble_ghost_cells 1`), to test whether blending works with an even narrower overlap. Untested so far.
3. Seeds 1/2 for g1. It is the strongest hard-routing baseline on global PSNR.
4. A second dataset (for example a smaller scalar field) to check that the result generalizes.

**Presentation:** lead with `headline.png`. Then the capacity curve ("g16 needs 2.3x params to tie, and still has seams"), then the montage error row. Present the control slide honestly: blending is the main contribution, and the fine-tune helps through extra training.
5. Optional: render the original seam-heavy view in the paper's own renderer (`Code/renderer.py --ensemble_routing_mode blend`) for a figure directly comparable to the paper.

## Presentation (2026-11-12)

- 26-slide, 30-minute deck covering preliminaries, problem, implementation, results and extensions.
- Markdown version with speaker notes: presentation/presentation.md
- HTML source: presentation/html/slide01.html to slide26.html, styled by presentation/html/style.css (light theme, glass panels, green accent only).
- PPTX: presentation/output/Seamless_APMGSRN_presentation.pptx. Each slide is a full-bleed 1920x1080 render of its HTML, and speaker notes are attached.
- Rebuild: render the HTML to build/png with headless Edge, then run node build/build_pptx.js.
- Every number in the deck comes from the run-2 results above.

