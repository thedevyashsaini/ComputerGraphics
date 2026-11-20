# Recommended Delta: Seamless APMGSRN

The implementation-level runbook is available in `IMPLEMENTATION_PLAN_SEAMLESS_APMGSRN.md`. That file is the authoritative step-by-step guide; this document remains the research proposal and rationale.

## Working title

**Seamless APMGSRN: Continuity-Aware Blending for Domain-Decomposed Neural Volume Representations**

## 1. One-sentence pitch

Replace APMGSRN's hard brick selection with smooth partition-of-unity blending in overlap regions, then optionally fine-tune neighboring brick decoders with an interface-consistency loss to remove rendering seams without the PSNR loss and redundant overlap of large ghost-cell regions.

## 2. Why this is the best one-week delta

- It directly addresses a limitation and future-work item explicitly stated in the paper.
- The repository already contains domain partitioning, ghost cells, model loading, and rendering.
- The main inference modification is localized to `Ensemble_SRN.forward`.
- The first useful result requires no retraining: blend already-trained overlapping models.
- A lightweight fine-tuning phase makes the contribution more than a rendering trick.
- The improvement is visually obvious in a zoomed volume render.
- It permits clean quantitative evaluation beyond generic PSNR.
- It fits a 4 GB GPU because only neighboring small models need to be loaded/fine-tuned at once.

## 3. Research question and hypothesis

**Research question:** Can a continuity-aware overlap mechanism remove domain-decomposition seams while preserving global reconstruction quality and using fewer ghost cells than the original approach?

**Hypothesis:** Smoothly blending neighboring predictions and explicitly penalizing disagreement in overlap regions will substantially reduce value and gradient jumps at brick interfaces. It should match the visual quality of a 16-ghost-cell baseline using only 2-4 ghost cells, with less redundant training volume and a smaller global-PSNR penalty.

## 4. Baseline failure

The current ensemble makes a hard decision:

```text
prediction(x) = model_hash(x)(x)
```

At a boundary between models `i` and `j`:

```text
model_i(x_boundary) need not equal model_j(x_boundary)
```

Even small scalar discontinuities become strong visual seams after transfer-function classification and alpha compositing. Gradient discontinuities can also change shading or isosurface normals.

Ghost cells expose both models to neighboring data, but hard routing discards one prediction and independent training does not force agreement.

## 5. Proposed method

### 5.1 Stage A: smooth overlap blending

For a coordinate inside the support of multiple brick models, evaluate all valid neighbors and compute normalized spatial weights:

```text
y(x) = sum_i w_i(x) * f_i(x) / sum_i w_i(x)
```

Each `w_i` is a separable product of smooth one-dimensional window functions. It is 1 in the brick core and falls smoothly to 0 across the ghost overlap. A cubic smoothstep is sufficient:

```text
smoothstep(t) = t^2 * (3 - 2t),  t in [0, 1]
```

This gives a partition-of-unity-style fusion: the weights sum to one after normalization and the prediction changes smoothly instead of switching abruptly.

Recommended implementation behavior:

- Query one model in brick interiors.
- Query at most 2 models near a face.
- Query at most 4 near an edge.
- Query at most 8 near a corner.
- Preserve the old hard-router mode for exact baseline comparison.

### 5.2 Stage B: boundary-consistency fine-tuning

For adjacent models `f_i` and `f_j`, sample identical global coordinates from their shared overlap. Fine-tune their decoders with:

```text
L_total = L_reconstruction + lambda_value * L_value + lambda_grad * L_grad

L_value = mean((f_i(x) - f_j(x))^2)
```

The optional gradient term is:

```text
L_grad = mean(||grad f_i(x) - grad f_j(x)||^2)
```

For the one-week version, use a finite-difference normal derivative instead of full second-order autograd if memory is tight. Begin with value consistency; add normal-gradient consistency only if Stage A and value consistency are stable.

To prevent the interface objective from degrading the learned field:

- retain reconstruction supervision from the raw volume in the overlap;
- freeze adaptive grid transformations;
- initially freeze feature grids and tune only the small MLP decoders;
- use a small learning rate and 500-2,000 fine-tuning iterations per neighboring pair;
- ramp the consistency weight from zero over the first part of fine-tuning.

### 5.3 Optional confidence-aware weighting

If time remains, weight each model by both spatial support and local residual statistics collected during training:

```text
w_i'(x) = w_spatial_i(x) / (epsilon + estimated_error_i(x))
```

This is a stretch goal. Do not make it part of the minimum viable contribution.

## 6. Exact code changes

### Required

1. `Code/Models/ensemble_SRN.py`
   - Add `routing_mode = hard | blend`.
   - Store full support and core extents for every brick.
   - Find candidate neighboring bricks for overlap coordinates.
   - Compute smooth weights and weighted predictions.
   - Batch model queries by model index to avoid per-point Python calls.

2. `Code/Models/options.py`
   - Add `ensemble_routing_mode`.
   - Add `ensemble_blend_width` or infer it from ghost cells.
   - Add consistency fine-tuning settings.

3. `Code/start_jobs.py`
   - Save unambiguous core and ghost extents.
   - Optionally write a brick-adjacency JSON manifest.

4. New `Code/finetune_boundaries.py`
   - Enumerate face-adjacent model pairs.
   - Sample common global points in the overlap.
   - Convert those points into both models' local coordinates.
   - Optimize reconstruction plus consistency loss.
   - Save fine-tuned checkpoints separately from baselines.

5. `Code/test.py`
   - Add boundary-band PSNR.
   - Add value-jump and normal-gradient-jump metrics.
   - Add inference throughput comparison for hard versus blend routing.

### Useful support code

- `Code/Other/seam_metrics.py`
- `Code/BatchRunSettings/train_seamless_plume.json`
- `Code/BatchRunSettings/test_seamless_plume.json`
- a script to render identical camera/transfer-function views for every ablation.

## 7. Metrics that make the project research-grade

Generic global PSNR alone can hide a thin but visually severe seam. Report all of the following:

### Data-space metrics

1. **Global PSNR**
   - Ensures the method does not damage overall reconstruction.

2. **Boundary-band PSNR**
   - PSNR only within `k` voxels of decomposition planes.

3. **Interface value jump**
   - Sample paired points immediately to either side of every interface.
   - Report mean and 95th-percentile absolute prediction difference.

4. **Interface normal-gradient jump**
   - Compare finite-difference derivatives normal to the interface.

### Image-space metrics

5. **Rendered-image PSNR** against raw-data rendering.
6. **SSIM** against raw-data rendering.
7. Optional **LPIPS** if installation is easy.

### Cost metrics

8. Training and fine-tuning time.
9. Peak GPU memory.
10. Inference throughput and render time.
11. Storage size, expected to remain almost unchanged.

## 8. Minimum experiment matrix

Use the same architecture and total parameter count for every row.

| ID | Ghost cells | Routing | Fine-tuning | Purpose |
|---|---:|---|---|---|
| A | 1 | Hard | None | Original baseline |
| B | 16 | Hard | None | Paper's expensive seam-reduction baseline |
| C | 2 or 4 | Smooth blend | None | Tests inference-only contribution |
| D | 2 or 4 | Smooth blend | Value consistency | Main proposed method |
| E | 2 or 4 | Smooth blend | Value + gradient consistency | Stretch ablation |

If GPU time is extremely limited, run A, C, and D first. B can use a shorter training schedule or an existing checkpoint if available.

## 9. Dataset plan

### Primary

**Plume** from the official archive:

- `128 x 128 x 512`
- Small enough for the laptop.
- Contains localized structures that suit APMGSRN.
- Already used by the reference `train.json`.

Use a `2 x 2 x 2` decomposition and sequentially train the eight bricks.

### Backup / debugging

Generate an analytic `128^3` volume containing:

- two Gaussian blobs;
- a thin shell or torus crossing brick boundaries;
- a high-frequency plume-like perturbation.

The known analytic structure makes interface errors easy to interpret and avoids blocking on external data.

### Optional second dataset

Use a downsampled Supernova volume only after the Plume pipeline is complete.

## 10. GPU-time budget

Target configuration for the RTX 3050 4 GB:

- reduce `points_per_iteration` from 100,000 to 32,768 or 65,536 if needed;
- use 8-16 grids per brick;
- use 1-2 features per grid;
- use `16^3` or `24^3` feature grids during development, then `32^3` for the final run if memory permits;
- train 5,000-10,000 iterations;
- render at `256 x 256` or `512 x 512` with moderate samples per ray;
- sequential brick training on the single GPU.

Expected disciplined budget:

- debugging/synthetic run: 10-30 GPU minutes;
- final eight-brick Plume baseline: roughly 20-60 GPU minutes depending on PyTorch/TCNN availability;
- consistency fine-tuning and ablations: roughly 30-60 GPU minutes;
- final renders and metric sweeps: roughly 15-30 GPU minutes.

The exact time must be measured after environment setup. A PyTorch fallback without Tiny-CUDA-NN will be slower.

## 11. One-week execution plan

### Day 1 - Environment and baseline

- Create a modern PyTorch CUDA environment.
- Download Plume data.
- Run a tiny single-model smoke test.
- Reproduce the hard-routed `2 x 2 x 2` seam.

### Day 2 - Seam metric suite

- Implement interface plane discovery.
- Add boundary-band PSNR and value-jump metrics.
- Produce the baseline seam heatmap and render.

### Day 3 - Smooth blending

- Implement candidate neighbor routing.
- Implement smooth overlap weights.
- Benchmark correctness, memory, and throughput.

### Day 4 - Consistency fine-tuning

- Implement adjacent-pair overlap sampling.
- Fine-tune decoder parameters with value consistency.
- Check that global PSNR does not regress substantially.

### Day 5 - Ablations

- Hard 1 ghost versus hard 16 ghost versus blend versus blend+consistency.
- Tune blend width and consistency weight.
- Add normal-gradient consistency only if stable.

### Day 6 - Final results

- Generate identical data-space slices and rendered views.
- Create plots for seam metrics, PSNR, and overhead.
- Select the most convincing transfer function and camera.

### Day 7 - Report and presentation

- Explain failure, method, equations, and implementation.
- Show zoomed seam comparisons and error heatmaps.
- Present an ablation table and limitations honestly.

## 12. Expected result pattern

The ideal outcome is:

- hard routing with one ghost cell has the largest interface jump and a visible seam;
- 16 ghost cells reduces the seam but sacrifices global PSNR/training efficiency;
- smooth blending sharply reduces visible seams without retraining;
- consistency fine-tuning further lowers boundary error and gradient jumps;
- global PSNR stays close to the one-ghost baseline;
- inference slows only for points inside narrow overlap bands.

Even if consistency fine-tuning gives only a modest gain, Stage A remains a valid and useful result, and the negative/neutral fine-tuning result can be explained through the ablation.

## 13. How to present it convincingly

Use a clean research narrative:

1. The original method solves capacity allocation and scaling, but independent bricks create discontinuities.
2. Large ghost regions hide the seam inefficiently.
3. The proposed method treats the ensemble as overlapping local neural fields.
4. Partition-of-unity blending guarantees a smooth transition in the overlap.
5. Consistency fine-tuning makes neighboring local fields agree before blending.
6. Specialized seam metrics reveal improvements that global PSNR misses.

Best figures:

- a diagram of two overlapping bricks and their weight ramps;
- a zoomed render with the decomposition plane marked;
- a boundary-error heatmap;
- a line profile crossing the boundary;
- an ablation table with global PSNR, boundary PSNR, jump, render SSIM, and time.

This looks substantial because it is a complete method-evaluation package: a defined failure mode, mathematical solution, training component, efficient implementation, specialized metrics, ablation study, and visual validation.

## 14. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| Modern CUDA dependencies fail | Start with pure PyTorch; renderer can be tested after model training works |
| 4 GB VRAM overflow | Reduce point batch, grid count, and feature-grid resolution |
| Eight-brick training takes too long | Use 5,000 iterations and the analytic dataset for development |
| Blending lowers sharpness | Narrow the blend band or use confidence/residual-aware weights |
| Consistency harms local accuracy | Freeze feature grids, lower lambda, retain raw-data reconstruction loss |
| Gradient loss is too expensive | Report value consistency as the main method and gradient loss as future work |
| Seam is weak under one transfer function | Select a transfer function whose opacity changes near the boundary error values and also report objective seam metrics |

## 15. Alternative deltas considered

### Feature/gradient-aware adaptive placement

Modify the target density to prioritize scalar gradients, curvature, or selected isovalues. This is computationally feasible and could improve thin structures, but it requires more design choices and may trade global PSNR for task-specific accuracy. It is a good backup if domain decomposition was not part of the mid-semester implementation.

### Transfer-function-aware training

Prioritize values that contribute strongly to a selected rendering transfer function. This can produce dramatic images but becomes transfer-function-specific, while the original renderer emphasizes arbitrary transfer functions.

### Sparse top-k grid routing

Avoid evaluating every adaptive grid for every point. This attacks the paper's memory bottleneck, but an efficient implementation likely requires custom kernels or careful sparse batching and is too risky for one week.

### 4D temporal APMGSRN

Scientifically interesting but requires suitable time-varying data, much more memory, and a larger evaluation effort.

### Uncertainty-aware APMGSRN

Not recommended as the main delta because the authors' later RMDSRN work and other 2024-2025 papers already study uncertainty-aware feature-grid SRNs in depth.

## 16. Decision

Proceed with **Seamless APMGSRN** unless the existing mid-semester code did not implement domain decomposition at all. If it did not, use **gradient-aware adaptive placement** as the fallback because it touches only the single-model training loop.

