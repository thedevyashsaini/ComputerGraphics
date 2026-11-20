# Seamless APMGSRN: continuity-aware blending for domain-decomposed APMGSRN

Delta on top of *Adaptively Placed Multi-Grid Scene Representation Networks for Large-Scale
Data Visualization* (Wurster et al., IEEE VIS 2023). Branch: `seamless-apmgsrn`.

## Problem

For large volumes APMGSRN trains one small network per brick of a `n x n x n` decomposition.
Each brick is trained on its region grown by a few *ghost cells*, but at inference a point is
answered by exactly one brick (`floor((x+1)/2 * n)`). Neighbouring networks never agree exactly,
so the field jumps at brick faces and renderings show seams. The paper reduces this by growing
the overlap to 16 cells, which costs training data, memory, and time in every brick.

## What this delta adds

1. **Partition-of-unity overlap blending** (`ensemble_routing_mode = blend`).
   For brick *i* with support `[s_min, s_max]` (training extents) and blend core `[c_min, c_max]`
   the per-axis weight is

       w_i(q) = S((q - s_min)/(c_min - s_min)) * S((s_max - q)/(s_max - c_max)),   S(t) = 3t^2 - 2t^3

   and the 3-D weight is the product over axes. The blend core is shrunk so both neighbours
   ramp across the *same* `2g-1` voxel overlap in opposite directions, hence
   `S(t) + S(1-t) = 1`: the weights are an exact partition of unity (C1, centred on the
   original hard split). Output: `F(x) = sum_i w_i f_i(x) / sum_i w_i`. Points in a brick
   interior touch one network; face/edge/corner overlaps touch 2/4/8.
2. **Interface-consistency fine-tuning** (`Code/finetune_boundaries.py`).
   Copies a checkpoint, freezes every adaptive grid and transform, and fine-tunes only the
   MLP decoders (about 4% of parameters) with
   `L = L_rec + ramp * (lambda_v * |f_i - f_j|^2 + lambda_g * |d_n f_i - d_n f_j|^2) / range^2`
   on points sampled inside each face-adjacent overlap, plus interior anchor points per brick.
3. **Seam evaluation** (`Code/evaluate_seams.py`, `Code/compare_renderings.py`,
   `Code/summarize_seamless.py`): exact global PSNR, boundary-band PSNR, child-pair value and
   normal-derivative disagreement, composite value/gradient jumps of the routed field (with the
   ground truth's own gradient change as reference), throughput, active-brick fractions,
   peak memory, fixed-camera volume renderings with PSNR/SSIM, error maps, line profiles.

The original hard router is unchanged and remains the default (verified bit-for-bit against
the original code in `Tests/test_blending.py`).

## Ghost-cell convention

`start_jobs.py` uses `ensemble_ghost_cells + 1` actual overlap cells. Config value 0/3/15 =
actual overlap 1/4/16. Checkpoints are named by the actual overlap (`_g1`, `_g4`, `_g16`).

## Files

| File | Change |
|---|---|
| `Code/Models/ensemble_utils.py` | new: all decomposition geometry, DHW/XYZ conversions, adjacency, weights |
| `Code/Models/ensemble_SRN.py` | `forward_hard` (original), `forward_blend`, `forward_child`, `global_to_local` |
| `Code/Models/options.py` | new options, defaults keep original behaviour |
| `Code/start_jobs.py` | shared geometry, writes `ensemble_manifest.json`, Windows interpreter path |
| `Code/train.py` | `--max_training_minutes`, records training seconds/peak memory, PyTorch 2.x fix |
| `Code/test.py`, `Code/renderer.py` | `--ensemble_routing_mode`; renderer `--benchmark_frames` |
| `Code/finetune_boundaries.py` | new |
| `Code/evaluate_seams.py`, `Code/compare_renderings.py`, `Code/summarize_seamless.py` | new |
| `Code/BatchRunSettings/train_plume_dd_g{1,4,16}.json` | Plume, 2x2x2, identical architecture |
| `Tests/` | 18 CPU unit tests |
| `run_seamless_pipeline.ps1` | fine-tune + evaluate + render + summarize |

## Reproduce

```powershell
cd APMGSRN
$py = "D:\Computer_Graphics_env\.venv\Scripts\python.exe"
& $py -m pytest -q Tests
foreach ($g in 'g1','g4','g16') { & $py Code/start_jobs.py --settings "train_plume_dd_$g.json" --devices cuda:0 --data_devices same }
./run_seamless_pipeline.ps1
```

Results land in `Output/SeamlessAPMGSRN/` (`metrics.csv`, `metrics.md`, `render_metrics.csv`,
`renders/montage.png`, `plots/`, `runs/<run_id>/`).

Note: `--data_devices same` is required on this laptop; `cpu` makes the original DataLoader
spawn 4 worker processes per job that exhaust the Windows page file.
