# Seamless APMGSRN: Complete Implementation Runbook

This document is the implementation authority for the delta addition. Follow it in order. Do not skip validation gates. Do not overwrite baseline checkpoints.

Project root:

```text
C:\Users\Devyash\Documents\Computer_Graphics
```

Repository root:

```text
C:\Users\Devyash\Documents\Computer_Graphics\APMGSRN
```

## 1. Mission

Extend the original domain-decomposed APMGSRN with two mechanisms:

1. **Smooth overlap blending:** replace hard model switching at brick boundaries with normalized smooth weights over neighboring models' ghost-cell overlap.
2. **Boundary-consistency fine-tuning:** fine-tune neighboring decoders so their scalar predictions, and optionally their normal derivatives, agree inside shared overlap regions.

Evaluate whether the proposed method can approach the seam quality of an actual 16-cell overlap while using an actual 4-cell overlap.

The final comparison must include:

- global PSNR;
- boundary-band PSNR;
- scalar disagreement at interfaces;
- normal-gradient disagreement at interfaces;
- rendered-image PSNR and SSIM;
- model-query throughput;
- rendered-frame time;
- peak GPU memory;
- training and fine-tuning time.

## 2. Fixed project scope

Do not expand the scope until the mandatory experiment is complete.

### Dataset

- Primary: `Plume.nc`
- Original dimensions reported by the paper: `128 x 128 x 512`
- Decomposition: `2 x 2 x 2`
- Total brick models: 8

### Models to train

1. Actual 1-cell overlap baseline.
2. Actual 4-cell overlap baseline/proposed checkpoint.
3. Actual 16-cell overlap comparison, only after the first two are complete.
4. Actual 4-cell overlap checkpoint after consistency fine-tuning.

### Routing/evaluation modes

- Hard routing on 1-cell checkpoint.
- Hard routing on 4-cell checkpoint.
- Smooth blending on the same 4-cell checkpoint.
- Smooth blending on the fine-tuned 4-cell checkpoint.
- Hard routing on 16-cell checkpoint.
- Optional: smooth blending on 16-cell checkpoint.

### Training constraints

- One random seed: 42.
- Pure PyTorch decoder first. Do not spend project time compiling Tiny-CUDA-NN.
- Target total GPU usage: 110 minutes.
- Absolute GPU ceiling: 120 minutes.
- Development and unit testing must use CPU or tiny synthetic data.

## 3. Required final artifacts

The work is not complete unless all of these exist:

```text
APMGSRN/Code/Models/ensemble_utils.py
APMGSRN/Code/Models/ensemble_SRN.py                    # modified
APMGSRN/Code/Models/options.py                         # modified
APMGSRN/Code/start_jobs.py                             # modified
APMGSRN/Code/finetune_boundaries.py                    # new
APMGSRN/Code/evaluate_seams.py                         # new
APMGSRN/Code/compare_renderings.py                     # new
APMGSRN/Code/BatchRunSettings/train_plume_dd_g1.json   # new
APMGSRN/Code/BatchRunSettings/train_plume_dd_g4.json   # new
APMGSRN/Code/BatchRunSettings/train_plume_dd_g16.json  # new
APMGSRN/Code/BatchRunSettings/test_seamless.json       # new or equivalent
APMGSRN/Tests/test_ensemble_utils.py                    # new
APMGSRN/Tests/test_blending.py                          # new
APMGSRN/Output/SeamlessAPMGSRN/metrics.csv
APMGSRN/Output/SeamlessAPMGSRN/metrics.json
APMGSRN/Output/SeamlessAPMGSRN/timing.json
APMGSRN/Output/SeamlessAPMGSRN/renders/*.png
APMGSRN/Output/SeamlessAPMGSRN/profiles/*.csv
APMGSRN/Output/SeamlessAPMGSRN/plots/*.png
```

Also produce a short final report containing:

- method diagram;
- equations;
- implementation description;
- experiment table;
- boundary line profile;
- error heatmaps;
- zoomed rendering comparison;
- limitations.

## 4. Understand the existing coordinate conventions before coding

This repository uses two different axis orders. Mixing them will silently invalidate the method.

### Data tensor order

NetCDF data is loaded as:

```text
[batch, channel, depth, height, width]
```

`full_shape` is therefore:

```text
[depth, height, width]
```

This runbook calls that **DHW order**.

### Model coordinate order

`make_coord_grid()` constructs coordinates from DHW dimensions and then calls `flip(-1)`. Coordinates passed to neural models are therefore:

```text
[x, y, z] = [width coordinate, height coordinate, depth coordinate]
```

This runbook calls that **XYZ order**.

### Existing misleading names

`start_jobs.py` names `full_shape[0]` as `x`, but it is actually the depth/storage axis. Do not copy these names into new utilities.

Use these conversions everywhere:

```text
DHW axis 0 <-> XYZ axis 2
DHW axis 1 <-> XYZ axis 1
DHW axis 2 <-> XYZ axis 0
```

Implement the conversion once in `ensemble_utils.py`. Never reproduce it ad hoc in multiple files.

## 5. Understand the ghost-cell convention

Existing `start_jobs.py` does this:

```python
ghost_cells = base_opt['ensemble_ghost_cells'] + 1
```

Therefore:

| JSON value | Actual overlap cells used by code |
|---:|---:|
| 0 | 1 |
| 3 | 4 |
| 15 | 16 |

Keep this behavior for backward compatibility. Name checkpoints by **actual** overlap:

```text
Plume_APMGSRN_dd_g1   -> ensemble_ghost_cells = 0
Plume_APMGSRN_dd_g4   -> ensemble_ghost_cells = 3
Plume_APMGSRN_dd_g16  -> ensemble_ghost_cells = 15
```

Every log must include both `configured_extra_ghost_cells` and `actual_ghost_cells`.

## 6. Safety and baseline rules

1. Create a new Git branch before code changes:

```powershell
Set-Location C:\Users\Devyash\Documents\Computer_Graphics\APMGSRN
git switch -c seamless-apmgsrn
```

2. Never change the mathematical behavior of the default hard router.
3. Default `ensemble_routing_mode` must be `hard`.
4. Never overwrite `SavedModels/Plume_APMGSRN_dd_g1`, `_g4`, or `_g16` during fine-tuning.
5. Fine-tuning must create `SavedModels/Plume_APMGSRN_dd_g4_consistency`.
6. New scripts must accept CPU devices for smoke tests.
7. Every new numeric option must be saved in `options.json`.
8. Every stage must write elapsed time and peak GPU memory.

## 7. Environment setup

Use the existing Python 3.10 installation. Create the environment outside the cloned repository or at the project root.

```powershell
Set-Location C:\Users\Devyash\Documents\Computer_Graphics
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with a CUDA wheel supported by the installed driver. Start with the current official CUDA 12.8 wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install the minimum repository dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 netCDF4 h5py matplotlib tensorboard imageio imageio-ffmpeg opencv-python pyquaternion vtk pytest gdown
```

Install renderer dependencies only after model training works:

```powershell
.\.venv\Scripts\python.exe -m pip install nerfacc
```

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory/1024**3)"
```

Expected:

```text
True
NVIDIA GeForce RTX 3050 Laptop GPU
approximately 4 GB
```

Do not install Tiny-CUDA-NN during the mandatory implementation. All new configs must set:

```json
"use_tcnn_if_available": false
```

### PyTorch compatibility check

Run:

```powershell
Set-Location C:\Users\Devyash\Documents\Computer_Graphics\APMGSRN
..\.venv\Scripts\python.exe -m compileall -q Code
```

If current PyTorch rejects `verbose=True` in `ReduceLROnPlateau`, remove only that keyword from `Code/train.py`. Do not alter the scheduler otherwise.

## 8. Dataset setup

First check whether the mid-semester implementation already downloaded `Plume.nc`:

```powershell
Get-ChildItem -Path C:\Users\Devyash\Documents -Filter Plume.nc -Recurse -ErrorAction SilentlyContinue
```

If found, copy it into:

```text
APMGSRN/Data/Plume.nc
```

Otherwise download the official data archive from the repository README:

```powershell
Set-Location C:\Users\Devyash\Documents\Computer_Graphics\APMGSRN
New-Item -ItemType Directory -Force -Path Data | Out-Null
..\.venv\Scripts\gdown.exe --fuzzy "https://drive.google.com/file/d/1a1363lnv154kWcIeg8cvtjnx8uTE9xe2/view?usp=sharing" -O DataArchive.zip
Expand-Archive -LiteralPath .\DataArchive.zip -DestinationPath . -Force
```

Locate the result and ensure the final path is exactly:

```text
APMGSRN/Data/Plume.nc
```

Verify dimensions:

```powershell
..\.venv\Scripts\python.exe -c "from netCDF4 import Dataset; f=Dataset(r'Data/Plume.nc'); print([(k,v.shape) for k,v in f.variables.items()]); f.close()"
```

Do not begin GPU training until this succeeds.

## 9. Phase 0: untouched baseline smoke test

Before implementing the delta, prove that the repository trains one tiny model.

Create a temporary config named `Code/BatchRunSettings/train_smoke.json`:

```json
[
  ["train.py", {
    "save_name": "smoke_plume",
    "model": "APMGSRN",
    "data": "Plume.nc",
    "n_features": 1,
    "n_grids": 4,
    "feature_grid_shape": "8,8,8",
    "n_layers": 2,
    "nodes_per_layer": 32,
    "precondition": false,
    "use_tcnn_if_available": false,
    "log_every": 0,
    "log_features_every": 0,
    "lr": 0.01,
    "points_per_iteration": 4096,
    "iterations": 100
  }]
]
```

Run on CPU first:

```powershell
..\.venv\Scripts\python.exe Code/start_jobs.py --settings train_smoke.json --devices cpu --data_devices cpu
```

Then run on GPU:

```powershell
..\.venv\Scripts\python.exe Code/start_jobs.py --settings train_smoke.json --devices cuda:0 --data_devices cpu
```

Validation gate:

- both runs exit with code 0;
- `SavedModels/smoke_plume/model.ckpt.tar` exists;
- GPU run does not exceed 3.5 GB;
- record GPU seconds per iteration.

If this gate fails, do not implement blending yet. Fix the environment first.

## 10. Phase 1: centralize ensemble geometry

Create:

```text
Code/Models/ensemble_utils.py
```

This module must contain all brick geometry, axis conversion, adjacency, and weighting logic. It must not import the renderer or datasets.

### 10.1 Required functions

Implement these exact public functions:

```python
def parse_int_tuple(text: str) -> tuple[int, ...]:
    """Parse '2,2,2' without eval()."""

def dhw_to_xyz(values: torch.Tensor) -> torch.Tensor:
    """Reverse the final dimension."""

def xyz_to_dhw(values: torch.Tensor) -> torch.Tensor:
    """Reverse the final dimension."""

def flat_brick_index(index_dhw: tuple[int, int, int], grid_dhw: tuple[int, int, int]) -> int:
    """Match the existing Ensemble_SRN flattening order exactly."""

def compute_core_extents_dhw(
    full_shape_dhw: tuple[int, int, int],
    ensemble_grid_dhw: tuple[int, int, int],
    index_dhw: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Return [d0,d1,h0,h1,w0,w1], end-exclusive."""

def expand_extents_dhw(
    core_extents_dhw,
    full_shape_dhw,
    actual_ghost_cells: int,
):
    """Clip expanded support to the volume."""

def voxel_extents_dhw_to_normalized_xyz(extents_dhw, full_shape_dhw) -> torch.Tensor:
    """Return [xmin,ymin,zmin,xmax,ymax,zmax] using align_corners=True."""

def enumerate_face_adjacencies(grid_dhw) -> list[dict]:
    """Return each face-adjacent pair once, including its DHW normal axis."""

def smoothstep01(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0, 1).square() * (3 - 2 * t.clamp(0, 1))

def support_weight(
    points_xyz: torch.Tensor,
    support_min_xyz: torch.Tensor,
    core_min_xyz: torch.Tensor,
    core_max_xyz: torch.Tensor,
    support_max_xyz: torch.Tensor,
) -> torch.Tensor:
    """Return one nonnegative weight per point."""
```

### 10.2 Core extent formula

Match `start_jobs.py` exactly:

```python
step = full_size / brick_count
start = int(brick_index * step)
end = full_size if brick_index == brick_count - 1 else int((brick_index + 1) * step)
```

Extents are end-exclusive.

### 10.3 Normalized extent formula

For an end-exclusive voxel interval `[start, end)` in a dimension of size `N`:

```python
normalized_min = -1 + 2 * start / (N - 1)
normalized_max = -1 + 2 * (end - 1) / (N - 1)
```

Convert DHW endpoints into XYZ order after normalization.

### 10.4 Weight definition

For each model and coordinate axis:

```text
outside support                         -> 0
support_min to core_min                -> smooth ramp 0 to 1
inside core                            -> 1
core_max to support_max                -> smooth ramp 1 to 0
```

Pseudocode:

```python
def axis_weight(q, support_min, core_min, core_max, support_max):
    w = torch.ones_like(q)

    outside = (q < support_min) | (q > support_max)
    w[outside] = 0

    if core_min > support_min:
        left = (q >= support_min) & (q < core_min)
        t = (q[left] - support_min) / (core_min - support_min)
        w[left] = smoothstep01(t)

    if support_max > core_max:
        right = (q > core_max) & (q <= support_max)
        t = (support_max - q[right]) / (support_max - core_max)
        w[right] = smoothstep01(t)

    return w
```

The 3D model weight is the product of the three axis weights.

### 10.5 Unit tests

Create `Tests/test_ensemble_utils.py` with at least:

1. `2 x 2 x 2` produces 8 unique flat indices.
2. It produces exactly 12 face-adjacent pairs.
3. Core extents cover every voxel exactly once.
4. Expanded extents never leave the volume.
5. DHW-to-XYZ round trip is exact.
6. Weight is 1 in a brick core.
7. Weight is 0 outside support.
8. Weight is monotonic across each overlap ramp.
9. Weight has no NaN for volume-edge bricks where support equals core at the outer boundary.

Run:

```powershell
..\.venv\Scripts\python.exe -m pytest -q Tests/test_ensemble_utils.py
```

Validation gate: all tests pass on CPU.

## 11. Phase 2: save explicit brick metadata

Modify `Code/start_jobs.py`.

### 11.1 Replace duplicated geometry

Import the new helpers and use them to compute core and support extents. Preserve the commands generated by the old implementation.

### 11.2 Write a manifest

When an ensemble job is prepared, write:

```text
SavedModels/<ensemble_name>/ensemble_manifest.json
```

Schema:

```json
{
  "full_shape_dhw": [128, 128, 512],
  "ensemble_grid_dhw": [2, 2, 2],
  "configured_extra_ghost_cells": 3,
  "actual_ghost_cells": 4,
  "bricks": [
    {
      "flat_index": 0,
      "index_dhw": [0, 0, 0],
      "core_extents_dhw": [0, 64, 0, 64, 0, 256],
      "support_extents_dhw": [0, 68, 0, 68, 0, 260],
      "subfolder": "0,68,0,68,0,260"
    }
  ]
}
```

Do not assume the example numbers for every brick; compute them.

### 11.3 Backward compatibility

`Ensemble_SRN` must still load older ensembles without a manifest by reconstructing metadata from `options.json` and submodel extents.

Validation gate:

- build commands for an existing ensemble JSON without launching jobs;
- compare every generated support extent against the pre-change version;
- they must match exactly.

## 12. Phase 3: add configuration options

Modify `Code/Models/options.py` and add defaults:

```python
opt['ensemble_routing_mode'] = 'hard'
opt['ensemble_blend_epsilon'] = 1e-8
opt['boundary_consistency_weight'] = 1.0
opt['boundary_gradient_weight'] = 0.0
opt['boundary_finetune_iterations'] = 1500
opt['boundary_points_per_iteration'] = 4096
opt['boundary_finetune_lr'] = 5e-4
opt['boundary_warmup_iterations'] = 250
opt['max_training_minutes'] = 0.0
```

Meanings:

- `hard`: original router.
- `blend`: smooth normalized overlap blending.
- `max_training_minutes = 0`: unlimited, preserving original behavior.

Add matching parser arguments only where required:

- `train.py`: `--max_training_minutes`
- `test.py`: `--ensemble_routing_mode`
- `renderer.py`: `--ensemble_routing_mode`, `--benchmark_frames`
- `finetune_boundaries.py`: all fine-tuning settings

## 13. Phase 4: preserve and isolate the original hard router

Modify `Code/Models/ensemble_SRN.py`.

Refactor the existing `forward` method without changing its results:

```python
def forward(self, x):
    if self.routing_mode == 'hard':
        return self.forward_hard(x)
    if self.routing_mode == 'blend':
        return self.forward_blend(x)
    raise ValueError(...)
```

Move the current body into `forward_hard` with minimal edits.

Add:

```python
def global_to_local(self, x_xyz, model_index):
    return -1 + 2 * (
        (x_xyz - self.local_min_extents[model_index]) /
        (self.local_max_extents[model_index] - self.local_min_extents[model_index])
    )
```

Use this method in both routers and in fine-tuning.

### Regression test

Create a mock ensemble of analytic child models and compare the old hard-forward implementation against `forward_hard` on 10,000 random coordinates. Require exact model selection and numerical equality within `1e-7`.

Do not proceed until hard routing is unchanged.

## 14. Phase 5: implement smooth blending

Add to `Ensemble_SRN.__init__`:

- `core_min_extents` buffer in XYZ normalized coordinates;
- `core_max_extents` buffer;
- existing support min/max buffers remain;
- `routing_mode` string from options.

Add:

```python
def model_support_weights(self, x_xyz):
    """Return [number_of_models, batch] weights."""
```

### 14.1 Correctness-first blending algorithm

Implement the simple eight-model-safe version first:

```python
def forward_blend(self, x):
    numerator = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    denominator = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

    for i, child in enumerate(self.models):
        weights = support_weight(...).unsqueeze(-1)
        mask = weights[:, 0] > 0
        if not mask.any():
            continue

        local_x = self.global_to_local(x[mask], i)
        prediction = child(local_x)
        numerator[mask] += weights[mask] * prediction
        denominator[mask] += weights[mask]

    missing = denominator[:, 0] <= self.blend_epsilon
    result = numerator / denominator.clamp_min(self.blend_epsilon)

    if missing.any():
        result[missing] = self.forward_hard(x[missing])

    return result
```

This loops over all eight models but only evaluates points inside each model's support. That is acceptable for the project. Optimize only after correctness.

### 14.2 Required blending tests

Create `Tests/test_blending.py` using mock child fields.

Test A: identical fields

- Every child returns the same analytic function.
- Hard and blended outputs must match everywhere within `1e-6`.

Test B: constant neighboring fields

- Left child returns 0; right child returns 1.
- Blended output must transition monotonically from 0 to 1 through overlap.
- No discontinuous jump larger than 0.05 between dense adjacent samples.

Test C: normalization

- Sum of active weights must be positive for every coordinate in the global domain.
- Normalized weights must sum to 1 within `1e-6`.

Test D: face, edge, corner

- Confirm maximum active model counts are 2, 4, and 8 respectively.

Test E: autograd

- Blended output must have finite gradients with respect to coordinates away from exact support endpoints.

Validation gate:

```powershell
..\.venv\Scripts\python.exe -m pytest -q Tests
```

All tests pass on CPU.

## 15. Phase 6: add CLI routing overrides

### `Code/test.py`

Add parser argument:

```python
parser.add_argument('--ensemble_routing_mode', default=None, choices=['hard', 'blend'])
```

After loading options but before `load_model`:

```python
if args['ensemble_routing_mode'] is not None:
    opt['ensemble_routing_mode'] = args['ensemble_routing_mode']
```

### `Code/renderer.py`

Add the same routing override. Apply it before `load_model`.

Add:

```python
parser.add_argument('--benchmark_frames', default=1, type=int)
```

Replace the hard-coded `timesteps = 10` with this value. The old ten-frame benchmark wastes GPU time during experiments.

Validation gate:

- load the same ensemble twice, once hard and once blend;
- hard still produces the baseline;
- blend loads without checkpoint duplication.

## 16. Phase 7: implement boundary-consistency fine-tuning

Create:

```text
Code/finetune_boundaries.py
```

### 16.1 Command-line interface

Required arguments:

```text
--load_from Plume_APMGSRN_dd_g4
--save_name Plume_APMGSRN_dd_g4_consistency
--device cuda:0
--data_device cpu
--iterations 1500
--points_per_iteration 4096
--lr 0.0005
--value_weight 1.0
--gradient_weight 0.0
--warmup_iterations 250
--seed 42
--max_minutes 20
```

### 16.2 Clone checkpoint safely

At startup:

1. Refuse to run if source does not exist.
2. Refuse to run if source and destination names are equal.
3. Refuse to overwrite an existing destination unless an explicit `--overwrite false/true` option is provided; default false.
4. Copy the ensemble directory to the destination.
5. Rewrite the parent `options.json` so `save_name` is the destination.
6. Rewrite every child `options.json` so its `save_name` becomes `destination/subfolder`.

This rewrite is mandatory because the existing loader resolves checkpoint paths from each child's saved `save_name`.

### 16.3 Load models and freeze parameters

Load the destination as an ensemble. For every child APMGSRN:

```python
child.encoder.feature_grids.requires_grad_(False)
child.encoder.transformation_matrices.requires_grad_(False)
for parameter in child.decoder.parameters():
    parameter.requires_grad_(True)
```

Build one Adam optimizer containing all decoder parameters.

Why decoder-only:

- minimizes memory;
- preserves learned adaptive-grid positions;
- avoids destabilizing interior reconstruction;
- makes the method a cheap post-training correction.

### 16.4 Enumerate neighboring pairs

For `2 x 2 x 2`, require exactly 12 face-adjacent pairs.

Each pair record must contain:

```text
left/right model indices
normal axis in DHW
normal axis in XYZ
intersection of support extents
decomposition-plane coordinate
```

Do not train diagonal-only neighbors.

### 16.5 Sample overlap points

At every iteration:

1. Select a pair in round-robin order, with the pair list shuffled once per cycle using seed 42.
2. Compute the intersection of their support extents in DHW voxel coordinates.
3. Uniformly sample continuous coordinates from that intersection.
4. Convert coordinates to normalized global XYZ.
5. Convert the same global points to each child's local coordinates with `global_to_local`.
6. Sample ground-truth Plume values at the global points.

Ground-truth sampling must mirror `Dataset.get_random_points`:

```python
grid = points_xyz.reshape(1, 1, 1, batch, 3)
values = F.grid_sample(
    full_data,
    grid,
    mode='bilinear',
    align_corners=True,
).reshape(channels, batch).transpose(0, 1)
```

Keep `full_data` on CPU if GPU memory is limited. Move only the sampled targets to GPU.

### 16.6 Loss function

Let both models predict the same physical points:

```python
y_i = model_i(local_x_i)
y_j = model_j(local_x_j)
```

Use data-range-normalized losses:

```python
scale2 = (data_max - data_min) ** 2 + 1e-12

L_reconstruction = 0.5 * (
    mse(y_i, target) + mse(y_j, target)
) / scale2

L_value = mse(y_i, y_j) / scale2
```

Warm up the consistency weight:

```python
ramp = min(1.0, iteration / warmup_iterations)
lambda_value = value_weight * ramp
```

Minimum required loss:

```python
L_total = L_reconstruction + lambda_value * L_value
```

### 16.7 Optional normal-gradient loss

Do not enable this until value-only fine-tuning is stable.

Use a finite-difference step of one voxel in the pair's normal dimension:

```python
delta = 2.0 / (full_shape_xyz[normal_axis_xyz] - 1)
```

Evaluate each model at `x + delta` and `x - delta`, clamped to the common support. Then:

```python
grad_i = (f_i(x_plus) - f_i(x_minus)) / (2 * delta)
grad_j = (f_j(x_plus) - f_j(x_minus)) / (2 * delta)
L_gradient = mse(grad_i, grad_j) / scale2
```

Use:

```python
L_total += gradient_weight * ramp * L_gradient
```

Start with:

```text
gradient_weight = 0.05
```

Do not use full second-order autograd for the mandatory experiment.

### 16.8 Time and stability guards

Every 50 iterations log:

- elapsed seconds;
- current pair;
- reconstruction loss;
- value loss;
- gradient loss if enabled;
- learning rate;
- GPU allocated and reserved memory.

Stop and save early if:

- elapsed time exceeds `max_minutes`;
- any loss is NaN/Inf;
- GPU memory exceeds 3.7 GB;
- reconstruction loss rises by more than 5x relative to its first 100-iteration median.

### 16.9 Saving

Save every child model's state dictionary back into its destination child folder. Also save:

```text
SavedModels/<destination>/boundary_finetune_log.json
SavedModels/<destination>/boundary_finetune_log.csv
```

The parent options must record all fine-tuning hyperparameters.

## 17. Phase 8: implement seam metrics

Create:

```text
Code/evaluate_seams.py
```

### 17.1 CLI

```text
--load_from <ensemble checkpoint>
--routing_mode hard|blend
--device cuda:0|cpu
--data_device cpu
--boundary_band_cells 2
--face_samples 65536
--global_samples 1000000
--throughput_points 262144
--throughput_repeats 20
--output_dir Output/SeamlessAPMGSRN/<run_name>
--seed 42
```

### 17.2 Global PSNR

Use either the existing exact full-volume PSNR or a clearly labeled random-sample PSNR during development.

Final results must use exact full-volume PSNR if memory/time permits. Chunk the coordinate grid; never allocate the full reconstructed volume on GPU.

Formula:

```text
PSNR = 20 log10(data_range) - 10 log10(MSE)
```

### 17.3 Boundary-band PSNR

For each internal decomposition plane, select voxel centers within `boundary_band_cells` on both sides. Evaluate the ensemble and ground truth only at those coordinates.

Avoid double-counting by building one Boolean mask over the full DHW lattice or by deduplicating flattened voxel indices.

Report:

- combined boundary-band PSNR;
- per-plane PSNR;
- per-axis PSNR.

### 17.4 Scalar disagreement

This metric compares adjacent child models at the same physical points; do not compare points offset to either side because that includes the ground truth's natural spatial variation.

For each adjacent pair:

1. Sample points on the decomposition plane, restricted to the common support.
2. Convert points to both local coordinate systems.
3. Compute:

```text
abs(f_i(x) - f_j(x))
```

Report:

- mean;
- median;
- 95th percentile;
- maximum;
- values normalized by data range.

### 17.5 Normal-gradient disagreement

At the same face points, use central finite differences with one-voxel normalized step along the face normal.

Compute:

```text
abs(d f_i / d n - d f_j / d n)
```

Report the same summary statistics as scalar disagreement.

### 17.6 Throughput

Do not use the repository's fixed `2^23` batch on a 4 GB GPU.

Default:

```text
262,144 points per repeat
5 warm-up repeats
20 measured repeats
```

Measure:

- hard routing points/second;
- blended routing points/second;
- peak allocated memory;
- fraction of points that use 1, 2, 4, and 8 child models.

Synchronize CUDA before and after timing.

### 17.7 Required output schema

Write `metrics.json` with:

```json
{
  "checkpoint": "Plume_APMGSRN_dd_g4",
  "routing_mode": "blend",
  "actual_ghost_cells": 4,
  "global_psnr_db": 0.0,
  "boundary_psnr_db": 0.0,
  "value_jump": {
    "mean": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "gradient_jump": {
    "mean": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "throughput_points_per_second": 0.0,
  "peak_gpu_gb": 0.0,
  "elapsed_seconds": 0.0
}
```

Also write per-pair metrics to CSV.

## 18. Phase 9: rendering evaluation

First try the existing `Code/renderer.py` with `nerfacc`. Use identical parameters for raw data and every model.

Recommended fixed rendering setup:

```text
resolution: 256 x 256
samples per pixel: 128
batch size: 131072
azimuth: 45 degrees
polar: 45 degrees
colormap: Plume.json if present, otherwise Inferno.json
benchmark frames: 1
```

Example commands:

```powershell
..\.venv\Scripts\python.exe Code/renderer.py --raw_data true --load_from Plume.nc --device cuda:0 --data_device cuda:0 --hw 256,256 --spp 128 --batch_size 131072 --azi 45 --polar 45 --colormap Inferno.json --img_name seamless_gt.png --benchmark_frames 1

..\.venv\Scripts\python.exe Code/renderer.py --load_from Plume_APMGSRN_dd_g1 --ensemble_routing_mode hard --device cuda:0 --data_device cuda:0 --hw 256,256 --spp 128 --batch_size 131072 --azi 45 --polar 45 --colormap Inferno.json --img_name g1_hard.png --benchmark_frames 1

..\.venv\Scripts\python.exe Code/renderer.py --load_from Plume_APMGSRN_dd_g4 --ensemble_routing_mode blend --device cuda:0 --data_device cuda:0 --hw 256,256 --spp 128 --batch_size 131072 --azi 45 --polar 45 --colormap Inferno.json --img_name g4_blend.png --benchmark_frames 1
```

Create `Code/compare_renderings.py`:

```text
--reference Output/seamless_gt.png
--images Output/g1_hard.png Output/g4_hard.png Output/g4_blend.png Output/g4_consistency_blend.png Output/g16_hard.png
--output Output/SeamlessAPMGSRN/render_metrics.csv
```

Use repository utilities `PSNR` and `ssim`. Convert images to float `[0,1]`; shape for SSIM is `[1,3,H,W]`.

The script must also create:

- absolute RGB error images;
- a fixed crop around the strongest visible seam;
- a side-by-side montage with labels.

If `nerfacc` is incompatible with the current PyTorch version, do not spend more than 30 minutes debugging it. Implement a simple fixed-step PyTorch ray marcher or use the already working mid-semester renderer. The scientific contribution is the representation, not nerfacc.

## 19. Phase 10: experiment configurations

Create three training JSON files. Start with these conservative settings.

### Actual 1-cell overlap

`Code/BatchRunSettings/train_plume_dd_g1.json`

```json
[
  ["train.py", {
    "save_name": "Plume_APMGSRN_dd_g1",
    "model": "APMGSRN",
    "ensemble": true,
    "ensemble_grid": "2,2,2",
    "ensemble_ghost_cells": 0,
    "data": "Plume.nc",
    "n_features": 1,
    "n_grids": 8,
    "feature_grid_shape": "24,24,24",
    "n_layers": 2,
    "nodes_per_layer": 64,
    "precondition": false,
    "use_tcnn_if_available": false,
    "log_every": 0,
    "log_features_every": 0,
    "lr": 0.01,
    "points_per_iteration": 32768,
    "iterations": 5000
  }]
]
```

### Actual 4-cell overlap

Same settings, but:

```json
"save_name": "Plume_APMGSRN_dd_g4",
"ensemble_ghost_cells": 3
```

### Actual 16-cell overlap

Same settings, but:

```json
"save_name": "Plume_APMGSRN_dd_g16",
"ensemble_ghost_cells": 15
```

If the 500-iteration benchmark shows that `24^3` comfortably fits and time is below budget, the final run may use `32^3`. Never change architecture between overlap conditions.

## 20. Phase 11: GPU benchmark and budget calculation

Before the final training, run one representative brick for 500 iterations using the final architecture.

Record:

```text
seconds_per_iteration
peak_gpu_gb
total brick time
projected eight-brick time
```

Calculate:

```text
projected_condition_minutes = seconds_per_iteration * final_iterations * 8 / 60
```

Use this decision table:

| Projected time for one 8-brick condition | Action |
|---:|---|
| <= 25 min | Keep 5,000 iterations |
| 25-35 min | Reduce to 4,000 iterations |
| 35-45 min | Reduce to 3,000 iterations |
| > 45 min | Reduce grid to `16^3` or batch to 16,384 and re-benchmark |

Mandatory budget allocation:

```text
g1 training                      25 minutes
g4 training                      25 minutes
g16 training                     20 minutes maximum
g4 consistency fine-tuning       15 minutes
metrics and renders              15 minutes
emergency buffer                 10 minutes
total                           110 minutes
```

If g1 + g4 consume more than 60 minutes, shorten or omit full g16 training. A shortened g16 comparison is acceptable if clearly labeled. Never sacrifice the proposed g4 blend and g4 consistency results.

## 21. Phase 12: execute the final experiment

All training uses the single GPU and data in CPU RAM:

```powershell
Set-Location C:\Users\Devyash\Documents\Computer_Graphics\APMGSRN
```

### 21.1 Train actual 1-cell baseline

```powershell
Measure-Command { ..\.venv\Scripts\python.exe Code/start_jobs.py --settings train_plume_dd_g1.json --devices cuda:0 --data_devices cpu }
```

Immediately evaluate hard routing.

### 21.2 Train actual 4-cell checkpoint

```powershell
Measure-Command { ..\.venv\Scripts\python.exe Code/start_jobs.py --settings train_plume_dd_g4.json --devices cuda:0 --data_devices cpu }
```

Evaluate both hard and blend routing before fine-tuning. This isolates the contribution of blending.

### 21.3 Fine-tune 4-cell checkpoint

```powershell
..\.venv\Scripts\python.exe Code/finetune_boundaries.py --load_from Plume_APMGSRN_dd_g4 --save_name Plume_APMGSRN_dd_g4_consistency --device cuda:0 --data_device cpu --iterations 1500 --points_per_iteration 4096 --lr 0.0005 --value_weight 1.0 --gradient_weight 0.0 --warmup_iterations 250 --seed 42 --max_minutes 15
```

Evaluate the fine-tuned checkpoint with blend routing.

### 21.4 Train 16-cell comparison

Only proceed if at least 20 GPU minutes remain.

```powershell
Measure-Command { ..\.venv\Scripts\python.exe Code/start_jobs.py --settings train_plume_dd_g16.json --devices cuda:0 --data_devices cpu }
```

Evaluate hard routing.

## 22. Exact evaluation matrix

Use these stable run identifiers:

| Run ID | Checkpoint | Routing | Fine-tuned |
|---|---|---|---|
| `g1_hard` | `Plume_APMGSRN_dd_g1` | hard | no |
| `g4_hard` | `Plume_APMGSRN_dd_g4` | hard | no |
| `g4_blend` | `Plume_APMGSRN_dd_g4` | blend | no |
| `g4_consistency_hard` | `Plume_APMGSRN_dd_g4_consistency` | hard | yes |
| `g4_consistency_blend` | `Plume_APMGSRN_dd_g4_consistency` | blend | yes |
| `g16_hard` | `Plume_APMGSRN_dd_g16` | hard | no |

The most important comparisons are:

1. `g4_hard` vs `g4_blend`: effect of blending alone.
2. `g4_blend` vs `g4_consistency_blend`: effect of fine-tuning.
3. `g1_hard` vs `g4_consistency_blend`: complete proposed method versus original minimal-overlap baseline.
4. `g16_hard` vs `g4_consistency_blend`: whether 4 cells approach 16-cell seam quality.

## 23. Acceptance criteria

The delta succeeds if all mandatory engineering criteria and at least three scientific criteria pass.

### Mandatory engineering criteria

- Default hard routing is numerically unchanged.
- No baseline checkpoint is overwritten.
- All unit tests pass.
- Blend weights are finite and normalized.
- CPU smoke tests work.
- All experiment settings and elapsed times are saved.
- Total GPU usage does not exceed 120 minutes.

### Scientific criteria

Target thresholds relative to `g1_hard`:

1. Mean scalar disagreement decreases by at least 50%.
2. 95th-percentile scalar disagreement decreases by at least 40%.
3. Mean gradient disagreement decreases by at least 25%.
4. Boundary-band PSNR improves by at least 1 dB.
5. Global PSNR drops by no more than 0.25 dB.
6. Rendered SSIM increases.
7. The seam is visibly reduced in a fixed zoomed crop.
8. Blended inference slowdown is below 2x for uniformly random volume queries.

These are targets, not numbers to fabricate. Report actual results even if a target is missed.

## 24. Ablation rules

Do not conduct a large hyperparameter search. Use this small sequence:

### Blending

- smoothstep window: mandatory;
- linear window: one optional comparison only;
- no learned gating.

### Value consistency weight

Try at most:

```text
0.25, 1.0, 4.0
```

Use a short 300-iteration pilot for each. Select the largest weight that does not noticeably worsen reconstruction loss.

### Gradient consistency

Only after the main value-consistency result:

```text
gradient_weight = 0.05
500 iterations maximum
```

If it exceeds memory/time or harms PSNR, record it as a negative ablation and stop.

## 25. Output organization

Every evaluation run writes to:

```text
Output/SeamlessAPMGSRN/<run_id>/
```

Required contents:

```text
metrics.json
pair_metrics.csv
timing.json
value_jump_histogram.png
gradient_jump_histogram.png
boundary_error_slice.png
line_profile.csv
line_profile.png
render.png
render_error.png
render_crop.png
```

At the experiment root, aggregate:

```text
Output/SeamlessAPMGSRN/metrics.csv
Output/SeamlessAPMGSRN/render_metrics.csv
Output/SeamlessAPMGSRN/summary_table.md
```

## 26. Required plots

1. **Boundary line profile**
   - x-axis: coordinate crossing one selected brick boundary;
   - curves: ground truth, g1 hard, g4 hard, g4 blend, g4 consistency blend, g16 hard;
   - mark the decomposition plane.

2. **Scalar-jump box plot**
   - one box per run;
   - use all 12 interfaces.

3. **Gradient-jump box plot**
   - same organization.

4. **Boundary-band versus global PSNR chart**
   - grouped bars;
   - demonstrates why global PSNR alone is insufficient.

5. **Performance chart**
   - throughput and peak memory.

6. **Rendering montage**
   - ground truth plus all key runs;
   - identical camera, transfer function, and crop.

## 27. Suggested commit sequence

Keep commits small enough to debug:

```text
1. test: add ensemble geometry and coordinate tests
2. refactor: centralize ensemble brick geometry
3. feat: add explicit ensemble manifests
4. refactor: isolate unchanged hard router
5. feat: add smooth overlap blending
6. test: add blend normalization and continuity tests
7. feat: add boundary consistency fine-tuning
8. feat: add seam-specific evaluation metrics
9. feat: add routing overrides and render comparison
10. experiment: add Plume delta configurations
11. docs: add reproduction commands and results
```

Run tests before every commit.

## 28. Common failure modes

### Model output is mirrored or interfaces are on wrong axes

Cause: DHW/XYZ confusion.

Fix:

- inspect `make_coord_grid()`;
- verify a synthetic volume whose value is `x + 10y + 100z`;
- confirm every interface plane is located on the expected axis.

### Blend produces black/zero gaps

Cause: no active support or incorrect normalized extents.

Fix:

- log minimum denominator;
- assert every in-domain point has denominator greater than zero;
- use hard-routing fallback only as numerical insurance, not as a hidden normal path.

### Blend is not smooth

Cause: weights were computed from integer brick ownership rather than support/core extents.

Fix: plot the two neighbor weights across a 1D boundary before querying networks.

### Fine-tuned model loads baseline paths

Cause: copied child `options.json` files still contain the source `save_name`.

Fix: rewrite every child path during checkpoint cloning.

### Fine-tuning destroys global PSNR

Fix in order:

1. lower value weight;
2. reduce learning rate to `1e-4`;
3. shorten fine-tuning;
4. increase reconstruction term weight;
5. verify only decoder parameters require gradients.

### GPU out of memory

Fix in order:

1. reduce fine-tune batch 4096 -> 2048 -> 1024;
2. reduce training points 32768 -> 16384;
3. keep data on CPU;
4. reduce feature grids `24^3 -> 16^3`;
5. reduce number of grids 8 -> 4 only if all experiment conditions are changed identically.

### Renderer dependency failure

Do not block the representation work. Use a simple fixed-step ray marcher or the mid-semester renderer. Preserve identical rendering parameters across methods.

### Training exceeds time budget

Do not start additional conditions blindly. Reduce iterations based on the benchmark formula. Preserve g1, g4, and g4-consistency; shorten g16 first.

## 29. Minimum viable result if time becomes critical

If the week or GPU budget is nearly exhausted, deliver this reduced but valid study:

1. Actual 1-cell hard baseline.
2. Actual 4-cell checkpoint.
3. Hard and blend routing on the same 4-cell checkpoint.
4. Value-consistency fine-tuning on the 4-cell checkpoint.
5. Scalar jump, gradient jump, boundary PSNR, global PSNR.
6. One fixed rendered comparison.
7. State 16-cell comparison as paper-reported reference if no local run is possible, clearly distinguishing reported and locally measured results.

Never pretend a paper-reported number is a locally reproduced number.

## 30. Definition of done

The implementation is done only when a junior engineer can run the following sequence from a clean checkout and obtain the complete comparison:

```text
1. create environment
2. place Plume.nc in Data/
3. run CPU tests
4. train g1
5. train g4
6. evaluate g4 hard and blend
7. fine-tune g4 consistency checkpoint
8. evaluate g4 consistency blend
9. train/evaluate g16 if budget allows
10. render all methods with identical settings
11. aggregate metrics and plots
```

The research claim must be phrased conservatively:

> Smooth overlap blending and lightweight decoder consistency fine-tuning reduce discontinuities in domain-decomposed APMGSRNs, allowing a narrow-overlap ensemble to approach the visual continuity of substantially larger ghost-cell overlap while retaining global reconstruction accuracy.

Do not claim guaranteed continuity everywhere, a formal C1 guarantee, or superiority on datasets that were not tested.

