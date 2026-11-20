# APMGSRN Project Study

## 1. What was acquired

- Paper: `paper/APMGSRN_arXiv_2308.02494.pdf`
- Extracted paper text: `paper/APMGSRN_arXiv_2308.02494.txt`
- Repository: `APMGSRN/`
- Repository revision: `ab83249d0e276cabac602a76eddd4be6623a9498` on `main`
- Official repository paper: `APMGSRN/VIS2023__APMGSRN.pdf`

The local repository is a shallow clone of the current `main` revision. All tracked source files and assets are present. The repository is clean after cloning.

Paper metadata:

- Title: *Adaptively Placed Multi-Grid Scene Representation Networks for Large-Scale Data Visualization*
- Authors: Skylar W. Wurster, Tianyu Xiong, Han-Wei Shen, Hanqi Guo, Tom Peterka
- Venue: IEEE Transactions on Visualization and Computer Graphics, 30(1), 965-974, 2024; presented at IEEE VIS 2023
- DOI: [10.1109/TVCG.2023.3327194](https://doi.org/10.1109/TVCG.2023.3327194)
- arXiv: [2308.02494](https://arxiv.org/abs/2308.02494)
- Code: [skywolf829/APMGSRN](https://github.com/skywolf829/APMGSRN)

## 2. The problem in one paragraph

An implicit neural representation (INR), also called a scene representation network (SRN), stores a scalar volume in the parameters of a coordinate-to-value function. Given a normalized 3D coordinate, the model predicts the scalar value at that location. Existing grid-based SRNs allocate parameters uniformly even when the scientific volume has highly non-uniform spatial complexity. APMGSRN replaces one uniform feature grid with several learnable feature grids. Each grid can translate, scale, rotate, and shear so that feature capacity migrates toward difficult regions. For data too large for a GPU, the paper independently trains APMGSRNs on spatial bricks and uses a coordinate hash to route queries to the correct model.

## 3. Main contributions of the paper

1. **Adaptively placed multi-grid encoder**
   - Splits a feature budget across several small dense feature grids.
   - Learns an affine transformation for every grid.
   - Concentrates feature density in high-reconstruction-error regions.

2. **A differentiable feature-density objective**
   - A hard inside/outside box density has zero gradient almost everywhere.
   - The paper approximates every grid with a high-order flat-top Gaussian.
   - A KL-divergence loss moves grid transformations toward a residual-derived target density.

3. **Large-volume domain decomposition**
   - Splits the volume into a regular 3D brick grid.
   - Trains one model per brick, in parallel when multiple GPUs are available.
   - Routes inference queries by hashing their coordinates to a brick/model index.

4. **A plug-and-play neural volume renderer**
   - Uses PyTorch models directly.
   - Supports arbitrary camera views, colormaps, and opacity transfer functions.
   - Includes both command-line and Qt/pyqtgraph GUI renderers.

## 4. Model architecture

For a coordinate `x` in normalized global space `[-1, 1]^3`, every feature grid has a learned homogeneous transformation matrix `G_i`:

```text
global coordinate x
        |
        +--> G_0(x) --> trilinear sample grid 0 --+
        +--> G_1(x) --> trilinear sample grid 1 --+
        |                                           +--> concatenate --> shallow MLP --> scalar value
        +--> G_M(x) --> trilinear sample grid M --+
```

The implementation is in `APMGSRN/Code/Models/APMGSRN.py`.

### 4.1 Encoder

- Feature tensor shape: `[number_of_grids, features_per_grid, depth, height, width]`.
- Transformation tensor shape: `[number_of_grids, 4, 4]` for a 3D volume.
- `transform(x)` expands a coordinate batch into every grid's local frame.
- `torch.nn.functional.grid_sample` performs trilinear interpolation.
- Coordinates outside a grid return a zero feature vector.
- Features from all grids are concatenated.

Important implementation detail: transformed coordinates are detached before `grid_sample` during training. Therefore, reconstruction loss trains the grid features and decoder, but does not directly train the affine transformations. Grid transformations are trained by the separate feature-density loss.

### 4.2 Decoder

The decoder is a shallow MLP, normally two layers of 64 nodes. The project prefers Tiny-CUDA-NN's fully fused MLP when available and falls back to standard PyTorch layers. The final normalized prediction is rescaled to the original data range.

### 4.3 Feature density

The exact support of a transformed box is not differentiable with respect to its transformation. The code instead uses a high-order generalized Gaussian:

```text
rho_i(x) proportional to det(G_i) * exp(-0.5 * sum(G_i(x)^20))
rho(x) = sum_i rho_i(x)
```

The exponent 20 creates a differentiable function with a nearly flat top and steep sides. The determinant accounts for the fact that smaller global-space grids have denser features.

### 4.4 Residual-driven target density

For each training batch, the code computes pointwise reconstruction error. It then constructs a target density that increases relative density at high-error points and decreases it at low-error points. KL divergence between current and target densities provides gradients for the transformation matrices.

The result is an unusual two-optimizer system:

- Optimizer 1: feature grids plus decoder, learning rate `lr`.
- Optimizer 2: affine transformation matrices, learning rate `0.05 * lr`.

## 5. Training schedule

The core routine is `train_step_APMGSRN` in `APMGSRN/Code/train.py`.

1. Sample random continuous coordinates uniformly from the volume.
2. Interpolate ground-truth values with `grid_sample`.
3. Predict values and optimize mean squared reconstruction error.
4. After iteration 500, also optimize the transformation matrices using density KL loss.
5. Stop moving grids at 80% of total training at the latest.
6. The code can stop grid motion earlier if the density loss plateaus.
7. After grid motion stops, the representation fine-tunes at fixed grid locations.

The reference `train.json` uses:

- Plume dataset
- 16 grids
- 2 features per grid
- each feature grid `32 x 32 x 32`
- 2 decoder layers, 64 nodes per layer
- 10,000 iterations
- 100,000 random points per iteration
- Adam learning rate 0.01

The paper reports approximately one minute for this small run on an RTX 2080 Ti, with machine-dependent variation.

## 6. Domain decomposition

The decomposition path is implemented across:

- `APMGSRN/Code/start_jobs.py`
- `APMGSRN/Code/Models/ensemble_SRN.py`
- ensemble JSON files under `APMGSRN/Code/BatchRunSettings/`

### 6.1 Training

`start_jobs.py` divides the full volume into a regular `X x Y x Z` grid. Every brick is expanded by a configurable number of ghost cells. It launches a normal `train.py` process for each brick and schedules processes across the available devices.

The brick models are otherwise independent: there is no shared loss, parameter exchange, or interface constraint.

### 6.2 Inference

`Ensemble_SRN.forward`:

1. Maps normalized coordinates to integer brick indices.
2. Converts the 3D brick index into a flat model index.
3. Selects exactly one model per coordinate.
4. Remaps global coordinates to that model's local normalized coordinates.
5. Evaluates the selected model.

This is a hard router. Even if neighboring models overlap because of ghost cells, the current implementation does not blend them.

### 6.3 Consequence

Independently trained models need not predict the same value or gradient at a shared boundary. The paper shows a visible rendering seam with one ghost cell. Increasing overlap to 16 ghost cells removes the visual seam in its example, but decreases global PSNR by about 1 dB because more capacity is spent redundantly modeling overlap.

This is the most actionable weakness in the original system.

## 7. Renderer

The renderer is in `APMGSRN/Code/renderer.py`.

- Generates camera rays.
- Uses `nerfacc` to sample along rays.
- Queries either raw NetCDF data or an SRN.
- Applies an editable color/opacity transfer function.
- Composites samples into pixels.
- Provides an offline renderer and an interactive GUI.

The renderer deliberately favors general PyTorch compatibility over peak performance. The paper notes that it is slower than specialized contemporary neural volume renderers.

## 8. Repository map

| Path | Role |
|---|---|
| `Code/Models/APMGSRN.py` | Adaptive encoder, density function, decoder, forward path |
| `Code/train.py` | Two-optimizer training loop and grid movement schedule |
| `Code/Datasets/datasets.py` | NetCDF loading and random continuous sampling |
| `Code/Models/ensemble_SRN.py` | Hard-routed domain-decomposition inference |
| `Code/start_jobs.py` | Multi-device job scheduling and brick generation |
| `Code/test.py` | PSNR, reconstruction, error volume, density, throughput |
| `Code/renderer.py` | Offline neural volume rendering |
| `Code/UI/renderer_app.py` | Interactive renderer GUI |
| `Code/BatchRunSettings/` | All paper-style experiment configurations |
| `Figures/` | Architecture diagrams, qualitative results, videos |

## 9. What the experiments establish

- APMGSRN generally improves reconstruction PSNR over fVSRN and NGP at comparable model sizes.
- Its benefit is strongest when the volume has spatially localized complexity.
- Initialization matters because an oversized grid has weak transformation gradients while an undersized grid may become trapped in a local optimum.
- Domain decomposition can improve reconstruction at the same aggregate parameter count and allows very large volumes to be trained in-core as independent bricks.
- The paper reports fitting a 450 GB volume in under seven minutes using eight GPUs.
- For one large-data rendering comparison, the domain-decomposition model achieved 30.70 dB image PSNR versus 22.24 dB for a similarly sized downsampled representation.
- APMGSRN remains a lossy representation and is not as compression-efficient as offline compressors such as TTHRESH, but it can render directly from the compressed neural representation.

## 10. Limitations stated by the paper

1. Memory and compute scale with `batch_size * number_of_grids` because every query is transformed and sampled against every grid.
2. The general PyTorch renderer is slower than specialized renderers.
3. Domain-decomposition boundary artifacts are not solved beyond ghost-cell overlap.
4. The method uses dense trilinear grids inside every adaptive region; other encodings could be explored.
5. The paper does not implement 4D temporal grids.
6. Non-uniform grids such as AMR and unstructured data are suggested but not implemented.

## 11. Code and reproducibility observations

### Positive

- Source is compact and readable.
- The architecture maps closely to the paper.
- The repository provides the exact paper configurations.
- All Python files pass syntax compilation under Python 3.10.
- All 20 batch JSON files parse successfully.
- The cloned repository is clean.

### Risks and setup gaps

- Data and pretrained models are external downloads, not tracked in Git.
- The dataset download is about 360 MB compressed / 664 MB decompressed.
- The model archive is about 2.98 GB compressed / 3.23 GB decompressed.
- PyTorch is not pinned in `environment.yml`; the README describes PyTorch 1.12/1.13 and CUDA 11.4/11.7-era setups.
- Tiny-CUDA-NN and nerfacc are source-built CUDA dependencies and may need adaptation for a modern PyTorch/CUDA environment.
- No Conda installation or project Python environment currently exists in this workspace.
- Neither the system Python nor the bundled workspace Python currently has PyTorch installed.
- There are no unit tests.
- Several options use `eval` to parse numeric strings.
- The default model name in `Options.get_default()` is the older string `AMRSRN`; experiment configs explicitly override it.
- Ensemble inference loops through all models in Python even though each coordinate belongs to only one model.
- The throughput test uses a very large fixed batch (`2^23`) and will not fit a 4 GB laptop GPU without adjustment.

## 12. Local hardware assessment

Detected GPU:

- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB VRAM

This is adequate for:

- the paper's small Plume configuration with reduced batch size if necessary;
- a `2 x 2 x 2` decomposition with small per-brick networks trained sequentially;
- low-resolution rendering and seam analysis;
- a focused ablation study with one main dataset and one synthetic backup.

It is not adequate for reproducing the paper's multi-GPU large-volume experiments.

## 13. Related work that affects the project choice

- The authors later published *Regularized Multi-Decoder Ensemble for an Error-Aware Scene Representation Network* ([arXiv:2407.19082](https://arxiv.org/abs/2407.19082), TVCG 2025). It already extends APMGSRN-style feature-grid models with uncertainty/error awareness. Therefore, an uncertainty-head or Monte-Carlo-dropout delta would be too close to established follow-up work.
- *Uncertainty-Informed Volume Visualization using Implicit Neural Representation* ([arXiv:2408.06018](https://arxiv.org/abs/2408.06018)) also studies deep ensembles and MC dropout for volume INRs.
- Recent local-neural-field work supports the general idea of fusing overlapping local implicit models. For example, *Scalable and High-Quality Neural Implicit Representation for 3D Reconstruction* ([arXiv:2501.08577](https://arxiv.org/abs/2501.08577)) uses overlapping local SDFs and blending, although its task and representation differ from scientific scalar volumes.

The clearest useful gap remains continuity across APMGSRN domain-decomposition boundaries.

