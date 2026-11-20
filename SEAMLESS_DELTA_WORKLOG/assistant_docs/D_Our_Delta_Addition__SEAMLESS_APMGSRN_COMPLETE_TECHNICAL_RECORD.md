# Seamless APMGSRN: complete technical record of our extension

## 0. Purpose and scope of this document

This document records the complete project extension implemented on top of Wurster et al.'s APMGSRN repository. It is written as a standalone knowledge base for technical questioning. It covers the motivation, exact algorithm, geometry conventions, formulas, implementation files, configuration variables, dataset, experiments, controls, metrics, results, costs, limitations, and reproducibility.

The extension is called **Seamless APMGSRN**. The repository branch is `seamless-apmgsrn`.

## 1. Original behavior that created the problem

The original domain-decomposition mode divides a volume into \(I\times J\times K\) bricks and independently trains one APMGSRN per brick. Each brick's training region is enlarged by ghost cells, so neighbors learn some shared voxels.

At inference, the original code routes every coordinate to exactly one brick:

\[
i=\left\lfloor I\frac{x+1}{2}\right\rfloor,\qquad
j=\left\lfloor J\frac{y+1}{2}\right\rfloor,\qquad
k=\left\lfloor K\frac{z+1}{2}\right\rfloor.
\]

The selected model index is flattened with width fastest and depth slowest. In the repository's DHW indexing convention:

\[
\operatorname{flat}(d,h,w)=w+hK+dKJ.
\]

Although neighboring models share training voxels, their learned functions are independent. At a split, the routed field changes abruptly from \(f_i\) to \(f_j\). The predictions do not agree exactly, so a visible seam can appear.

The paper mitigates this by increasing overlap up to sixteen ghost cells. A wider overlap consumes fixed model capacity on redundant data and can lower PSNR. The project goal was to obtain equal or better interface quality using only two to four ghost cells.

## 2. Final contribution and honest interpretation

The implemented extension has three parts.

1. **Partition-of-unity blending:** smoothly combine every brick whose ghost-cell support contains the query. This is the main seam-removal mechanism and works on existing checkpoints with no training.
2. **Short decoder-only fine-tuning:** freeze feature grids and transformations, then update only decoders. We tested an explicit interface-consistency objective and a reconstruction-only control.
3. **Seam-aware evaluation:** measure boundary accuracy, routed discontinuities, gradients, rendering, speed, memory, seeds, and parameter capacity.

The final evidence shows that blending is the primary contribution. A short decoder fine-tune adds accuracy, but the reconstruction-only control performs slightly better than the designed consistency loss. The consistency loss is therefore reported as an ablation, not as the proven source of the gain.

## 3. Dataset and experimental scope

### 3.1 Dataset

- Name: **Plume**.
- File: `APMGSRN/Data/Plume.nc`.
- Shape in repository storage order: \(D\times H\times W=512\times128\times128\).
- Voxels: 8,388,608.
- Raw storage at float32: 32 MiB.
- Quantity: velocity magnitude of particles in a simulated solar plume.
- Reason for selection: it is small enough for the RTX 3050, contains localized complex structures, appears in the paper, and exposes the ghost-cell trade-off.

### 3.2 Decomposition

- Grid: \(2\times2\times2\).
- Number of child models: 8.
- Face-adjacent pairs: 12.
- Face overlaps can activate 2 models.
- Edge overlaps can activate 4 models.
- Corner overlaps can activate 8 models.

### 3.3 Equal-size architecture used in the main comparison

Each child model uses:

- `model = APMGSRN`
- `n_grids = 8`
- `n_features = 1`
- `feature_grid_shape = 24,24,24`
- `n_layers = 2`
- `nodes_per_layer = 64`
- `use_tcnn_if_available = false`

Across all eight bricks, every 24-cubed condition contains exactly **923,136 parameters**, approximately **3.52 MiB** in float32.

### 3.4 Base training schedule used in this project

- 5,000 iterations per brick.
- 32,768 randomly sampled points per iteration.
- Learning rate 0.01.
- Maximum six minutes per brick in the study configs.
- Seeds 42, 1, and 2 for the key g4 and g16 equal-size conditions.
- Training used one RTX 3050 laptop GPU with 4 GB VRAM.

This is a resource-limited reproduction and extension, not the paper's 50,000-iteration A100 setup.

## 4. Ghost-cell naming convention

The original job generator adds one mandatory alignment cell internally:

\[
g_{\text{actual}}=\texttt{ensemble\_ghost\_cells}+1.
\]

Therefore:

| Name used in the study | Config value | Actual overlap cells |
|---|---:|---:|
| g1 | 0 | 1 |
| g2 | 1 | 2 |
| g4 | 3 | 4 |
| g16 | 15 | 16 |

All tables and plots use actual overlap width in their names.

## 5. Coordinate and extent conventions

The repository uses two axis orders.

- **DHW:** volume storage and configuration order. Tensors load as `[B,C,D,H,W]`. `full_shape`, `ensemble_grid`, brick indices, and voxel extents use DHW.
- **XYZ:** neural-network coordinate order. Coordinates passed to models are `[x,y,z]=[W,H,D]` in \([-1,1]^3\).

All conversions were centralized in `Code/Models/ensemble_utils.py` to avoid silent axis errors.

### 5.1 Core brick extents

For an axis of length \(N\), decomposed into \(B\) bricks, brick \(b\) receives the end-exclusive core interval:

\[
s_b=\operatorname{int}\left(b\frac{N}{B}\right),
\]

\[
e_b=
\begin{cases}
N,&b=B-1\\
\operatorname{int}\left((b+1)\frac{N}{B}\right),&\text{otherwise.}
\end{cases}
\]

The use of Python `int` deliberately matches the original `start_jobs.py` behavior.

### 5.2 Support extents

For actual ghost width \(g\), the training support becomes:

\[
s_b^{\text{support}}=\max(0,s_b-g),\qquad
e_b^{\text{support}}=\min(N,e_b+g).
\]

### 5.3 Voxel-to-normalized mapping

Continuous voxel coordinate \(v\) maps to the model domain using align-corners geometry:

\[
q(v,N)=-1+\frac{2v}{N-1}.
\]

The inverse is:

\[
v(q,N)=\frac{q+1}{2}(N-1).
\]

An end-exclusive voxel extent `[s,e)` maps to normalized minimum \(q(s,N)\) and maximum \(q(e-1,N)\).

### 5.4 Original hard split

For split number \(k\) in a \(B\)-brick axis, the original router changes model at:

\[
q_{\text{split}}=-1+\frac{2k}{B}.
\]

The blend is centered on this original split so the spatial meaning of the decomposition does not move.

## 6. Partition-of-unity blending

### 6.1 Blend core

The ordinary brick core is not used directly as the weight-one interval. Neighboring weights must ramp across the same complete overlap in opposite directions.

For core `[s,e)`, volume length \(N\), and actual ghost width \(g\), the inclusive weight-one interval is:

\[
c_{\min}=
\begin{cases}
0,&s=0\\
\min(s+g-1,N-1),&s>0,
\end{cases}
\]

\[
c_{\max}=
\begin{cases}
N-1,&e\geq N\\
\max(e-g,0),&e<N.
\end{cases}
\]

This makes the two neighboring ramps span the shared voxel interval \([s-g,s+g-1]\).

### 6.2 Smoothstep

The clamped smoothstep is:

\[
S(t)=\operatorname{clamp}(t,0,1)^2\left(3-2\operatorname{clamp}(t,0,1)\right).
\]

Properties used by the method:

\[
S(0)=0,\quad S(1)=1,\quad S'(0)=S'(1)=0,
\]

and

\[
S(t)+S(1-t)=1.
\]

### 6.3 Per-axis brick weight

For query component \(q\), support interval \([s_{\min},s_{\max}]\), and weight-one core \([c_{\min},c_{\max}]\):

\[
t_L=\frac{q-s_{\min}}{c_{\min}-s_{\min}},\qquad
t_R=\frac{s_{\max}-q}{s_{\max}-c_{\max}},
\]

\[
w^{(a)}(q)=S(t_L)S(t_R).
\]

At a physical volume edge, support and core may coincide. The implementation substitutes a value of one for the degenerate ramp denominator, avoiding division by zero and keeping the boundary brick active.

The weight is explicitly set to zero if the point lies outside support.

### 6.4 Three-dimensional weight

The brick's full weight is separable:

\[
w_i(\mathbf{x})=w_i^{(x)}(x)w_i^{(y)}(y)w_i^{(z)}(z).
\]

### 6.5 Blended field

For all bricks with nonzero support weight:

\[
F(\mathbf{x})=
\frac{\sum_i w_i(\mathbf{x})f_i(\mathbf{x})}
{\sum_i w_i(\mathbf{x})}.
\]

The denominator is clamped by `ensemble_blend_epsilon`, default \(10^{-8}\). If no brick receives a valid weight because of numerical or malformed geometry, the code falls back to the original hard router.

Because adjacent ramps are symmetric, their weights already sum to one in ideal overlap geometry. The explicit denominator makes the method robust at volume boundaries, edges, corners, and floating-point limits.

### 6.6 Continuity properties

- At a face, two models blend.
- At an edge, four models blend.
- At a corner, eight models blend.
- Smoothstep is \(C^1\) at ramp endpoints.
- The weighted field is continuous as long as individual child predictions are finite.
- The field is not guaranteed to have identical derivatives from every child. The derivative of the weighted composite is smooth at support endpoints, but remaining within-overlap gradient disagreement can still affect shading.

### 6.7 Hard and blend routing modes

New model option:

- `ensemble_routing_mode = hard`: original behavior and default.
- `ensemble_routing_mode = blend`: new partition-of-unity composite.

The original hard path was kept unchanged. A unit test compares it with a reimplementation of the old code bit for bit.

### 6.8 Optional fast path

`ensemble_blend_fast_path` checks whether a point lies in the owning brick's weight-one region. Such points can be evaluated by hard routing because only one brick has nonzero unit weight. Overlap points use the full blend.

The optimization is mathematically exact and tested against the full blend, including input gradients. It is disabled by default because the vectorized full blend measured faster on the RTX 3050, approximately 15.4 versus 13.7 million points per second in the development measurement.

## 7. Interface fine-tuning

### 7.1 What is frozen and trained

The script clones the source checkpoint to a new directory. It never overwrites the baseline unless explicitly requested.

Frozen:

- Adaptive feature-grid values.
- Grid transformation matrices.
- Every other child parameter outside the decoder.

Trainable:

- Decoder MLP parameters only, approximately 37 thousand parameters or 4% of the complete model.

### 7.2 Face pairs and sampled regions

`enumerate_face_adjacencies` lists every adjacent pair exactly once and stores:

- `left`, `right`: flattened child indices.
- `left_index_dhw`, `right_index_dhw`: 3D brick indices.
- `normal_axis_dhw`: interface-normal axis in storage order.
- `normal_axis_xyz`: matching model-coordinate axis.
- `split_index`: decomposition split number.

For each pair, the shared sampling box is the intersection of the two support extents. Points are sampled continuously and uniformly within the end-exclusive voxel box, then converted from DHW to normalized XYZ.

### 7.3 Ground-truth sampling

Ground truth uses PyTorch `grid_sample` with trilinear interpolation and `align_corners=True`, matching the original training targets.

### 7.4 Reconstruction term

For overlap sample \(\mathbf{x}\), ground truth \(y\), and adjacent predictions \(y_i=f_i(\mathbf{x})\), \(y_j=f_j(\mathbf{x})\):

\[
L_{\text{overlap-rec}}=rac{1}{2}left[\operatorname{MSE}(y_i,y)+\operatorname{MSE}(y_j,y)\right].
\]

Each brick also receives uniformly sampled reconstruction anchors from its complete support region. These anchors prevent interface optimization from degrading the rest of a brick.

All reconstruction terms are averaged and normalized by squared data range:

\[
L_{\text{rec}}=\frac{\operatorname{mean}(\text{reconstruction terms})}{(v_{\max}-v_{\min})^2+10^{-12}}.
\]

### 7.5 Value-consistency term

\[
L_{\text{val}}=
\frac{\operatorname{mean}_{(i,j),\mathbf{x}}\left(f_i(\mathbf{x})-f_j(\mathbf{x})\right)^2}
{(v_{\max}-v_{\min})^2+10^{-12}}.
\]

### 7.6 Normal-derivative consistency

For interface-normal axis \(a\), one normalized voxel step is:

\[
\delta=\frac{2}{N_a-1}.
\]

The script forms \(\mathbf{x}^{+}\) and \(\mathbf{x}^{-}\) by adding and subtracting \(\delta\), clamped inside the shared overlap. For child \(i\):

\[
D_{n,i}(\mathbf{x})=
\frac{f_i(\mathbf{x}^{+})-f_i(\mathbf{x}^{-})}
{x_a^{+}-x_a^{-}}.
\]

The derivatives are multiplied by \(\delta\) before their MSE, expressing the mismatch in per-voxel value units. The normalized gradient loss is:

\[
L_{\text{grad}}=
\frac{\operatorname{mean}(\delta D_{n,i}-\delta D_{n,j})^2}
{(v_{\max}-v_{\min})^2+10^{-12}}.
\]

### 7.7 Warm-up and complete fine-tune objective

At iteration \(t\), zero-indexed in code:

\[
r(t)=\min\left(1,\frac{t+1}{N_{\text{warmup}}}\right).
\]

The complete objective is:

\[
L=L_{\text{rec}}+r(t)\lambda_v L_{\text{val}}+r(t)\lambda_g L_{\text{grad}}.
\]

Scheduled study values:

| Variable | CLI name | Value |
|---|---|---:|
| Iterations | `iterations` | 1,500 |
| Overlap points per face pair | `points_per_iteration` | 4,096 |
| Interior anchors per brick | `interior_points` | 2,048 |
| Learning rate | `lr` | 5e-4 |
| Value weight | `value_weight` | 1.0 |
| Gradient weight | `gradient_weight` | 0.05 |
| Warm-up | `warmup_iterations` | 250 |
| Seed | `seed` | 42, 1, or 2 according to checkpoint |
| Time guard | `max_minutes` | 20 |
| GPU-memory guard | `max_gpu_gb` | 3.7 |

The script default for `gradient_weight` is zero. The 0.05 value was explicitly passed for the consistency runs.

### 7.8 Reconstruction-only control

The control uses the exact same checkpoint cloning, trainable decoder parameters, 1,500 iterations, overlap sampling, anchor sampling, optimizer, and learning rate, but:

\[
\lambda_v=0,\qquad \lambda_g=0.
\]

Therefore:

\[
L=L_{\text{rec}}.
\]

This control determines whether any gain comes from interface consistency or simply from extra decoder training.

### 7.9 Fine-tune safety guards

The run stops if:

- Total loss becomes non-finite.
- Elapsed time exceeds `max_minutes`.
- Peak allocated GPU memory exceeds `max_gpu_gb`.
- After 150 history entries, normalized reconstruction loss exceeds five times the median of its first 100 values.

Fine-tune metadata, JSON logs, and CSV logs are stored with the checkpoint.

## 8. Seam-aware evaluation

### 8.1 Exact volume reconstruction

The evaluator queries the model on every voxel of the complete \(512\times128\times128\) lattice. Reconstruction is performed in slabs along D to limit memory. Coordinates follow the same align-corners mapping as training.

### 8.2 PSNR

For data range \(R=v_{\max}-v_{\min}\) and mean squared error \(E\):

\[
\operatorname{PSNR}=20\log_{10}(R)-10\log_{10}(E).
\]

The implementation clamps MSE to at least \(10^{-30}\) before the logarithm.

### 8.3 Boundary and interior PSNR

For each internal split at voxel \(s\) and boundary width \(b\), the mask includes:

\[
[s-b,s+b)
\]

along the normal axis and the complete volume along tangential axes. The study uses `boundary_band_cells = 2`. Masks from all three internal split planes in the 2 by 2 by 2 decomposition are unioned.

- Boundary PSNR uses voxels in the union mask.
- Interior PSNR uses all remaining voxels.
- Per-plane PSNR is also saved.

### 8.4 Interface sample locations

For adjacent bricks whose right brick begins at voxel \(s\), random tangential coordinates are drawn within their shared support. The normal coordinate is fixed at:

\[
s-0.5,
\]

the physical half-voxel plane where the original hard router changes ownership.

The study draws 65,536 samples per face pair.

### 8.5 Child-pair value disagreement

This measures the trained models themselves, independent of the routing rule:

\[
J_{\text{pair-value}}=
\frac{|f_i(\mathbf{x})-f_j(\mathbf{x})|}{R}.
\]

Mean, median, 95th percentile, and maximum are saved.

### 8.6 Child-pair gradient disagreement

Let one normalized voxel step be:

\[
h=\frac{2}{N_a-1}.
\]

The code evaluates each child at \(\mathbf{x}\pm h/2\) and compares its one-voxel central value difference:

\[
J_{\text{pair-grad}}=
\frac{|[f_i(\mathbf{x}+h/2)-f_i(\mathbf{x}-h/2)]-[f_j(\mathbf{x}+h/2)-f_j(\mathbf{x}-h/2)]|}{R}.
\]

This is reported in per-voxel derivative units.

### 8.7 Routed composite value jump

The viewer sees the routed field \(F\), not both child functions separately. Define:

\[
\epsilon=10^{-3}h.
\]

The composite value jump is:

\[
J_{\text{value}}=
\frac{|F(\mathbf{x}+\epsilon)-F(\mathbf{x}-\epsilon)|}{R}.
\]

Hard routing exposes child disagreement. Blending turns the discontinuous switch into a smooth field. The finite \(\epsilon\) means a continuous field can still produce a tiny nonzero difference from its normal slope, so values near \(6\times10^{-6}\) are described as the numerical or finite-sampling floor rather than exact zero.

### 8.8 Routed composite gradient jump

The evaluator compares one-sided normal changes on either side:

\[
J_{\text{grad}}=
\frac{|[F(\mathbf{x}+h)-F(\mathbf{x}+\epsilon)]-[F(\mathbf{x}-\epsilon)-F(\mathbf{x}-h)]|}{R}.
\]

The same calculation on the raw trilinearly sampled volume gives the ground-truth gradient-change reference. For g4 geometry, its mean is approximately \(1.75\times10^{-3}\).

### 8.9 Throughput and active-model fractions

The evaluator and fair benchmark use 262,144 uniformly random points generated from a fixed seed.

The dedicated benchmark:

- Loads every compared model once.
- Performs three warm-up cycles.
- Runs ten repeated queries per model in each round.
- Uses seven rounds.
- Reverses model order on alternate rounds to reduce clock and thermal bias.
- Synchronizes CUDA before and after timing.
- Reports median, minimum, and maximum million points per second.

The evaluator also records the fraction of queries activating exactly 1, 2, 4, or 8 bricks.

## 9. Rendering evaluation

The original repository's nerfacc and Tiny CUDA Neural Networks dependencies were unavailable in the Windows/PyTorch environment. A self-contained PyTorch fixed-step emission-absorption renderer was implemented so every model and the raw volume use identical camera rays, sample points, transfer function, and compositing.

### 9.1 Study render settings

- Resolution: 512 by 256.
- Samples per ray: 768.
- Same azimuth, polar angle, distance, field of view, background, and Plume transfer function for all runs.
- Raw volume reference uses trilinear `grid_sample`.

### 9.2 Transfer function

The ParaView-style JSON colormap is piecewise-linearly interpolated for RGB and opacity. A scalar is normalized using the raw data minimum and maximum before transfer-function lookup.

### 9.3 Ray-box intersection and sampling

Each camera ray is intersected with the normalized volume box. Samples use a fixed world-space step and are converted into normalized XYZ before querying the field.

### 9.4 Emission-absorption compositing

For transfer-function opacity \(o_s\), density multiplier \(\sigma\), and step length \(\Delta\):

\[
\alpha_s=1-\exp(-o_s\sigma\Delta).
\]

Color and transmittance update front to back:

\[
\mathbf{C}\leftarrow\mathbf{C}+T\alpha_s\mathbf{c}_s,
\]

\[
T\leftarrow T(1-\alpha_s).
\]

Rays with transmittance below \(10^{-3}\) are periodically removed from further evaluation. The remaining transmittance blends with the background.

### 9.5 Render metrics

- PSNR between model render and raw-volume render.
- SSIM between model render and raw-volume render.
- Complete frame time.
- Peak allocated GPU memory.
- Absolute RGB error images amplified eight times for visualization.

These metrics compare our variants fairly. They are not numerically comparable with render measurements produced by the paper's different renderer.

## 10. Model variants and naming

| Variant label | Checkpoint/training | Routing |
|---|---|---|
| g1 hard | Base g1 | Hard |
| g4 hard | Base g4 | Hard |
| g16 hard | Base g16 | Hard |
| g16 27-cubed hard | Larger g16 feature grids | Hard |
| g16 32-cubed hard | Larger g16 feature grids | Hard |
| g4 blend | Base g4, no extra training | Blend |
| g16 blend | Base g16, no extra training | Blend |
| g4 consistency-FT hard | Decoder fine-tune with consistency | Hard |
| g4 consistency-FT plus blend | Same fine-tuned checkpoint | Blend |
| g4 recon-FT hard | Reconstruction-only decoder control | Hard |
| g4 recon-FT plus blend | Same control checkpoint | Blend |
| g16 recon-FT hard/blend | Equal-extra-training control for g16 | Hard or blend |

## 11. Main results

Mean and standard deviation are over available seeds. All 24-cubed rows have the same 923,136 parameters.

| Method | Seeds | Params | Global PSNR | Boundary PSNR | Render PSNR | SSIM | Value jump | Gradient jump | Mpts/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g1 hard | 1 | 0.92M | 50.92 | 47.37 | 43.11 | 0.9919 | 2.64e-3 | 2.89e-3 | 11.5 |
| g4 hard | 3 | 0.92M | 50.16 +/- 0.08 | 46.56 +/- 0.17 | 42.23 +/- 0.85 | 0.9907 | 2.69e-3 | 2.50e-3 | 11.5 |
| g16 hard, paper style | 3 | 0.92M | 48.66 +/- 0.43 | 45.48 +/- 0.38 | 41.25 +/- 0.54 | 0.9883 | 3.30e-3 | 2.93e-3 | 11.9 |
| g16 hard, 27-cubed | 1 | 1.30M | 49.57 | 46.53 | 42.23 | 0.9902 | 2.75e-3 | 3.05e-3 | 11.7 |
| g16 hard, 32-cubed | 1 | 2.14M | 50.62 | 47.45 | 43.57 | 0.9924 | 2.56e-3 | 2.99e-3 | 11.8 |
| g16 plus recon-FT hard | 1 | 0.92M | 49.04 | 46.09 | 42.08 | 0.9893 | 2.56e-3 | 2.92e-3 | 11.5 |
| g16 blend | 3 | 0.92M | 49.32 +/- 0.36 | 47.17 +/- 0.16 | 42.06 +/- 0.45 | 0.9900 | 6.34e-6 | 2.43e-3 | 8.7 |
| g4 blend, no training | 3 | 0.92M | 50.35 +/- 0.07 | 47.80 +/- 0.01 | 42.42 +/- 0.82 | 0.9911 | 5.81e-6 | 2.15e-3 | 9.2 |
| g4 consistency-FT hard | 3 | 0.92M | 50.53 +/- 0.31 | 47.17 +/- 0.26 | 43.19 +/- 0.16 | 0.9918 | 1.97e-3 | 2.45e-3 | 12.1 |
| **g4 consistency-FT plus blend** | 3 | 0.92M | **50.66 +/- 0.31** | **48.00 +/- 0.21** | **43.33 +/- 0.17** | **0.9921** | **5.69e-6** | **2.12e-3** | 9.1 |
| g4 recon-FT hard | 2 | 0.92M | 50.83 +/- 0.31 | 47.45 +/- 0.33 | 43.42 +/- 0.18 | 0.9921 | 2.00e-3 | 2.47e-3 | 12.2 |
| **g4 recon-FT plus blend, best measured** | 2 | 0.92M | **50.98 +/- 0.29** | **48.43 +/- 0.25** | **43.58 +/- 0.19** | **0.9924** | **5.71e-6** | **2.14e-3** | 9.6 |

## 12. Paired same-seed comparisons

| Comparison | Global | Boundary | Render | Value jump | Gradient jump | Seeds won |
|---|---:|---:|---:|---:|---:|---:|
| Consistency-FT blend vs g16 hard | +2.00 +/- 0.12 dB | +2.51 dB | +2.08 dB | 579 times lower | 1.38 times lower | 3/3 on every metric |
| g4 blend alone vs g16 hard | +1.69 dB | +2.32 dB | +1.16 dB | 567 times lower | 1.36 times lower | 3/3, render 2/3 |
| Consistency-FT blend vs g4 hard | +0.51 dB | +1.44 dB | +1.10 dB | 474 times lower | 1.18 times lower | 3/3 on every metric |
| Blend alone vs g4 hard | +0.20 +/- 0.03 dB | +1.24 dB | +0.19 dB | 463 times lower | 1.16 times lower | 3/3 on every metric |
| Consistency-FT blend vs g16 recon-FT | +1.50 dB | +1.77 dB | +1.08 dB | 445 times lower | 1.32 times lower | 1/1 |
| g4 recon-FT blend vs g16 recon-FT | +1.73 dB | +2.17 dB | +1.36 dB | 442 times lower | 1.31 times lower | 1/1 |
| Consistency-FT blend vs g16 32-cubed | -0.09 dB | +0.40 dB | -0.41 dB | 444 times lower | 1.36 times lower | 1/1 comparison |
| Consistency-FT vs reconstruction-only FT | -0.21 dB | -0.38 dB | -0.25 dB | approximately equal | approximately equal | 0/2 |

## 13. Capacity result

| g16 feature grid | Parameters | g16 global PSNR | Our 24-cubed consistency blend minus g16 | g16 jump relative to ours |
|---|---:|---:|---:|---:|
| 24-cubed | 0.92M | 48.66 | +2.00 dB | approximately 580 times larger |
| 27-cubed | 1.30M | 49.57 | +0.97 dB | approximately 478 times larger |
| 32-cubed | 2.14M | 50.62 | +0.04 dB, effectively tied | approximately 444 times larger |

The 32-cubed g16 model uses approximately 2.3 times the parameters, 8.15 MiB versus 3.52 MiB. It nearly ties global PSNR and slightly wins render PSNR against the consistency variant, but still has lower boundary PSNR and a much larger seam jump. This capacity sweep has one seed.

## 14. Cost and performance

| Measure | Hard routing | Blend | Interpretation |
|---|---:|---:|---|
| Query throughput | 11.5 to 12.2 Mpts/s | 8.7 to 9.6 Mpts/s | Blend is about 22% slower |
| Frame time, 512 by 256 and 768 samples/ray | 3.5 to 4.3 s | 3.5 to 4.3 s | No consistent frame-level penalty |
| Peak evaluation memory | 0.181 GB | 0.182 GB | Effectively unchanged |
| Parameters | 923,136 | 923,136 | Blend adds no parameters |
| Extra training | None | None for blend, 4 to 7 min for optional fine-tune | Existing checkpoints can be blended |
| Relative stored training voxels | g4 about 1.15 times, g16 about 1.66 times | Same geometry | g4 stores about 30% fewer overlap voxels than g16 |

## 15. Negative finding and final claim

The explicit consistency objective did not beat its fair reconstruction-only control.

- Reconstruction-only fine-tuning won global, boundary, and render PSNR on both available paired seeds.
- Consistency reduced pairwise brick disagreement by only about 3.5% relative to the control.
- Gradient-jump improvement relative to the control was about 1%.
- Blending had already reduced the routed value jump to about \(6\times10^{-6}\), leaving little for value consistency to improve.

Therefore the final supported claim is:

> A narrow four-cell overlap with partition-of-unity blending removes the hard-routing seam and improves accuracy at the same model size. A short decoder fine-tune adds accuracy, but explicit consistency was not better than reconstruction-only extra training in the tested setting.

Unsupported overclaim to avoid:

> The consistency loss is responsible for the two-decibel gain.

## 16. Unit tests

Nineteen CPU tests cover:

1. Parsing integer tuples without `eval`.
2. Unique flattened brick indices.
3. Twelve face pairs for a 2 by 2 by 2 grid.
4. Core bricks cover each voxel exactly once.
5. Expanded support matches original job-generation extents.
6. Configured versus actual ghost-cell convention.
7. DHW and XYZ round trips.
8. Normalized extents match the original formula.
9. Weight equals one in core and zero outside support.
10. Ramp monotonicity and finite values.
11. Exact partition of unity from symmetric ramps.
12. Smoothstep endpoint and midpoint values.
13. Original hard router unchanged.
14. Identical neighboring fields give identical hard and blended outputs.
15. Constant-neighbor transition is smooth and monotonic.
16. Normalized weights sum to one everywhere.
17. Active counts are 2, 4, and 8 at face, edge, and corner overlaps.
18. Blend input gradients remain finite.
19. Optional fast path matches full blend in values and gradients for g1, g4, and g16.

## 17. Code map

| File | Responsibility |
|---|---|
| `Code/Models/ensemble_utils.py` | Axis conversion, core and support extents, manifests, adjacency, smoothstep, blend weights |
| `Code/Models/ensemble_SRN.py` | Original hard router, child query, support weights, full blend, optional fast path |
| `Code/Models/options.py` | New routing options and backward-compatible defaults |
| `Code/start_jobs.py` | Shared decomposition geometry, checkpoint manifest, resumable job queue |
| `Code/train.py` | Training seed, time and memory recording, run guards |
| `Code/finetune_boundaries.py` | Decoder-only fine-tuning and controls |
| `Code/evaluate_seams.py` | Full-volume reconstruction, boundary PSNR, interface metrics, profiles, error slices |
| `Code/compare_renderings.py` | Fixed-step renderer, image PSNR and SSIM, error images |
| `Code/benchmark_throughput.py` | Interleaved median query-speed benchmark |
| `Code/aggregate_study.py` | Aggregates seeds, paired comparisons, capacity tables, cost tables, plots |
| `Tests/test_ensemble_utils.py` | Geometry and weight tests |
| `Tests/test_blending.py` | Routing, continuity, active-count, gradient, and fast-path tests |
| `run_seamless_pipeline.ps1` | Initial train/fine-tune/evaluate/render workflow |
| `run_seed_capacity_study.ps1` | Resumable multi-seed, capacity, and control study |

## 18. New options and important variables

### 18.1 Ensemble options

| Variable | Meaning | Default |
|---|---|---|
| `ensemble_routing_mode` | `hard` or `blend` | `hard` |
| `ensemble_blend_epsilon` | Minimum denominator for normalized blend | `1e-8` |
| `ensemble_blend_fast_path` | Use exact hard path inside weight-one core | `false` |
| `actual_ghost_cells` | `ensemble_ghost_cells + 1` | Derived |
| `local_min_extents`, `local_max_extents` | Normalized support limits per model | Derived buffers |
| `core_min_extents`, `core_max_extents` | Normalized weight-one limits per model | Derived buffers |

### 18.2 Fine-tune arguments

| Argument | Meaning |
|---|---|
| `load_from` | Source checkpoint |
| `save_name` | New cloned checkpoint name |
| `device` | Model and optimization device |
| `data_device` | Raw-volume device |
| `iterations` | Maximum fine-tune steps |
| `points_per_iteration` | Continuous overlap points sampled per face pair per step |
| `interior_points` | Reconstruction anchors sampled per brick per step |
| `lr` | Decoder Adam learning rate |
| `value_weight` | \(\lambda_v\) |
| `gradient_weight` | \(\lambda_g\) |
| `warmup_iterations` | Steps for linear consistency ramp |
| `seed` | Torch, Python, and sampling-generator seed |
| `max_minutes` | Wall-time guard |
| `max_gpu_gb` | Peak allocated-memory guard |
| `log_every` | Logging interval |
| `overwrite` | Whether an existing destination checkpoint may be replaced |

### 18.3 Evaluation arguments

| Argument | Study value or role |
|---|---|
| `load_from` | Checkpoint |
| `routing_mode` | `hard` or `blend` |
| `run_id` | Output identifier |
| `device` | Query device |
| `data_device` | Raw-volume storage device |
| `boundary_band_cells` | 2 |
| `face_samples` | 65,536 per interface |
| `throughput_points` | 262,144 |
| `throughput_repeats` | 20 in evaluator |
| `seed` | 42 for deterministic interface sampling unless run-specific |
| `save_reconstruction` | Optionally save complete array |

### 18.4 Rendering arguments

| Argument | Meaning |
|---|---|
| `runs` | `run_id=checkpoint:hard_or_blend` specifications |
| `hw` | Output height and width |
| `samples` | Samples per ray |
| `azi`, `polar`, `dist`, `fov` | Camera parameters |
| `density` | Opacity-density multiplier |
| `background` | RGB background |
| `colormap` | ParaView-style JSON transfer function |
| `data` | Raw volume filename |
| `crop` | Montage crop around a split |
| `montage_runs` | Selected run IDs for montage |

## 19. Reproduction commands

From `APMGSRN`:

```powershell
$py = 'D:\Computer_Graphics_env\.venv\Scripts\python.exe'

& $py -m pytest -q Tests

foreach ($g in 'g1','g4','g16') {
  & $py Code/start_jobs.py `
    --settings "train_plume_dd_$g.json" `
    --devices cuda:0 `
    --data_devices same `
    --skip_existing true
}

./run_seamless_pipeline.ps1

# Full seed, capacity, control, evaluation and rendering study
./run_seed_capacity_study.ps1 -TrainDeadline 14:00

# Re-evaluate without new training
./run_seed_capacity_study.ps1 -SkipTraining

& $py Code/aggregate_study.py --output_root Output/SeedCapacityStudy
```

`--data_devices same` is important on this Windows laptop. Keeping data on CPU makes the original DataLoader create worker processes that can exhaust the small page file.

## 20. GPU budget

| Stage | Approximate GPU time |
|---|---:|
| Initial g1, g4, g16 training | 20.5 min |
| Initial fine-tune | 2.8 min |
| Initial evaluations and renders | About 9 min over repeated pipeline runs |
| Additional seeds and g16 capacity runs | About 90 min |
| Additional consistency and reconstruction controls | About 5 min plus light evaluation |
| **Total** | **About 2 h 10 min** |

## 21. Limitations and exact claim boundaries

1. Main evidence uses one dataset, Plume.
2. Only a 2 by 2 by 2 decomposition is evaluated.
3. g4 and equal-size g16 key conditions have three seeds.
4. g1, g16 27-cubed, g16 32-cubed, and g16 reconstruction control have one seed.
5. g4 reconstruction-only control has two seeds.
6. g2 was not completed within the GPU budget.
7. Render PSNR varies noticeably across seeds, so small render differences are less reliable than full-volume PSNR.
8. The custom renderer differs from the paper's renderer.
9. Value jump near \(6\times10^{-6}\) is finite-sampling residual, not mathematical exact zero.
10. Gradient jump is a finite-difference approximation. It includes true ground-truth variation and model gradient noise.
11. Blending adds no parameters but does add query computation.
12. The 2.3-times capacity-to-tie statement is based on a single g16 32-cubed seed.

## 22. Answers to likely technical questions

### Why does blending improve PSNR when it was introduced for seams?

Near a boundary, each child has learned overlapping data but has independent error. A weighted average can reduce uncorrelated prediction error while avoiding the abrupt selection of one child. Narrow overlap also leaves more capacity for unique voxels than g16.

### Is partition of unity enough to guarantee correctness?

It guarantees normalized weights and removes the hard discontinuity. It does not guarantee that the composite equals ground truth or that child gradients agree. Accuracy still depends on the trained children.

### Why smoothstep instead of a linear ramp?

A linear ramp is continuous but has a nonzero slope that starts and stops abruptly at support boundaries. Smoothstep has zero endpoint derivatives, giving a \(C^1\) transition at the ramp boundaries.

### Does blending change the output away from interfaces?

No. In the weight-one core, only the owning brick has weight one. The composite equals its prediction exactly.

### Why fine-tune only decoders?

Decoders are a small fraction of the parameters, so optimization is fast and memory-light. Freezing grids and transformations protects the learned spatial representation and makes the comparison easier to control.

### Why did the consistency loss fail to beat reconstruction-only training?

Blending already makes the routed value field continuous. Additional decoder capacity is better spent fitting ground truth than making independent child predictions identical throughout the overlap. The consistency objective can trade reconstruction accuracy for agreement that the composite no longer needs.

### Is the method better than g1?

The single-seed g1 hard model has slightly higher global PSNR than the consistency variant, but much worse value continuity and lower boundary PSNR. The reconstruction-fine-tuned blend slightly exceeds g1 global PSNR in its two-seed mean. A complete three-seed g1 comparison remains future work.

### Why not claim zero seam?

The mathematical composite is continuous under valid finite child outputs, but the measured metric evaluates two points separated by finite \(2\epsilon\), so it records real field slope and floating-point effects. “At the numerical floor” is more accurate.

### What is the strongest supported result?

At equal 923,136-parameter size over three matched seeds, g4 consistency-fine-tune plus blending beats g16 hard by 2.00 dB global, 2.51 dB boundary, 2.08 dB render PSNR, and 579 times in routed value-jump reduction.
