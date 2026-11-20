# Detailed summary of the original APMGSRN paper

## Citation

Skylar W. Wurster, Tianyu Xiong, Han-Wei Shen, Hanqi Guo, and Tom Peterka. **Adaptively Placed Multi-Grid Scene Representation Networks for Large-Scale Data Visualization.** *IEEE Transactions on Visualization and Computer Graphics*, 30(1), 2024, presented at IEEE VIS 2023. arXiv:2308.02494.

## The paper in one paragraph

The paper proposes **APMGSRN**, a scene representation network for compressing and visualizing scientific scalar volumes. Instead of using one fixed dense feature grid, it uses multiple small feature grids whose affine transformations are learned, allowing model capacity to move toward difficult, high-error regions. It also proposes a **domain decomposition** system that partitions a large volume into bricks, trains one APMGSRN per brick in parallel across GPUs, and selects a brick at inference using a spatial hash. Finally, it releases a plug-and-play PyTorch neural volume renderer. The paper reports strong reconstruction quality at fixed model sizes, training of data as large as 450 GB, and interactive progressive rendering. Its unresolved domain-decomposition issue is that independently trained bricks can create visible seams. The paper reduces these with ghost-cell overlap but does not make neighboring networks communicate.

## 1. What problem the paper tried to solve

The authors identify two weaknesses in prior scientific-data scene representation networks.

### 1.1 Fixed allocation of network capacity

Existing feature-grid networks generally distribute their parameters uniformly over the spatial domain. Scientific volumes are rarely uniform in complexity. Large regions may be nearly empty or smooth, while small regions contain turbulent structures, shocks, particles, or other difficult features. A uniform grid wastes parameters in easy regions and starves complex regions.

Tree-based adaptive representations can refine difficult regions, but the paper argues that they introduce tree storage, traversal, pruning, and manually scheduled structural updates. Their cost also scales poorly with dimension and tree depth.

### 1.2 Training very large volumes

A large volume may not fit in GPU memory, and a sufficiently large single model may train slowly or require too much memory. Out-of-core random sampling from CPU memory or storage introduces expensive data transfers and I/O. The paper instead targets model-parallel training on the multiple GPUs commonly available on scientific-computing nodes.

## 2. Main contributions

The paper claims three practical contributions.

1. **Adaptively Placed Multi-Grid SRN:** multiple learnable feature grids move, scale, rotate, and shear to allocate features where reconstruction error is high.
2. **Domain decomposition training and inference:** the volume is divided into bricks, one network is trained per brick, jobs are scheduled across GPUs, and a coordinate hash selects the correct network at inference.
3. **Open neural volume renderer:** a PyTorch renderer supports raw volumes and coordinate-to-density neural models such as APMGSRN, fVSRN, and NGP, with arbitrary transfer functions and progressive rendering.

The architecture and equations are mainly on paper pages 3 to 5. Domain decomposition is on pages 5 and 6. Experiments are on pages 7 to 9.

## 3. Background concepts

### 3.1 Scene representation networks

A scene representation network, also called an implicit neural representation or coordinate network, maps a coordinate directly to a value. For this paper:

\[
f:\;[-1,1]^3 \rightarrow \mathbb{R}
\]

The input is a normalized 3D coordinate and the output is the scalar value at that position. The network weights and feature parameters become the stored representation.

### 3.2 Feature-grid encoding

A dense MLP must learn spatial detail entirely in its weights. Feature-grid models instead interpolate trainable features near the query coordinate and use a small MLP to decode them. This generally improves training and inference speed.

The paper compares APMGSRN with:

- **fVSRN:** a dense feature-grid SRN with Fourier coordinate encoding.
- **NGP:** Instant Neural Graphics Primitives, using a multiresolution hash-grid encoding.
- Fully connected implicit models and tree-based adaptive models in related work.

## 4. APMGSRN architecture

The model has an encoder and decoder:

\[
f(\mathbf{x})=d(e(\mathbf{x})).
\]

### 4.1 Multiple feature grids

The encoder contains \(M\) learnable feature grids. Each grid has spatial resolution \(D\times H\times W\) and \(C\) feature channels. The complete feature tensor has shape:

\[
[M,C,D,H,W].
\]

Each grid has a learned 4 by 4 affine transformation matrix \(G_i\). It maps a global coordinate into that grid's local coordinate system. The upper-left 3 by 3 block controls scale, rotation, and shear. The last column controls translation.

The local grid domain is \([-1,1]^3\). If a transformed query lies inside the local cube, the model trilinearly interpolates the grid. If it lies outside, that grid contributes a zero feature vector.

For grid \(i\):

\[
e_i(\mathbf{x}_l)=
\begin{cases}
\operatorname{interp}(F_i,\mathbf{x}_l), & -1\leq \mathbf{x}_l\leq 1\\
0, & \text{otherwise.}
\end{cases}
\]

The encoder concatenates all grid features:

\[
e(\mathbf{x}_g)=e_0(G_0(\mathbf{x}_g))\oplus e_1(G_1(\mathbf{x}_g))\oplus\cdots\oplus e_{M-1}(G_{M-1}(\mathbf{x}_g)).
\]

The resulting vector contains \(C\cdot M\) features.

### 4.2 Decoder and value scaling

The decoder is a shallow two-layer MLP with 64 neurons per layer, ReLU activations, and no bias terms. The paper uses Tiny CUDA Neural Networks for tensor-core acceleration.

Scientific data can have arbitrary value ranges. The paper stores the minimum and maximum scalar values and trains the MLP to produce a normalized value. The physical output is recovered as:

\[
d(\mathbf{y})=m(\mathbf{y})(\max-\min)+\min.
\]

## 5. How the feature grids learn where to move

### 5.1 Why reconstruction loss alone is insufficient

If a point lies outside a grid, that grid returns zero features. A normal reconstruction loss therefore gives the grid transformation no useful gradient from difficult points outside its current support. High-error regions cannot reliably pull a grid toward them.

### 5.2 Feature density

The paper defines feature density as the amount of grid capacity covering a point. A simple count of covering grids ignores grid scale. A small grid packs its features more densely than a global grid. The determinant of the transformation's 3 by 3 spatial block represents this density change.

An exact box-support density still has zero gradient outside the grid, so the authors approximate it with a differentiable flat-top Gaussian, also called a super-Gaussian. In simplified form:

\[
\rho(\mathbf{x})=\sum_{i=0}^{M-1}\det(G_{i,0:3,0:3})
\exp\left[-\sum_{d=0}^{2}(G_i(\mathbf{x})_d)^{2p}\right].
\]

They use \(p=10\), which produces a box-like plateau while retaining nonzero gradients outside the nominal grid.

### 5.3 Target feature density

Let \(h(\mathbf{x})\) be prediction error and \(\bar h\) the average batch error. The target density rescales current density according to relative error:

\[
\rho^*(\mathbf{x})=
\exp\left[
\frac{\bar h}{h(\mathbf{x})+\epsilon}
\log(\rho_{\text{scaled}}(\mathbf{x})+\epsilon)
\right].
\]

If local error equals the average, target density remains unchanged. High relative error increases target importance, while low relative error decreases it.

### 5.4 Density loss

The transformation matrices minimize a KL-divergence-style relative-entropy loss:

\[
\mathcal{L}_{\text{density}}=
\frac{1}{|X|}\sum_{\mathbf{x}\in X}
\rho^*(\mathbf{x})
\log\left(
\frac{\rho^*(\mathbf{x})}{\rho_{\text{scaled}}(\mathbf{x})}
\right).
\]

The ordinary reconstruction loss is:

\[
\mathcal{L}_{\text{rec}}=
\frac{1}{|X|}\sum_{\mathbf{x}\in X}
(f(\mathbf{x})-v(\mathbf{x}))^2.
\]

The reconstruction loss updates feature values and decoder weights. The density loss updates grid transformations.

## 6. Original training routine

- All models train for at most 50,000 iterations or until early stopping.
- Each iteration samples 100,000 coordinates uniformly from \([-1,1]^3\).
- Ground truth is obtained by trilinear interpolation.
- Adam uses \(\beta_1=0.9\), \(\beta_2=0.99\).
- Feature grids and decoder use learning rate 0.01.
- Transformation matrices use learning rate 0.001.
- Transformation learning starts after 500 iterations, because early errors are dominated by random initialization.
- Transformation learning stops if the 1,000-iteration moving average of density loss fails to improve by 0.01%, or after 80% of total training.
- Grid transformations begin near global scale. Diagonal terms are sampled from \(\mathcal{N}(1,0.05)\); other entries in the first three rows are sampled from \(\mathcal{N}(0,0.05)\).
- Feature values are initialized from \(U(-0.0001,0.0001)\).
- Decoder weights use Glorot initialization.

The default near-global initialization is important. Figure 4 shows that initializing tiny grids at random positions can leave grids in empty space with almost no useful transformation gradient, producing only 14.434 dB compared with approximately 41.8 to 42.0 dB for the two near-global initializations.

## 7. Domain decomposition

### 7.1 Partitioning

The volume is partitioned into an \(I\times J\times K\) grid of bricks. The authors recommend a decomposition that gives each brick an aspect ratio similar to the feature-grid aspect ratio.

Each brick is extended by ghost cells. Adjacent networks therefore train on overlapping voxels. The paper is explicit that ghost cells reduce but do not guarantee removal of seams.

### 7.2 Parallel training

One APMGSRN is initialized per brick. Only that brick's cropped data is loaded to the assigned GPU. A job queue sends bricks to available GPUs. When a GPU finishes one brick, it receives another.

For each brick, a plateau scheduler reduces learning rate by a factor of ten when reconstruction loss fails to decrease by 0.01% over 500 iterations. Training ends after this trigger occurs three times. Sparse bricks may complete in about 20 seconds.

### 7.3 Inference hash

For a coordinate \(\mathbf{p}\in[-1,1]^3\), the router computes a brick coordinate:

\[
i=\left\lfloor I\frac{p_x+1}{2}\right\rfloor,\qquad
j=\left\lfloor J\frac{p_y+1}{2}\right\rfloor,\qquad
k=\left\lfloor K\frac{p_z+1}{2}\right\rfloor.
\]

Indices at the positive boundary are clamped. The 3D index is flattened as:

\[
i+Ij+IJk.
\]

The selected network receives the coordinate transformed into its local brick domain. The paper's implementation loops through networks and notes that custom CUDA could accelerate inference.

## 8. Ghost cells and the seam limitation

Figure 6 on paper page 8 compares a 4 by 4 by 4 decomposition with one versus sixteen ghost cells.

- **1 ghost cell:** visible seam, 41.590 dB.
- **16 ghost cells:** seam visually reduced, 40.558 dB.

The paper explains the PSNR drop using fixed network capacity. A 256-cubed brick expanded to 288 cubed contains about 42% more voxels. Several networks redundantly learn portions of the same volume, leaving less capacity for the core data.

The authors recommend one or a few ghost cells for general volume rendering and sixteen for tasks such as isosurface extraction where seams are especially distracting. They explicitly propose network communication during training as future work. The original paper does **not** implement smooth blending, interface loss, or decoder coordination.

## 9. Neural volume renderer

The renderer is written in Python and PyTorch and uses nerfacc for CUDA ray marching and compositing. It accepts models that map 3D coordinates to density, along with raw volumes. It supports:

- Arbitrary camera views.
- Transfer functions exported from ParaView.
- Interactive rotate, pan, and zoom controls.
- Batch-size control trading interactivity for complete-frame speed.
- Progressive checkerboard rendering, with a bilinearly upscaled partial image shown between passes.
- Reported interactive update rates from about 30 to 400 frames per second during progressive refinement.

The renderer prioritizes plug-and-play research flexibility rather than maximum optimized speed.

## 10. Datasets

All raw sizes assume single-precision floating-point storage.

| Dataset | Resolution | Raw size | Stored scalar | Structure |
|---|---:|---:|---|---|
| Plume | \(512\times128\times128\) | 32 MB | Velocity magnitude of particles in a solar-plume simulation | Somewhat sparse |
| Nyx | \(256^3\) | 64 MB | Log density of dark matter in a cosmological simulation | Somewhat sparse |
| Supernova | \(432^3\) | 403 MB | Entropy in a supernova simulation | Somewhat sparse |
| Asteroid | \(1000^3\) | 3.73 GB | Volume fraction of asteroid and water fields from an asteroid-ocean impact simulation | Sparse, SciVis Contest 2018 |
| Isotropic | \(1024^3\) | 4 GB | Velocity magnitude of forced isotropic turbulence | Not sparse |
| Rotstrat | \(4096^3\) | 250 GB | Velocity magnitude of rotating stratified turbulence at \(t=4\) | Not sparse |
| Channel | \(7680\times1568\times10240\) | 450 GB | Channel-flow field at Reynolds number approximately 5200, \(t=10\) | Not sparse |

## 11. Quantitative reconstruction results from Table 1

PSNR is measured in data space. The small, medium, and large categories compare approximately equal storage budgets within a category.

### 11.1 Plume

| Size | fVSRN | NGP | APMGSRN single | APMGSRN domain decomposition |
|---|---:|---:|---:|---:|
| Small, 280 KB | 43.924 | 46.713 | **48.954** | - |
| Medium, 4 MB | 49.332 | 50.847 | **56.327** | - |
| Large, 64 MB | 52.165 | 53.282 | 58.134 | **58.363**, 4 by 1 by 1 |

### 11.2 Nyx

| Size | fVSRN | NGP | APMGSRN single | APMGSRN domain decomposition |
|---|---:|---:|---:|---:|
| Small, 280 KB | 29.110 | 28.391 | **29.424** | - |
| Medium, 4 MB | 37.022 | 35.142 | **37.807** | - |
| Large, 64 MB | 42.092 | 42.750 | 44.075 | **46.118**, 2 by 2 by 2 |

### 11.3 Supernova

| Size | fVSRN | NGP | APMGSRN single | APMGSRN domain decomposition |
|---|---:|---:|---:|---:|
| Small, 280 KB | 39.505 | 41.523 | **41.891** | - |
| Medium, 4 MB | 42.697 | 44.811 | **46.831** | - |
| Large, 64 MB | 46.271 | 47.806 | 49.880 | **50.644**, 2 by 2 by 2 |

### 11.4 Asteroid

| Size | fVSRN | NGP | APMGSRN single | APMGSRN domain decomposition |
|---|---:|---:|---:|---:|
| Small, 280 KB | 31.072 | **33.713** | 30.130 | - |
| Medium, 4 MB | 33.783 | **39.012** | 35.799 | - |
| Large, 64 MB | 39.121 | 42.928 | 39.455 | **43.188**, 4 by 4 by 4 |

NGP performs especially well here because the dataset is highly sparse, which favors hash-grid behavior.

### 11.5 Isotropic turbulence

| Size | fVSRN | NGP | APMGSRN single | APMGSRN domain decomposition |
|---|---:|---:|---:|---:|
| Small, 280 KB | 27.379 | 27.007 | **27.622** | - |
| Medium, 4 MB | 31.570 | 30.786 | **31.825** | - |
| Large, 64 MB | 36.858 | 37.198 | 38.120 | **41.086**, 4 by 4 by 4 |

### 11.6 Large-volume-only domain-decomposition cases

| Dataset | Model storage | Decomposition | PSNR | Average time per brick |
|---|---:|---:|---:|---:|
| Rotstrat | 55 MB | 3 by 3 by 3 | 41.322 | 1 min 3 s |
| Rotstrat | 183 MB | 3 by 3 by 3 | 44.294 | 1 min 15 s |
| Rotstrat | 864 MB | 3 by 3 by 3 | 49.074 | 1 min 58 s |
| Channel | 97 MB | 4 by 1 by 6 | 40.055 | 35 s |

The paper reports a total Rotstrat training time of about seven minutes including data I/O when eight GPUs train bricks in parallel.

## 12. Main experimental findings

### 12.1 Fixed-size SRN comparison

APMGSRN ranks first in reconstruction PSNR in all but three single-network settings. Its benefit is strongest on somewhat sparse Plume, Nyx, and Supernova data, where the large models beat other SRNs by more than 2 dB. It underperforms NGP on the highly sparse Asteroid data.

Training time is comparable but generally slower than fVSRN and NGP because APMGSRN performs affine transformations for several grids and trains a second loss. The multi-grid encoder is implemented in PyTorch rather than custom CUDA.

### 12.2 Compression comparison

The paper compares APMGSRN with TTHRESH and SZ3 on Isotropic.

- APMGSRN offers a better compression ratio mainly at an extreme 10,000-times reduction where reconstruction quality is below 30 dB.
- At most other quality levels, TTHRESH and SZ3 compress more strongly.
- APMGSRN decodes at roughly 500 million point queries per second on GPU.
- Its central advantage is arbitrary random access in the compressed representation. Traditional compressors generally require decompression before trilinear queries.

For 250 GB Rotstrat, SZ3 takes 38 minutes to compress and 14 minutes to decompress, giving 20.60 dB at 509-times reduction. The paper's small and medium Rotstrat networks obtain 41.32 and 44.29 dB at approximately 4,791-times and 1,432-times reduction. TTHRESH and SZ3 run out of the machine's 1 TB memory on the 450 GB Channel dataset; TTHRESH also fails on Rotstrat.

### 12.3 Domain decomposition

For the smaller datasets, domain decomposition improves PSNR at the same total storage budget because each brick obtains its own decoder and feature grids. Its disadvantage is longer serial training if only one GPU is available. Parallel training hides that cost on multi-GPU hardware.

### 12.4 Rendering comparison on Supernova

Figure 7 renders 1024 by 1024 images from large 64 MB models.

| Representation | Time | GPU memory | Render PSNR | SSIM |
|---|---:|---:|---:|---:|
| Raw reference | 0.83 s | 0.50 GB | Reference | Reference |
| fVSRN | 1.49 s | 0.42 GB | 30.86 dB | 0.911 |
| NGP | 1.10 s | 0.36 GB | 31.67 dB | 0.930 |
| APMGSRN single | 1.75 s | 1.89 GB | 35.49 dB | 0.958 |
| APMGSRN domain decomposition | 1.63 s | 0.41 GB | **37.41 dB** | **0.973** |

NGP renders fastest among neural models and uses the least memory. APMGSRN renders more slowly but produces the highest image quality. The single APMGSRN's memory is high because every query is transformed into all 64 local grid coordinate systems.

### 12.5 Rotstrat model versus downsampled data

The paper compares a 183 MB domain-decomposition network representing the 250 GB Rotstrat volume with a 198 MB raw volume downsampled by 11 along each dimension. Both are rendered at 8000 by 8000 pixels with identical visual mapping.

- Neural model: **30.70 dB, SSIM 0.946**.
- Downsampled raw data: **22.24 dB, SSIM 0.918**.

The model preserves features completely missed by the similar-size downsampled grid.

## 13. Hardware used in the paper

Training and compression experiments:

- NVIDIA DGX system.
- Eight NVIDIA A100 GPUs with 40 GB each.
- 64-core AMD EPYC 7742 CPU.
- 1 TB system memory.
- Lustre-based file system.

Rendering evaluation:

- Windows 11.
- NVIDIA RTX 2080 Ti.
- 12-core AMD Ryzen 9 3900X.
- 32 GB system memory.

## 14. Limitations acknowledged by the paper

1. Memory and training cost increase with the number of feature grids because each query is transformed and interpolated in every grid.
2. The pure-PyTorch multi-grid encoder lacks the efficiency of a custom CUDA kernel.
3. The flexible Python renderer is slower than heavily optimized contemporary neural renderers.
4. Ghost cells only reduce boundary artifacts. They do not guarantee agreement between independently trained brick networks.
5. Wider ghost overlap reduces usable capacity through redundant learning.

## 15. Future work proposed by the authors

- Use encodings other than trilinear feature grids inside each adaptive grid, including NGP-style hash grids.
- Extend the adaptive grids into time for 4D temporal data.
- Reduce domain-decomposition seams through communication between networks during training.
- Optimize the multi-grid encoder and domain-decomposition inference.
- Extend to nonuniform grids, point data, unstructured data, and AMR if arbitrary scalar point queries are available.
- Integrate neural volume rendering into mature tools such as ParaView.

## 16. What belongs to the original paper and what does not

The original paper includes:

- Adaptive learnable feature-grid placement.
- The density loss and flat-top Gaussian approximation.
- Domain-decomposed independent brick training.
- Hard coordinate hashing to one brick at inference.
- Ghost-cell overlaps from one to sixteen cells.
- The nerfacc-based renderer.

The original paper does **not** include:

- Partition-of-unity blending between neighboring predictions.
- Smoothstep interface weights.
- Interface value or gradient consistency loss.
- Boundary-band PSNR or explicit routed value-jump metrics.
- The RTX 3050 Plume experiment in our project.

Those are part of the Seamless APMGSRN extension documented separately.

## 17. Short answers for likely questions

### Why not use one huge network?

The raw data or the required model may exceed GPU memory, training becomes slow, and out-of-core random sampling introduces substantial data-transfer overhead. Domain decomposition permits parallel in-core training of smaller models.

### Why use multiple movable grids instead of an octree?

The learned affine grids are directly differentiable and avoid explicit tree storage, traversal, pruning, and periodic structural refinement.

### Why is a density loss needed?

Reconstruction gradients only affect a grid when the queried point already lies inside it. The differentiable density approximation provides transformation gradients from regions outside the current support.

### Why does wide ghost overlap lower PSNR?

Each fixed-size brick network must fit more voxels, including redundant voxels learned by neighbors. Capacity per unique voxel decreases.

### Did the paper solve seams?

No. It reduced them using larger ghost overlap and identified network communication as future work.

### Was APMGSRN always best?

No. NGP performs better on the highly sparse Asteroid dataset, and conventional compressors usually provide better rate-distortion compression. APMGSRN's advantages include adaptive scientific-volume accuracy, random access, rendering in the compressed representation, and scalable domain decomposition.
