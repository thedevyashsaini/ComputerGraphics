| Ours / candidate | Baseline | Metric | Mean diff | Std | Per-seed diffs | Wins |
|---|---|---|---|---|---|---|
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) | global_psnr_db | +2.004 dB | 0.124 | +2.09, +1.86, +2.06 | 3/3 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) | boundary_psnr_db | +2.513 dB | 0.218 | +2.76, +2.36, +2.41 | 3/3 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) | render_psnr_db | +2.075 dB | 0.402 | +2.40, +1.63, +2.19 | 3/3 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) | composite_value_jump_mean | 579x lower | 97.2 | 682x, 489x, 567x | 3/3 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) | composite_gradient_jump_mean | 1.38x lower | 0.0466 | 1.41x, 1.40x, 1.33x | 3/3 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [27^3 grids] | global_psnr_db | +0.968 dB | - | +0.97 | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [27^3 grids] | boundary_psnr_db | +1.333 dB | - | +1.33 | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [27^3 grids] | render_psnr_db | +0.933 dB | - | +0.93 | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [27^3 grids] | composite_value_jump_mean | 478x lower | - | 478x | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [27^3 grids] | composite_gradient_jump_mean | 1.38x lower | - | 1.38x | 1/1 |
| g4 consistency-FT + blend (OURS) | g4 hard | global_psnr_db | +0.506 dB | 0.360 | +0.31, +0.92, +0.28 | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 hard | boundary_psnr_db | +1.437 dB | 0.380 | +1.21, +1.88, +1.23 | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 hard | render_psnr_db | +1.099 dB | 0.996 | +0.52, +2.25, +0.53 | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 hard | composite_value_jump_mean | 474x lower | 65.5 | 442x, 549x, 431x | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 hard | composite_gradient_jump_mean | 1.18x lower | 0.00412 | 1.18x, 1.18x, 1.17x | 3/3 |
| g4 consistency-FT + blend (OURS) | g1 hard (paper, min overlap) | global_psnr_db | -0.388 dB | - | -0.39 | 0/1 |
| g4 consistency-FT + blend (OURS) | g1 hard (paper, min overlap) | boundary_psnr_db | +0.492 dB | - | +0.49 | 1/1 |
| g4 consistency-FT + blend (OURS) | g1 hard (paper, min overlap) | render_psnr_db | +0.052 dB | - | +0.05 | 1/1 |
| g4 consistency-FT + blend (OURS) | g1 hard (paper, min overlap) | composite_value_jump_mean | 459x lower | - | 459x | 1/1 |
| g4 consistency-FT + blend (OURS) | g1 hard (paper, min overlap) | composite_gradient_jump_mean | 1.31x lower | - | 1.31x | 1/1 |
| g4 blend (no training) | g4 hard | global_psnr_db | +0.195 dB | 0.030 | +0.17, +0.23, +0.18 | 3/3 |
| g4 blend (no training) | g4 hard | boundary_psnr_db | +1.243 dB | 0.181 | +1.12, +1.45, +1.16 | 3/3 |
| g4 blend (no training) | g4 hard | render_psnr_db | +0.187 dB | 0.031 | +0.17, +0.22, +0.16 | 3/3 |
| g4 blend (no training) | g4 hard | composite_value_jump_mean | 463x lower | 58.9 | 436x, 531x, 423x | 3/3 |
| g4 blend (no training) | g4 hard | composite_gradient_jump_mean | 1.16x lower | 0.00209 | 1.16x, 1.16x, 1.16x | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 blend (no training) | global_psnr_db | +0.311 dB | 0.331 | +0.14, +0.69, +0.10 | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 blend (no training) | boundary_psnr_db | +0.194 dB | 0.201 | +0.09, +0.43, +0.07 | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 blend (no training) | render_psnr_db | +0.912 dB | 0.965 | +0.35, +2.03, +0.36 | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 blend (no training) | composite_value_jump_mean | 1.02x lower | 0.0113 | 1.01x, 1.04x, 1.02x | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 blend (no training) | composite_gradient_jump_mean | 1.01x lower | 0.00179 | 1.01x, 1.02x, 1.01x | 3/3 |
| g4 consistency-FT + blend (OURS) | g4 recon-FT + blend (OURS, best variant) | global_psnr_db | -0.206 dB | 0.044 | -0.17, -0.24 | 0/2 |
| g4 consistency-FT + blend (OURS) | g4 recon-FT + blend (OURS, best variant) | boundary_psnr_db | -0.379 dB | 0.022 | -0.36, -0.39 | 0/2 |
| g4 consistency-FT + blend (OURS) | g4 recon-FT + blend (OURS, best variant) | render_psnr_db | -0.250 dB | 0.050 | -0.21, -0.29 | 0/2 |
| g4 consistency-FT + blend (OURS) | g4 recon-FT + blend (OURS, best variant) | composite_value_jump_mean | 1.01x lower | 0.000836 | 1.01x, 1.01x | 2/2 |
| g4 consistency-FT + blend (OURS) | g4 recon-FT + blend (OURS, best variant) | composite_gradient_jump_mean | 1.01x lower | 0.00224 | 1.02x, 1.01x | 2/2 |
| g4 consistency-FT + blend (OURS) | g16 + recon-FT (paper + equal extra training) | global_psnr_db | +1.495 dB | - | +1.49 | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 + recon-FT (paper + equal extra training) | boundary_psnr_db | +1.773 dB | - | +1.77 | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 + recon-FT (paper + equal extra training) | render_psnr_db | +1.077 dB | - | +1.08 | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 + recon-FT (paper + equal extra training) | composite_value_jump_mean | 445x lower | - | 445x | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 + recon-FT (paper + equal extra training) | composite_gradient_jump_mean | 1.32x lower | - | 1.32x | 1/1 |
| g4 recon-FT + blend (OURS, best variant) | g16 + recon-FT (paper + equal extra training) | global_psnr_db | +1.731 dB | - | +1.73 | 1/1 |
| g4 recon-FT + blend (OURS, best variant) | g16 + recon-FT (paper + equal extra training) | boundary_psnr_db | +2.168 dB | - | +2.17 | 1/1 |
| g4 recon-FT + blend (OURS, best variant) | g16 + recon-FT (paper + equal extra training) | render_psnr_db | +1.362 dB | - | +1.36 | 1/1 |
| g4 recon-FT + blend (OURS, best variant) | g16 + recon-FT (paper + equal extra training) | composite_value_jump_mean | 442x lower | - | 442x | 1/1 |
| g4 recon-FT + blend (OURS, best variant) | g16 + recon-FT (paper + equal extra training) | composite_gradient_jump_mean | 1.31x lower | - | 1.31x | 1/1 |
| g4 blend (no training) | g16 hard (paper, wide overlap) | global_psnr_db | +1.693 dB | 0.454 | +1.95, +1.17, +1.96 | 3/3 |
| g4 blend (no training) | g16 hard (paper, wide overlap) | boundary_psnr_db | +2.319 dB | 0.369 | +2.67, +1.94, +2.34 | 3/3 |
| g4 blend (no training) | g16 hard (paper, wide overlap) | render_psnr_db | +1.163 dB | 1.358 | +2.06, -0.40, +1.83 | 2/3 |
| g4 blend (no training) | g16 hard (paper, wide overlap) | composite_value_jump_mean | 567x lower | 101 | 673x, 472x, 556x | 3/3 |
| g4 blend (no training) | g16 hard (paper, wide overlap) | composite_gradient_jump_mean | 1.36x lower | 0.0454 | 1.40x, 1.38x, 1.31x | 3/3 |
| g4 recon-FT, hard (control) | g4 hard | global_psnr_db | +0.659 dB | 0.420 | +0.96, +0.36 | 2/2 |
| g4 recon-FT, hard (control) | g4 hard | boundary_psnr_db | +0.957 dB | 0.513 | +1.32, +0.59 | 2/2 |
| g4 recon-FT, hard (control) | g4 hard | render_psnr_db | +1.473 dB | 1.152 | +2.29, +0.66 | 2/2 |
| g4 recon-FT, hard (control) | g4 hard | composite_value_jump_mean | 1.39x lower | 0.257 | 1.57x, 1.21x | 2/2 |
| g4 recon-FT, hard (control) | g4 hard | composite_gradient_jump_mean | 1x lower | 0.000226 | 1.00x, 1.00x | 2/2 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [32^3 grids] | global_psnr_db | -0.092 dB | - | -0.09 | 0/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [32^3 grids] | boundary_psnr_db | +0.404 dB | - | +0.40 | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [32^3 grids] | render_psnr_db | -0.411 dB | - | -0.41 | 0/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [32^3 grids] | composite_value_jump_mean | 444x lower | - | 444x | 1/1 |
| g4 consistency-FT + blend (OURS) | g16 hard (paper, wide overlap) [32^3 grids] | composite_gradient_jump_mean | 1.36x lower | - | 1.36x | 1/1 |
