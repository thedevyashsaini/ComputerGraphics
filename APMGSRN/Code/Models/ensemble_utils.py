"""
Geometry helpers for domain-decomposed (ensemble) APMGSRN models.

Two axis orders exist in this repository:
  * DHW: data / storage order. NetCDF volumes load as [B, C, D, H, W] and
    `full_shape`, `ensemble_grid`, `extents` and `grid_index` all use DHW.
  * XYZ: model coordinate order. make_coord_grid() flips its output, so the
    coordinates fed to networks are [x, y, z] = [W, H, D] in [-1, 1].

Every conversion between the two lives in this file. Nothing here imports
datasets or renderers so it can be unit-tested on CPU in isolation.
"""
from typing import Dict, List, Sequence, Tuple
import torch

Extents = Tuple[int, int, int, int, int, int]


def parse_int_tuple(text) -> Tuple[int, ...]:
    """Parse '2,2,2' (or a list) into a tuple of ints without eval()."""
    if isinstance(text, (list, tuple)):
        return tuple(int(v) for v in text)
    return tuple(int(v.strip()) for v in str(text).split(",") if v.strip() != "")


def dhw_to_xyz(values):
    """Reverse the final dimension (DHW -> XYZ). Works for tensors and sequences."""
    if isinstance(values, torch.Tensor):
        return values.flip(-1)
    return type(values)(reversed(values)) if isinstance(values, tuple) else list(reversed(values))


def xyz_to_dhw(values):
    """Reverse the final dimension (XYZ -> DHW)."""
    return dhw_to_xyz(values)


def dhw_axis_to_xyz_axis(axis_dhw: int) -> int:
    return 2 - axis_dhw


def flat_brick_index(index_dhw: Sequence[int], grid_dhw: Sequence[int]) -> int:
    """Flattening order used by Ensemble_SRN (W fastest, D slowest)."""
    return int(index_dhw[2] + index_dhw[1] * grid_dhw[2] + index_dhw[0] * grid_dhw[2] * grid_dhw[1])


def unflatten_brick_index(flat: int, grid_dhw: Sequence[int]) -> Tuple[int, int, int]:
    w = flat % grid_dhw[2]
    h = (flat // grid_dhw[2]) % grid_dhw[1]
    d = flat // (grid_dhw[2] * grid_dhw[1])
    return (int(d), int(h), int(w))


def _core_interval(full_size: int, brick_count: int, brick_index: int) -> Tuple[int, int]:
    # Must match start_jobs.py exactly (int() truncation, last brick runs to the end).
    step = full_size / brick_count
    start = int(brick_index * step)
    end = int(full_size) if brick_index == brick_count - 1 else int((brick_index + 1) * step)
    return start, end


def compute_core_extents_dhw(full_shape_dhw, ensemble_grid_dhw, index_dhw) -> Extents:
    """[d0, d1, h0, h1, w0, w1], end-exclusive voxel extents without ghost cells."""
    out = []
    for a in range(3):
        s, e = _core_interval(int(full_shape_dhw[a]), int(ensemble_grid_dhw[a]), int(index_dhw[a]))
        out += [s, e]
    return tuple(out)


def expand_extents_dhw(core_extents_dhw, full_shape_dhw, actual_ghost_cells: int) -> Extents:
    """Grow core extents by the ghost cells, clipped to the volume (start_jobs.py logic)."""
    out = []
    for a in range(3):
        s = max(0, int(core_extents_dhw[2 * a]) - actual_ghost_cells)
        e = min(int(full_shape_dhw[a]), int(core_extents_dhw[2 * a + 1]) + actual_ghost_cells)
        out += [s, e]
    return tuple(out)


def extents_to_string(extents_dhw) -> str:
    return ",".join(str(int(v)) for v in extents_dhw)


def voxel_extents_dhw_to_normalized_xyz(extents_dhw, full_shape_dhw) -> torch.Tensor:
    """End-exclusive voxel extents -> [xmin, ymin, zmin, xmax, ymax, zmax] (align_corners=True).

    The arithmetic mirrors the original Ensemble_SRN code so float32 results are bit-identical.
    """
    mins_dhw, maxs_dhw = [], []
    for a in range(3):
        n = full_shape_dhw[a]
        mins_dhw.append(((extents_dhw[2 * a] / (n - 1)) - 0.5) * 2)
        maxs_dhw.append((((extents_dhw[2 * a + 1] - 1) / (n - 1)) - 0.5) * 2)
    return torch.tensor(list(reversed(mins_dhw)) + list(reversed(maxs_dhw)), dtype=torch.float32)


def voxel_to_normalized(voxel, full_size: int):
    """Continuous voxel coordinate -> normalized coordinate (align_corners=True)."""
    return -1.0 + 2.0 * voxel / (full_size - 1)


def normalized_to_voxel(coord, full_size: int):
    return (coord + 1.0) * 0.5 * (full_size - 1)


def hard_split_normalized(split_index: int, brick_count: int) -> float:
    """Normalized coordinate where the original hard router switches bricks.

    Ensemble_SRN.forward_hard uses floor((x+1)/2 * n), so split k (1..n-1) is at -1 + 2k/n.
    """
    return -1.0 + 2.0 * split_index / brick_count


def build_brick_metadata(full_shape_dhw, ensemble_grid_dhw, actual_ghost_cells: int) -> List[Dict]:
    bricks = []
    for d in range(ensemble_grid_dhw[0]):
        for h in range(ensemble_grid_dhw[1]):
            for w in range(ensemble_grid_dhw[2]):
                idx = (d, h, w)
                core = compute_core_extents_dhw(full_shape_dhw, ensemble_grid_dhw, idx)
                support = expand_extents_dhw(core, full_shape_dhw, actual_ghost_cells)
                bricks.append({
                    "flat_index": flat_brick_index(idx, ensemble_grid_dhw),
                    "index_dhw": list(idx),
                    "core_extents_dhw": list(core),
                    "support_extents_dhw": list(support),
                    "subfolder": extents_to_string(support),
                })
    bricks.sort(key=lambda b: b["flat_index"])
    return bricks


def build_manifest(full_shape_dhw, ensemble_grid_dhw, configured_extra_ghost_cells: int) -> Dict:
    actual = int(configured_extra_ghost_cells) + 1
    return {
        "full_shape_dhw": [int(v) for v in full_shape_dhw],
        "ensemble_grid_dhw": [int(v) for v in ensemble_grid_dhw],
        "configured_extra_ghost_cells": int(configured_extra_ghost_cells),
        "actual_ghost_cells": actual,
        "axis_order_note": "extents are DHW voxel indices, end-exclusive; model coordinates are XYZ",
        "bricks": build_brick_metadata(full_shape_dhw, ensemble_grid_dhw, actual),
    }


def enumerate_face_adjacencies(grid_dhw) -> List[Dict]:
    """Each face-adjacent brick pair once. 'left' is the lower index along the normal."""
    pairs = []
    for d in range(grid_dhw[0]):
        for h in range(grid_dhw[1]):
            for w in range(grid_dhw[2]):
                idx = (d, h, w)
                for axis in range(3):
                    if idx[axis] + 1 < grid_dhw[axis]:
                        nbr = list(idx)
                        nbr[axis] += 1
                        pairs.append({
                            "left": flat_brick_index(idx, grid_dhw),
                            "right": flat_brick_index(nbr, grid_dhw),
                            "left_index_dhw": list(idx),
                            "right_index_dhw": nbr,
                            "normal_axis_dhw": axis,
                            "normal_axis_xyz": dhw_axis_to_xyz_axis(axis),
                            "split_index": idx[axis] + 1,
                        })
    return pairs


def intersect_extents_dhw(a, b) -> Extents:
    out = []
    for i in range(3):
        out += [max(a[2 * i], b[2 * i]), min(a[2 * i + 1], b[2 * i + 1])]
    return tuple(out)


def smoothstep01(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0, 1)
    return t.square() * (3 - 2 * t)


def support_weight(points_xyz, support_min_xyz, core_min_xyz, core_max_xyz, support_max_xyz,
                   eps: float = 1e-12) -> torch.Tensor:
    """Separable C1 blending weight, one per point.

    Per axis: 0 outside the support, smoothstep ramp from support edge to core edge,
    1 inside the core. Volume-edge bricks have support == core on the outer side,
    where the ramp degenerates to a step at the volume boundary (no NaNs).
    """
    q = points_xyz
    left_den = core_min_xyz - support_min_xyz
    right_den = support_max_xyz - core_max_xyz
    ones = torch.ones_like(q)
    t_left = torch.where(left_den > 0, (q - support_min_xyz) / left_den.clamp_min(eps), ones)
    t_right = torch.where(right_den > 0, (support_max_xyz - q) / right_den.clamp_min(eps), ones)
    w = smoothstep01(t_left) * smoothstep01(t_right)
    inside = (q >= support_min_xyz) & (q <= support_max_xyz)
    w = w * inside.to(w.dtype)
    return w.prod(dim=-1)


def compute_blend_core_voxels_dhw(core_extents_dhw, full_shape_dhw, actual_ghost_cells: int):
    """Inclusive voxel interval per axis where a brick owns the field alone (weight 1).

    Between two neighbors sharing split s (left core ends at s, right core starts at s),
    the overlap is voxels [s - g, s + g - 1]. Both neighbors ramp across this whole
    interval in opposite directions, so smoothstep(t) + smoothstep(1 - t) = 1 gives an
    exact partition of unity centered on the original hard split (s - 0.5).
    Returns [d_lo, d_hi, h_lo, h_hi, w_lo, w_hi] as continuous voxel coordinates.
    """
    g = int(actual_ghost_cells)
    out = []
    for a in range(3):
        n = int(full_shape_dhw[a])
        s, e = int(core_extents_dhw[2 * a]), int(core_extents_dhw[2 * a + 1])
        lo = 0 if s == 0 else min(s + g - 1, n - 1)
        hi = n - 1 if e >= n else max(e - g, 0)
        out += [lo, hi]
    return tuple(out)


def blend_core_normalized_xyz(core_extents_dhw, full_shape_dhw, actual_ghost_cells: int) -> torch.Tensor:
    """[xmin, ymin, zmin, xmax, ymax, zmax] of the weight-1 region in normalized coordinates."""
    v = compute_blend_core_voxels_dhw(core_extents_dhw, full_shape_dhw, actual_ghost_cells)
    mins = [voxel_to_normalized(v[2 * a], full_shape_dhw[a]) for a in range(3)]
    maxs = [voxel_to_normalized(v[2 * a + 1], full_shape_dhw[a]) for a in range(3)]
    return torch.tensor(list(reversed(mins)) + list(reversed(maxs)), dtype=torch.float32)
