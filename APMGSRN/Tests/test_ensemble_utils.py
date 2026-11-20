import itertools
import torch
from Models.ensemble_utils import (parse_int_tuple, dhw_to_xyz, xyz_to_dhw, flat_brick_index,
    unflatten_brick_index, compute_core_extents_dhw, expand_extents_dhw, build_manifest,
    voxel_extents_dhw_to_normalized_xyz, enumerate_face_adjacencies, support_weight,
    blend_core_normalized_xyz, smoothstep01)

FULL = (512, 128, 128)   # Plume, DHW
GRID = (2, 2, 2)


def _old_start_jobs_extents(full_shape, ensemble_grid, extra_ghost):
    """Verbatim copy of the pre-change start_jobs.py geometry."""
    out = []
    x_step = full_shape[0] / ensemble_grid[0]
    y_step = full_shape[1] / ensemble_grid[1]
    z_step = full_shape[2] / ensemble_grid[2]
    ghost_cells = extra_ghost + 1
    for x_ind in range(ensemble_grid[0]):
        x_start = max(0, int(x_ind * x_step) - ghost_cells)
        x_end = int(full_shape[0]) if x_ind == ensemble_grid[0]-1 else int((x_ind+1) * x_step)
        x_end = min(full_shape[0], x_end+ghost_cells)
        for y_ind in range(ensemble_grid[1]):
            y_start = max(0, int(y_ind * y_step) - ghost_cells)
            y_end = int(full_shape[1]) if y_ind == ensemble_grid[1]-1 else int((y_ind+1) * y_step)
            y_end = min(full_shape[1], y_end+ghost_cells)
            for z_ind in range(ensemble_grid[2]):
                z_start = max(0, int(z_ind * z_step) - ghost_cells)
                z_end = int(full_shape[2]) if z_ind == ensemble_grid[2]-1 else int((z_ind+1) * z_step)
                z_end = min(full_shape[2], z_end+ghost_cells)
                out.append(f"{x_start},{x_end},{y_start},{y_end},{z_start},{z_end}")
    return out


def test_parse():
    assert parse_int_tuple("2, 2,3") == (2, 2, 3)
    assert parse_int_tuple([1, 2]) == (1, 2)


def test_unique_flat_indices():
    idx = [flat_brick_index(i, GRID) for i in itertools.product(range(2), repeat=3)]
    assert sorted(idx) == list(range(8))
    for f in range(8):
        assert flat_brick_index(unflatten_brick_index(f, GRID), GRID) == f


def test_twelve_face_pairs():
    pairs = enumerate_face_adjacencies(GRID)
    assert len(pairs) == 12
    assert len({(p["left"], p["right"]) for p in pairs}) == 12
    assert len(enumerate_face_adjacencies((3, 3, 3))) == 54


def test_core_partition_covers_every_voxel_once():
    for full, grid in [(FULL, GRID), ((10, 7, 13), (3, 2, 4))]:
        count = torch.zeros(full, dtype=torch.int32)
        for i in itertools.product(*[range(g) for g in grid]):
            e = compute_core_extents_dhw(full, grid, i)
            count[e[0]:e[1], e[2]:e[3], e[4]:e[5]] += 1
        assert bool((count == 1).all())


def test_expanded_extents_match_original_and_stay_inside():
    for full, grid in [(FULL, GRID), ((10, 7, 13), (3, 2, 4))]:
        for extra in (0, 3, 15):
            m = build_manifest(full, grid, extra)
            assert m["actual_ghost_cells"] == extra + 1
            ours = [b["subfolder"] for b in sorted(m["bricks"], key=lambda b: b["flat_index"])]
            assert ours == _old_start_jobs_extents(full, grid, extra)
            for b in m["bricks"]:
                s = b["support_extents_dhw"]
                for a in range(3):
                    assert 0 <= s[2*a] < s[2*a+1] <= full[a]


def test_axis_roundtrip():
    v = torch.randn(5, 3)
    assert torch.equal(xyz_to_dhw(dhw_to_xyz(v)), v)
    assert dhw_to_xyz([1, 2, 3]) == [3, 2, 1]


def test_normalized_matches_original_formula():
    ext = [0, 260, 60, 128, 0, 68]
    got = voxel_extents_dhw_to_normalized_xyz(ext, FULL)
    old = [float(i) for i in ext]
    old[0] = ((old[0] / (FULL[0]-1)) - 0.5) * 2
    old[1] = (((old[1]-1) / (FULL[0]-1)) - 0.5) * 2
    old[2] = ((old[2] / (FULL[1]-1)) - 0.5) * 2
    old[3] = (((old[3]-1) / (FULL[1]-1)) - 0.5) * 2
    old[4] = ((old[4] / (FULL[2]-1)) - 0.5) * 2
    old[5] = (((old[5]-1) / (FULL[2]-1)) - 0.5) * 2
    t = torch.tensor(old)
    assert torch.equal(got[0:3], t[0::2].flip(-1))
    assert torch.equal(got[3:6], t[1::2].flip(-1))


def _brick_weight_fn(flat, g=4):
    idx = unflatten_brick_index(flat, GRID)
    core = compute_core_extents_dhw(FULL, GRID, idx)
    sup = voxel_extents_dhw_to_normalized_xyz(expand_extents_dhw(core, FULL, g), FULL)
    bc = blend_core_normalized_xyz(core, FULL, g)
    return lambda p: support_weight(p, sup[:3], bc[:3], bc[3:], sup[3:])


def test_weight_one_in_core_zero_outside_support():
    w0 = _brick_weight_fn(0)
    assert torch.allclose(w0(torch.tensor([[-0.9, -0.9, -0.9]])), torch.ones(1))
    assert torch.allclose(w0(torch.tensor([[0.9, -0.9, -0.9]])), torch.zeros(1))


def test_weight_monotonic_and_finite_on_ramp():
    w0 = _brick_weight_fn(0)
    xs = torch.linspace(-1, 1, 4001)
    p = torch.stack([xs, torch.full_like(xs, -0.9), torch.full_like(xs, -0.9)], -1)
    w = w0(p)
    assert torch.isfinite(w).all()
    assert bool((w[1:] - w[:-1] <= 1e-7).all())   # non-increasing in +x for left brick
    assert w[0] == 1 and w[-1] == 0


def test_partition_of_unity_without_normalization():
    """Weights already sum to 1 everywhere because the ramps are symmetric smoothsteps."""
    torch.manual_seed(0)
    p = torch.rand(20000, 3) * 2 - 1
    total = sum(_brick_weight_fn(i)(p) for i in range(8))
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_smoothstep_endpoints():
    t = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    assert torch.allclose(smoothstep01(t), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
