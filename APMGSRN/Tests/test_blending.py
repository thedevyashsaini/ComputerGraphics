import torch
import torch.nn as nn
from Models.ensemble_SRN import Ensemble_SRN
from Models.ensemble_utils import (build_manifest, voxel_extents_dhw_to_normalized_xyz,
    enumerate_face_adjacencies)

FULL = [512, 128, 128]
GRID = "2,2,2"


class MockChild(nn.Module):
    """Analytic brick: evaluates fn(global_xyz) after undoing the local mapping."""
    def __init__(self, support_dhw, fn):
        super().__init__()
        e = voxel_extents_dhw_to_normalized_xyz(support_dhw, FULL)
        self.register_buffer("lo", e[:3])
        self.register_buffer("hi", e[3:])
        self.fn = fn

    def forward(self, x_local):
        g = self.lo + (x_local + 1) * 0.5 * (self.hi - self.lo)
        return self.fn(g)


def make_ensemble(fn_for_brick, extra_ghost=3, mode="hard"):
    m = build_manifest(FULL, (2, 2, 2), extra_ghost)
    bricks = sorted(m["bricks"], key=lambda b: b["flat_index"])
    models = [MockChild(b["support_extents_dhw"], fn_for_brick(b["flat_index"])) for b in bricks]
    opt = {"ensemble_grid": GRID, "full_shape": FULL, "ensemble_ghost_cells": extra_ghost,
           "ensemble_routing_mode": mode, "device": "cpu"}
    return Ensemble_SRN(opt, models=models,
                        support_extents_dhw=[b["support_extents_dhw"] for b in bricks],
                        core_extents_dhw=[b["core_extents_dhw"] for b in bricks])


def smooth_field(g):
    return (torch.sin(3 * g[:, 0:1]) + g[:, 1:2] * g[:, 2:3]).to(g.dtype)


def original_forward(ens, x):
    """Verbatim copy of the pre-change Ensemble_SRN.forward."""
    indices = (x+1)/2.0
    indices.clamp_(0.0, 0.99)
    indices *= ens.model_grid_shape
    indices = indices.long()
    indices = indices[:,0] + indices[:,1]*ens.model_grid_shape[0] + \
        indices[:,2]*(ens.model_grid_shape[0]*ens.model_grid_shape[1])
    y = torch.empty([x.shape[0], 1], device=x.device, dtype=x.dtype)
    for i in range(len(ens.models)):
        mask = (indices == i)
        x_i = x[mask]
        y[mask] = ens.models[i](-1+2*((x_i-ens.local_min_extents[i])/\
            (ens.local_max_extents[i]-ens.local_min_extents[i])))
    return y


def test_hard_router_unchanged():
    # Each brick returns a distinct offset so wrong routing is detected.
    ens = make_ensemble(lambda i: (lambda g: smooth_field(g) + 10 * i))
    torch.manual_seed(1)
    x = torch.rand(10000, 3) * 2 - 1
    a = ens.forward_hard(x.clone())
    b = original_forward(ens, x.clone())
    assert torch.allclose(a, b, atol=1e-7, rtol=0)
    assert torch.equal(ens(x.clone()), a)


def test_identical_fields_blend_equals_hard():
    ens = make_ensemble(lambda i: smooth_field)
    torch.manual_seed(2)
    x = torch.rand(20000, 3) * 2 - 1
    hard = ens.forward_hard(x.clone())
    ens.set_routing_mode("blend")
    blend = ens(x.clone())
    assert torch.allclose(hard, blend, atol=1e-5)


def _line_along_x(n=20001):
    xs = torch.linspace(-1, 1, n)
    return torch.stack([xs, torch.full_like(xs, -0.5), torch.full_like(xs, -0.5)], -1)


def test_constant_neighbours_transition_is_smooth_and_monotonic():
    # Bricks with W-index 1 (odd flat index) return 1, others 0. W axis == model x axis.
    ens = make_ensemble(lambda i: (lambda g: torch.full((g.shape[0], 1), float(i % 2))), mode="blend")
    y = ens(_line_along_x())[:, 0]
    assert y[0] == 0 and y[-1] == 1
    d = y[1:] - y[:-1]
    assert bool((d >= -1e-6).all())
    assert float(d.max()) < 0.05
    # Hard routing jumps by the full 1.0 at the split.
    ens.set_routing_mode("hard")
    yh = ens(_line_along_x())[:, 0]
    assert float((yh[1:] - yh[:-1]).max()) == 1.0


def test_weights_normalize_everywhere():
    ens = make_ensemble(lambda i: smooth_field, mode="blend")
    torch.manual_seed(3)
    x = torch.rand(50000, 3) * 2 - 1
    w = ens.model_support_weights(x)
    s = w.sum(0)
    assert bool((s > 0).all())
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_active_model_counts_face_edge_corner():
    ens = make_ensemble(lambda i: smooth_field, mode="blend")
    face = torch.tensor([[0.0, -0.5, -0.5]])
    edge = torch.tensor([[0.0, 0.0, -0.5]])
    corner = torch.tensor([[0.0, 0.0, 0.0]])
    count = lambda p: int((ens.model_support_weights(p) > 0).sum())
    assert count(face) == 2
    assert count(edge) == 4
    assert count(corner) == 8
    assert count(torch.tensor([[-0.9, -0.9, -0.9]])) == 1


def test_blend_gradients_finite():
    ens = make_ensemble(lambda i: (lambda g: smooth_field(g) + 0.1 * i), mode="blend")
    torch.manual_seed(4)
    x = (torch.rand(4096, 3) * 2 - 1).requires_grad_(True)
    y = ens(x)
    (gx,) = torch.autograd.grad(y.sum(), x)
    assert torch.isfinite(gx).all()


def test_g1_blend_runs_and_pairs():
    ens = make_ensemble(lambda i: smooth_field, extra_ghost=0, mode="blend")
    x = torch.rand(1000, 3) * 2 - 1
    assert torch.isfinite(ens(x)).all()
    assert len(enumerate_face_adjacencies((2, 2, 2))) == 12


def test_blend_fast_path_matches_full_blend():
    for extra_ghost in (0, 3, 15):
        ens = make_ensemble(lambda i: (lambda g: smooth_field(g) + 0.1 * i),
                            extra_ghost=extra_ghost, mode="blend")
        torch.manual_seed(7)
        x = torch.rand(20000, 3) * 2 - 1
        # Concentrate extra points tightly around every split plane and the volume edges.
        near = torch.rand(20000, 3) * 2 - 1
        axis = torch.randint(0, 3, (20000,))
        near[torch.arange(20000), axis] = (torch.rand(20000) - 0.5) * 0.2
        x = torch.cat([x, near, torch.tensor([[-1., -1., -1.], [1., 1., 1.], [0., 0., 0.]])])
        x = x.double()
        ens = ens.double()
        ens.blend_fast_path = True
        fast = ens(x)
        full = ens._forward_blend_full(x)
        assert torch.equal(fast, full) or torch.allclose(fast, full, atol=1e-12, rtol=0)
        xg = x[:4096].clone().requires_grad_(True)
        (g_fast,) = torch.autograd.grad(ens(xg).sum(), xg)
        xg2 = x[:4096].clone().requires_grad_(True)
        (g_full,) = torch.autograd.grad(ens._forward_blend_full(xg2).sum(), xg2)
        assert torch.allclose(g_fast, g_full, atol=1e-10, rtol=0)