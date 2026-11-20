import torch
import torch.nn as nn
import os
import json
from Models.options import load_options
from Models.models import load_model
from Models.ensemble_utils import (parse_int_tuple, flat_brick_index,
    compute_core_extents_dhw, voxel_extents_dhw_to_normalized_xyz,
    blend_core_normalized_xyz, support_weight, build_manifest)

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

ROUTING_MODES = ("hard", "blend")

class Ensemble_SRN(nn.Module):
    '''
    This class is purely for inference (not training). Our
    start_jobs script will ensure that the ensemble models
    are trained disjointly, but to query them efficiently
    after training, this ensemble model "wraps" the 
    many smaller models for inference/visualization tasks.

    Routing modes (opt['ensemble_routing_mode']):
      hard  - original APMGSRN behaviour: each point is answered by exactly one brick.
      blend - Seamless APMGSRN: points in a ghost-cell overlap are answered by a
              C1 partition-of-unity blend of every brick whose support contains them.
    '''
    def __init__(self, opt, models=None, support_extents_dhw=None, core_extents_dhw=None):
        super().__init__()
        self.opt = opt
        ensemble_grid = list(parse_int_tuple(opt['ensemble_grid']))
        full_shape = list(opt['full_shape'])
        self.routing_mode = opt.get('ensemble_routing_mode', 'hard')
        self.blend_epsilon = float(opt.get('ensemble_blend_epsilon', 1e-8))
        self.blend_fast_path = bool(opt.get('ensemble_blend_fast_path', False))
        self.actual_ghost_cells = int(opt.get('ensemble_ghost_cells', 0)) + 1
        self.submodel_folders = []
        self.submodel_opts = []

        if models is None:
            models, support_extents_dhw, core_extents_dhw = self._load_submodels(
                opt, ensemble_grid, full_shape)

        local_extents = torch.stack([
            voxel_extents_dhw_to_normalized_xyz(e, full_shape) for e in support_extents_dhw])
        blend_cores = torch.stack([
            blend_core_normalized_xyz(c, full_shape, self.actual_ghost_cells) for c in core_extents_dhw])
        self.support_extents_dhw = [list(e) for e in support_extents_dhw]
        self.core_extents_dhw = [list(c) for c in core_extents_dhw]

        self.register_buffer("model_grid_shape",
            torch.tensor(ensemble_grid, dtype=torch.long).flip(0),
            persistent=False)
        self.register_buffer("full_data_shape",
            torch.tensor(full_shape, dtype=torch.long),
            persistent=False)
        # Support (= training) extents of each brick, normalized XYZ.
        self.register_buffer("local_min_extents", local_extents[:, 0:3].clone(), persistent=False)
        self.register_buffer("local_max_extents", local_extents[:, 3:6].clone(), persistent=False)
        # Region where a brick has blend weight exactly 1, normalized XYZ.
        self.register_buffer("core_min_extents", blend_cores[:, 0:3].clone(), persistent=False)
        self.register_buffer("core_max_extents", blend_cores[:, 3:6].clone(), persistent=False)
        self.models = torch.nn.ModuleList(models)
        self.set_routing_mode(self.routing_mode)

    def _load_submodels(self, opt, ensemble_grid, full_shape):
        unsorted = {}
        root = os.path.join(save_folder, opt['save_name'])
        model_folders = sorted(os.listdir(root))
        for submodel in model_folders:
            if(os.path.isdir(os.path.join(root, submodel))):
                sub_opt = load_options(os.path.join(root, submodel))
                sub_opt['device'] = opt['device']
                submodel_model = load_model(sub_opt, opt['device'])
                support = list(parse_int_tuple(sub_opt['extents']))
                indices = list(parse_int_tuple(sub_opt['grid_index']))
                ind = flat_brick_index(indices, ensemble_grid)
                # Core extents are recomputed from the decomposition (works for old checkpoints).
                core = list(compute_core_extents_dhw(full_shape, ensemble_grid, indices))
                unsorted[ind] = [submodel_model, support, core, submodel, sub_opt]

        models, supports, cores = [], [], []
        n_models = ensemble_grid[0]*ensemble_grid[1]*ensemble_grid[2]
        for i in range(n_models):
            if i not in unsorted:
                raise RuntimeError(f"Ensemble {opt['save_name']} is missing brick {i}")
            m, s, c, folder, sub_opt = unsorted[i]
            models.append(m)
            supports.append(s)
            cores.append(c)
            self.submodel_folders.append(folder)
            self.submodel_opts.append(sub_opt)
        return models, supports, cores

    def set_routing_mode(self, mode):
        if mode not in ROUTING_MODES:
            raise ValueError(f"Unknown ensemble_routing_mode '{mode}', expected one of {ROUTING_MODES}")
        self.routing_mode = mode
        self.opt['ensemble_routing_mode'] = mode

    def manifest(self):
        return build_manifest(self.opt['full_shape'],
            parse_int_tuple(self.opt['ensemble_grid']),
            int(self.opt.get('ensemble_ghost_cells', 0)))

    def min(self):
        val = self.models[0].min()
        for i in range(1, len(self.models)):
            val = torch.min(val, self.models[i].min())
        return val

    def max(self):
        val = self.models[0].max()
        for i in range(1, len(self.models)):
            val = torch.max(val, self.models[i].max())
        return val
    
    def get_volume_extents(self):
        return self.opt['full_shape']

    def global_to_local(self, x_xyz, model_index: int):
        return -1+2*((x_xyz-self.local_min_extents[model_index])/\
            (self.local_max_extents[model_index]-self.local_min_extents[model_index]))

    def forward_child(self, x_xyz, model_index: int):
        """Query one brick at global coordinates (used by fine-tuning and metrics)."""
        return self.models[model_index](self.global_to_local(x_xyz, model_index))

    def hard_indices(self, x):
        # divide by slightly larger than 2
        # to avoid the result equaling 1
        indices = (x+1)/2.0
        indices.clamp_(0.0, 0.99)
        indices *= self.model_grid_shape     
        indices = indices.long()
        
        indices = indices[:,0] + indices[:,1]*self.model_grid_shape[0] + \
            indices[:,2]*(self.model_grid_shape[0]*self.model_grid_shape[1])
        return indices

    def forward_hard(self, x):
        indices = self.hard_indices(x)
        y = torch.empty([x.shape[0], 1], 
            device=x.device, dtype=x.dtype)

        for i in range(len(self.models)):
            mask = (indices == i)
            x_i = x[mask]
            y[mask] = self.models[i](self.global_to_local(x_i, i))
        return y

    def model_support_weights(self, x_xyz):
        """[number_of_models, batch] unnormalized blend weights, all bricks in one broadcast."""
        return support_weight(x_xyz.unsqueeze(0),
                              self.local_min_extents.unsqueeze(1), self.core_min_extents.unsqueeze(1),
                              self.core_max_extents.unsqueeze(1), self.local_max_extents.unsqueeze(1))

    def forward_blend(self, x):
        # Optional core/overlap split (exact): inside its owner brick's blend core a point has
        # weight 1 for that brick only, so blending reduces to hard routing there. Off by default:
        # the vectorized full blend measured faster on an RTX 3050 (15.4 vs 13.7 Mpts/s, g4).
        if not self.blend_fast_path:
            return self._forward_blend_full(x)
        idx = self.hard_indices(x)
        in_core = ((x >= self.core_min_extents[idx]) & (x <= self.core_max_extents[idx])).all(dim=-1)
        if bool(in_core.all()):
            return self.forward_hard(x)
        result = torch.empty([x.shape[0], 1], device=x.device, dtype=x.dtype)
        if bool(in_core.any()):
            result = result.index_put((in_core,), self.forward_hard(x[in_core]))
        slab = ~in_core
        return result.index_put((slab,), self._forward_blend_full(x[slab]))

    def _forward_blend_full(self, x):
        weights = self.model_support_weights(x).to(x.dtype)          # [M, N]
        active = weights > 0
        numerator = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        for i in range(len(self.models)):
            idx = active[i].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            pred = self.models[i](self.global_to_local(x[idx], i)).to(x.dtype)
            numerator = numerator.index_add(0, idx, weights[i, idx].unsqueeze(-1) * pred)
        denominator = weights.sum(0).unsqueeze(-1)
        result = numerator / denominator.clamp_min(self.blend_epsilon)
        missing = denominator[:, 0] <= self.blend_epsilon
        if bool(missing.any()):
            result = result.index_put((missing,), self.forward_hard(x[missing]))
        return result

    def forward(self, x):
        if self.routing_mode == 'hard':
            return self.forward_hard(x)
        if self.routing_mode == 'blend':
            return self.forward_blend(x)
        raise ValueError(f"Unknown ensemble_routing_mode {self.routing_mode}")
