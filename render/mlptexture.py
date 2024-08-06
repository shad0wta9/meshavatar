# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import tinycudann as tcnn
import numpy as np

from .embedder import get_embedder

#######################################################################################################################################################
# Small MLP using PyTorch primitives, internal helper class
#######################################################################################################################################################

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0, bias=False):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=bias), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=bias), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=bias),)
        self.net = torch.nn.Sequential(*net).cuda()
        
        self.net.apply(self._init_weights)
        
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

#######################################################################################################################################################
# Outward visible MLP class
#######################################################################################################################################################

class MLPTexture3D(torch.nn.Module):
    def __init__(self, AABB, channels = 3, internal_dims = 256, hidden = 2, min_max = None, 
                 cond_dim = 0, encoding = 'hash', bias = False):
        super(MLPTexture3D, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max
        self.cond_dim = cond_dim

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        if encoding == 'hash':
            desired_resolution = 4096
            base_grid_resolution = 16
            num_levels = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

            enc_cfg =  {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_grid_resolution,
                "per_level_scale" : per_level_scale
            }
        elif encoding == 'sin':
            num_freqs = 4

            enc_cfg = {
                "otype": "Frequency",
                "n_frequencies": num_freqs
            }

        gradient_scaling = 128.0 if cond_dim == 0 else 1.0
        self.encoder = tcnn.Encoding(3, enc_cfg)
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims + self.cond_dim,
            "n_output_dims" : self.channels,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling, bias=bias)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))

    def get_output(self, texc, cond):
        p_enc = self.encoder(texc.contiguous())
        if self.cond_dim > 0 and cond is not None:
            # p_enc [B, N, C] cond [B, C']
            p_enc = torch.cat([p_enc, cond], dim=-1)
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out

    # Sample texture at a given location
    def sample(self, texc, cond=None, msk=None):
        if self.cond_dim > 0 and cond is not None:
            if isinstance(cond, torch.Tensor):
                _cond = cond[:, None, None].expand(-1, texc.shape[1], texc.shape[2], -1).reshape(-1, self.cond_dim)
            else:
                _cond = cond(texc.reshape(texc.shape[0], -1, 3)).reshape(-1, self.cond_dim)
        else:
            _cond = None
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)

        if msk is not None:
            _msk = msk.reshape(-1)
            out = torch.zeros(_texc.shape[0], self.channels).to(_texc.device)
            out[_msk] = self.get_output(_texc[_msk], _cond[_msk] if _cond is not None else _cond)
        else:
            out = self.get_output(_texc, _cond)

        out = out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]

        return out

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()

class MLPRadiance3D(MLPTexture3D):
    def __init__(self, AABB, channels = 3, internal_dims = 256, hidden = 2, min_max = None, 
                 cond_dim = 0, encoding = 'hash', bias = False):
        
        view_embedder, view_ebddim = get_embedder(4, input_dims=3, include_input=True)
        super().__init__(AABB, channels, internal_dims, hidden, min_max, 
                         cond_dim+view_ebddim, encoding, bias)
        self.view_embedder, self.view_ebddim = view_embedder, view_ebddim
        self.cond_dim = cond_dim

    def get_output(self, texc, view_dir, cond):
        p_enc = self.encoder(texc.contiguous())
        v_enc = self.view_embedder(view_dir)
        if self.cond_dim > 0 and cond is not None:
            # p_enc [B, N, C] cond [B, C']
            p_enc = torch.cat([p_enc, cond], dim=-1)

        p_enc = torch.cat([p_enc, v_enc], dim=-1)
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out

    # Sample texture at a given location
    def sample(self, texc, view_dir, cond=None, msk=None):
        if cond is not None:
            if isinstance(cond, torch.Tensor):
                _cond = cond[:, None, None].expand(-1, texc.shape[1], texc.shape[2], -1).reshape(-1, self.cond_dim)
            else:
                _cond = cond(texc.reshape(texc.shape[0], -1, 3)).reshape(-1, self.cond_dim)
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        _view = view_dir.view(-1, 3)

        if msk is not None:
            _msk = msk.reshape(-1)
            out = torch.zeros(_texc.shape[0], self.channels).to(_texc.device)
            out[_msk] = self.get_output(_texc[_msk], _view[_msk], _cond[_msk])
        else:
            out = self.get_output(_texc, _view, _cond)

        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]


class FeatMapRadiance(torch.nn.Module):
    """
    seems out of date
    it's unused in the repo
    """
    def __init__(self, AABB, channels = 3, internal_dims = 256, hidden = 8, min_max = None, 
                 cond_dim = 0, encoding = 'hash', bias = False):
        super().__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max
        self.cond_dim = cond_dim

    def get_output(self, texc, view_dir, cond):
        msk = texc[..., [-1]] > 0
        out = msk * cond[..., :self.channels] + (~msk) * cond[..., self.channels:2*self.channels]

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        return out

    def sample(self, texc, view_dir, cond=None, msk=None):
        if cond is not None:
            if isinstance(cond, torch.Tensor):
                _cond = cond[:, None, None].expand(-1, texc.shape[1], texc.shape[2], -1).reshape(-1, self.cond_dim)
            else:
                _cond = cond(texc.reshape(texc.shape[0], -1, 3)).reshape(-1, self.cond_dim)
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        _view = view_dir.view(-1, 3)

        if msk is not None:
            _msk = msk.reshape(-1)
            out = torch.zeros(_texc.shape[0], self.channels).to(_texc.device)
            out[_msk] = self.get_output(_texc[_msk], _view[_msk], _cond[_msk])
        else:
            out = self.get_output(_texc, _view, _cond)

        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]

    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()