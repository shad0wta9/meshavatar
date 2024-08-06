# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None      
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        wo = util.safe_normalize(view_pos - gb_pos)

        if specular:
            roughness = ks[..., 1:2] # y component
            metallic  = ks[..., 2:3] # z component
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic
            diff_col  = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse * diff_col

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_col += spec * reflectance

        return shaded_col * (1.0 - ks[..., 0:1]) # Modulate by hemisphere visibility
    
######################################################################################
# Monte-carlo sampled environment light with PDF / CDF computation
######################################################################################

class EnvironmentLightOptix(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    """
    envmap default definition:
    +-------+Y-------+
    |                |
    +Z  -X  -Z  +X   |
    |                |
    +------(-Y)------+
    """

    def __init__(self, base):
        super(EnvironmentLightOptix, self).__init__()
        self.mtx = None
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.register_parameter('env_base', self.base)

        self.pdf_scale = (self.base.shape[0] * self.base.shape[1]) / (2 * np.pi * np.pi)
        self.update_pdf()

    def xfm(self, mtx):
        self.mtx = mtx

    def parameters(self):
        return [self.base]

    def clone(self):
        return EnvironmentLightOptix(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def update_pdf(self):
        with torch.no_grad():
            # Compute PDF
            Y = util.pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
            self._pdf = torch.max(self.base, dim=-1)[0] * torch.sin(Y * np.pi) # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
            self._pdf = self._pdf / torch.sum(self._pdf)

            # Compute cumulative sums over the columns and rows
            self.cols = torch.cumsum(self._pdf, dim=1)
            self.rows = torch.cumsum(self.cols[:, -1:].repeat([1, self.cols.shape[1]]), dim=0)

            # Normalize
            self.cols = self.cols / torch.where(self.cols[:, -1:] > 0, self.cols[:, -1:], torch.ones_like(self.cols))
            self.rows = self.rows / torch.where(self.rows[-1:, :] > 0, self.rows[-1:, :], torch.ones_like(self.rows))

    def convert_for_zzr(self):
        """
        convert the envmap of zzr (+X for 'up') to our defintiion
        p.s. it is maybe incorrect
        """
        cubemap = util.latlong_to_cubemap(self.base, [512, 512])

        cubemap[[3, 2, 0, 1]] = cubemap[[0, 1, 2, 3]]
        cubemap[0] = torch.rot90(cubemap[0], k=3, dims=[0, 1])
        cubemap[1] = torch.rot90(cubemap[1], k=3, dims=[0, 1])
        cubemap[2] = torch.rot90(cubemap[2], k=3, dims=[0, 1])
        cubemap[3] = torch.rot90(cubemap[3], k=3, dims=[0, 1])
        cubemap[4] = torch.rot90(cubemap[4], k=3, dims=[0, 1])
        cubemap[5] = torch.rot90(cubemap[5], k=1, dims=[0, 1])
        latlong_img = util.cubemap_to_latlong(cubemap, res=self.base.shape)

        self.base.data = latlong_img
        self.update_pdf()

    def convert_for_synthetic(self):
        """
        convert the envmap defined on
        +-------+Z-------+
        |                |
        -X  -Y  +X  +Y   |
        |                |
        +------(-Z)------+
        to our default definition
        """

        ### for mitsuba
        cubemap = util.latlong_to_cubemap(self.base, [512, 512])
        cubemap = cubemap[[5, 4, 1, 0, 2, 3]]
        cubemap[0] = torch.rot90(cubemap[0], k=1, dims=[0, 1])
        cubemap[1] = torch.rot90(cubemap[1], k=3, dims=[0, 1])
        cubemap[2] = torch.rot90(cubemap[2], k=2, dims=[0, 1])
        cubemap[4] = torch.rot90(cubemap[4], k=3, dims=[0, 1])
        cubemap[5] = torch.rot90(cubemap[5], k=3, dims=[0, 1])
        latlong_img = util.cubemap_to_latlong(cubemap, res=self.base.shape)

        self.base.data = latlong_img
        self.update_pdf()

    @torch.no_grad()
    def generate_image(self, res):
        texcoord = util.pixel_grid(res[1], res[0])
        return dr.texture(self.base[None, ...].contiguous(), texcoord[None, ...].contiguous(), filter_mode='linear')[0]

######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0, usage='pbr', res=None):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    if usage == 'pbr':
        cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

        l = EnvironmentLight(cubemap)
        l.build_mips()

        return l
    else:
        if res is not None:
            texcoord = util.pixel_grid(res[1], res[0])
            latlong_img = torch.clamp(dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0], min=0.0001)

        print("EnvProbe,", latlong_img.shape, ", min/max", torch.min(latlong_img).item(), torch.max(latlong_img).item())
        return EnvironmentLightOptix(base=latlong_img)

def load_env(fn, scale=1.0, usage='pbr', res=None):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale, usage, res)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight) or isinstance(light, EnvironmentLightOptix), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    elif isinstance(light, EnvironmentLightOptix):
        color = light.generate_image([512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25, usage='pbr'):
    if usage == 'pbr':
        base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
        return EnvironmentLight(base)
    else:
        base = torch.rand(base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
        l = EnvironmentLightOptix(base)
        return l
      
