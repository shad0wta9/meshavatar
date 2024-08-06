# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from . import optixutils as ou
from . import light

rnd_seed = 0

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        v_pos_clip,
        rast,
        rast_deriv,
        mesh,
        skinning_weights,
        resolution,
        spp,
        msaa,
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    # import ipdb; ipdb.set_trace()
    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos, rast_out_s, mesh.t_pos_idx[0].int())
    gb_weights, _ = interpolate(skinning_weights, rast_out_s, mesh.t_pos_idx[0].int())
    msk = rast_out_s[..., -1] > 0
    # if cano_pos is not None:
    #     cano_pos, _ = interpolate(cano_pos, rast_out_s, mesh.t_pos_idx[0].int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[:, mesh.t_pos_idx[0, :, 0], :]
    v1 = mesh.v_pos[:, mesh.t_pos_idx[0, :, 1], :]
    v2 = mesh.v_pos[:, mesh.t_pos_idx[0, :, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0, dim=-1))
    num_faces = face_normals.shape[1]
    face_normal_indices = (torch.arange(0, num_faces, dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals, rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm, rast_out_s, mesh.t_nrm_idx[0].int())
    gb_tangent, _ = interpolate(mesh.v_tng, rast_out_s, mesh.t_tng_idx[0].int()) # Interpolate tangents

    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex, rast_out_s, mesh.t_tex_idx[0].int(), rast_db=rast_out_deriv_s)

    # Interpolate z and z-gradient
    with torch.no_grad():
        eps = 0.00001
        clip_pos, clip_pos_deriv = interpolate(v_pos_clip, rast_out_s, mesh.t_pos_idx[0].int(), rast_db=rast_out_deriv_s)
        z0 = torch.clamp(clip_pos[..., 2:3], min=eps) / torch.clamp(clip_pos[..., 3:4], min=eps)
        z1 = torch.clamp(clip_pos[..., 2:3] + torch.abs(clip_pos_deriv[..., 2:3]), min=eps) / torch.clamp(clip_pos[..., 3:4] + torch.abs(clip_pos_deriv[..., 3:4]), min=eps)
        z_grad = torch.abs(z1 - z0)
        gb_depth = torch.cat((z0, z_grad), dim=-1)

    ################################################################################
    # Shade
    ################################################################################

    alpha = torch.ones_like(gb_pos[..., 0:1])
    buffers = {
        "gb_pos": torch.cat([gb_pos, alpha], dim=-1),
        "gb_normal": torch.cat([gb_normal, alpha], dim=-1),
        "gb_tangent": torch.cat([gb_tangent, alpha], dim=-1),
        "gb_weights": torch.cat([gb_weights, alpha], dim=-1)
    }

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        skinning_weights,
        mtx_in,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None,
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx[0].int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    # view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos, mtx_in, use_python=False)

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx[0].int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(v_pos_clip, rast, db, mesh, skinning_weights, resolution, spp, msaa), rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers
