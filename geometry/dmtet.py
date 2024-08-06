# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

from render import mesh
from render import render
from render import networks
from render import light
from render import regularizer
from .tetmesh import subdivide_tetmesh
import render.optixutils as ou

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        eps = 0. # 1e-6
        denominator = edges_to_interp_sdf.sum(1,keepdim = True)
        denominator += denominator.sign() * eps

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        return verts, faces, uvs, uv_idx

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_adj = ((sdf_f1x6x2[...,0] - sdf_f1x6x2[...,1]) ** 2).mean() * 10.
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff # + sdf_adj

def eikonal_loss(sdf, v_pos, AABB):
    num_samples = 5000
    sample_points = torch.rand(num_samples, 3, device=v_pos.device) * (AABB[1] - AABB[0]) + AABB[0]
    mesh_verts = v_pos.detach() + (torch.rand_like(v_pos) -0.5) * 0.1 * 1.0
    rand_idx = torch.randperm(len(mesh_verts), device=mesh_verts.device)[:num_samples]
    mesh_verts = mesh_verts[rand_idx]
    sample_points = torch.cat([sample_points, mesh_verts], 0)
    grad = sdf.gradient(sample_points)
    return ((torch.norm(grad, dim=-1) - 1.) ** 2).mean()

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
        self.lambda_kd     = FLAGS.lambda_kd
        self.lambda_ks     = FLAGS.lambda_ks
        self.lambda_nrm    = FLAGS.lambda_nrm
        self.mesh          = None

        tets = np.load('data/human-hull-tets/{}_tets.npz'.format(self.grid_res))
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()
        print('tetrahedral grid ranges from [%f %f %f] to [%f %f %f]' % (
            self.verts[:, 0].min(), self.verts[:, 1].min(), self.verts[:, 2].min(),
            self.verts[:, 0].max(), self.verts[:, 1].max(), self.verts[:, 2].max()))

        # Random init
        if self.FLAGS.use_mlp_sdf:
            self.sdf_func = networks.ImplicitNetwork(self.getAABB()).cuda()
        else:
            sdf_init = True
            if sdf_init:
                sdf = torch.from_numpy(np.load('data/human-hull-tets/{}_smplx_sdf.npy'.format(self.grid_res))).float().to(self.verts.device)
            else:
                sdf = torch.rand_like(self.verts[:,0]) - 0.1

            self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
            self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

        self.forward_deformer = networks.ForwardDeformer(FLAGS).cuda()

        if self.FLAGS.mc:
            with torch.no_grad():
                self.optix_ctx = ou.OptiXContext()
        else:
            self.optix_ctx = None

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def subdivide(self):
        with torch.no_grad():
            if self.FLAGS.use_mlp_sdf:
                v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
                self.sdf = self.sdf_func(v_deformed)[:, 0]

            # eliminate the useless tets
            tet_msk = (self.sdf[self.indices] > 0).sum(dim=-1)
            tet_msk = torch.logical_and(tet_msk > 0, tet_msk < 4)

            vert_cnt = torch.bincount(self.indices[tet_msk].reshape(-1), minlength=self.verts.shape[0])
            vert_msk = vert_cnt > 0

            deform = 2 / (self.grid_res * 2) * torch.tanh(self.deform)
            new_verts, new_sdf, new_deform = self.verts[vert_msk], self.sdf[vert_msk], deform[vert_msk]
            idx = torch.zeros_like(vert_msk).long()
            idx[vert_msk] = torch.arange(vert_msk.sum(), device=vert_msk.device).long()
            new_indices = idx[self.indices[tet_msk]]

            feat = torch.cat([new_sdf.unsqueeze(-1), new_deform], dim=-1)
            new_verts, feat = new_verts[None], feat[None]
            verts, indices, feat = subdivide_tetmesh(new_verts, new_indices, feat)
            verts, feat = verts[0], feat[0]
            sdf, deform = feat[:, 0], feat[:, 1:]

            self.verts, self.indices = verts, indices
            deform = torch.arctanh(deform * self.grid_res)
            self.deform = torch.nn.Parameter(deform, requires_grad=True)
            self.register_parameter('deform', self.deform)
            # self.sdf = sdf
            if not self.FLAGS.use_mlp_sdf:
                self.sdf = torch.nn.Parameter(sdf, requires_grad=True)
                self.register_parameter('sdf', self.sdf)

            # self.grid_res *= 2

        self.generate_edges()
        print("Subdivide successfully!")

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def buildMesh(self, material):
        self.mesh = self.getMesh(material)

    def getMesh(self, material):
        if self.mesh is not None:
            return self.mesh
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        # v_deformed = self.verts
        if self.FLAGS.use_mlp_sdf:
            self.sdf = self.sdf_func(v_deformed)
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf, self.indices)
        return mesh.make_mesh(verts[None], faces[None], uvs[None], uv_idx[None], material)

    def update_posmap(self, glctx):
        cano_mesh = self.getMesh(None)
        self.forward_deformer.update_posmap(glctx, cano_mesh)

    def render(self, glctx, target, lgt, opt_material, denoiser=None, shadow_scale=1.0, bsdf=None, training=False):
        cano_pos, motion_feat = None, None
        opt_mesh = self.getMesh(opt_material) # .extend(target['mvp'].shape[0])

        cano_pos = opt_mesh.v_pos
        if cano_pos.shape[1] == 0:
            assert False, "The geometry crashed (no triangles)! Please restart the program!"
        opt_mesh, motion_feat, deformation = self.forward_deformer(opt_mesh, target['poses'][[0]], 
                            None if target['idx'] is None else target['idx'][[0]],
                            None if target['rots'] is None else target['rots'][[0]],
                            None if target['trans'] is None else target['trans'][[0]])

        if self.FLAGS.mc:
            ou.optix_build_bvh(self.optix_ctx, opt_mesh.v_pos[0].contiguous(), opt_mesh.t_pos_idx[0].int(), rebuild=1)
        
        self.mesh_verts = opt_mesh.v_pos
        self.mesh_faces = opt_mesh.t_pos_idx
        if self.mesh_verts.requires_grad:
            self.mesh_verts.retain_grad()

        if denoiser is not None:
            denoiser.set_influence(shadow_scale / 4.)
        buffers = render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    msaa=True, background=target['background'], bsdf=bsdf, cano_pos=cano_pos,
                                    cond=motion_feat, optix_ctx=self.optix_ctx, denoiser=denoiser, shadow_scale=shadow_scale)
        self.rndr_img = buffers['shaded']
        if self.rndr_img.requires_grad:
            self.rndr_img.retain_grad()
        buffers.update({'deformation': deformation})

        if training is True:
            return buffers, cano_pos, motion_feat
        return buffers


    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, denoiser=None, perc_loss_fn=None):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        shadow_ramp = min(iteration / 2000, 1.0)
        buffers, cano_pos, motion_feat = self.render(glctx, target, lgt, opt_material, denoiser, shadow_ramp, training=True)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter
        losses = {}

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        msk = color_ref[..., 3] <= 1
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][msk][..., 3:], color_ref[msk][..., 3:])
        color_msk = color_ref[..., 3:].clone()
        color_msk[~msk] = 0
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_msk, color_ref[..., 0:3] * color_msk)

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01)*min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight # Dropoff to 0.01
        if self.FLAGS.use_mlp_sdf:
            eik = eikonal_loss(self.sdf_func, cano_pos[0].detach(), self.getAABB())
            eik_weight = 0.01 - (0.01 - 0.001) * min(1.0, 4 * t_iter)         
            reg_loss += eik * eik_weight

        # Deformation regularizer
        if 'deformation' in buffers:
            deformation_weight = 1.0 - (1.0 - 0.01)*min(1.0, 4.0 * t_iter)
            deformation_loss = torch.mean(buffers['deformation'] ** 2) * 1000. * deformation_weight # * min(1.0, iteration / 500)
            losses.update({"deformation_loss": deformation_loss})
        
        # Pose Feature Loss
        # reg_loss += torch.mean(motion_feat.feat_map ** 2) * 0.01

        # Perceptual loss
        if perc_loss_fn is not None:
            perc_loss = perc_loss_fn(buffers['shaded'][..., 0:3] * color_msk, 
                                        color_ref[..., 0:3] * color_msk, color_msk[..., 0])
            perc_loss = perc_loss * 0.1
            losses.update({"perc_loss": perc_loss})

        if 'nml' in target:
            nml, gb_normal = target['nml'][..., :3], buffers['gb_normal'][..., :3]
            nml_msk = color_msk * target['nml'][..., 3:]
            nml_loss = loss_fn(gb_normal * nml_msk, nml * nml_msk)
            losses.update({"nml_loss": nml_loss})

        # Albedo (k_d) smoothnesss regularizer
        # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)
        reg_loss += regularizer.material_smoothness_grad(buffers['kd_grad'], buffers['ks_grad'], 
                                buffers['normal_grad'] if 'normal_grad' in buffers else None,
                                lambda_kd=self.lambda_kd, lambda_ks=self.lambda_ks, lambda_nrm=self.lambda_nrm)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.005
        if isinstance(lgt, light.EnvironmentLight):
            pass
        else:
            reg_loss = reg_loss + regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref)

        losses.update({
            "img_loss": img_loss,
            "reg_loss": reg_loss
        })

        return losses
