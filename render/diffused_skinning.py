import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import trimesh

from smplx.lbs import batch_rodrigues

def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = [ 'pink', #0
                'blue', #1
                'green', #2
                'red', #3
                'pink', #4
                'pink', #5
                'pink', #6
                'green', #7
                'blue', #8
                'red', #9
                'pink', #10
                'pink', #11
                'pink', #12
                'blue', #13
                'green', #14
                'red', #15
                'cyan', #16
                'darkgreen', #17
                'pink', #18
                'pink', #19
                'blue', #20
                'green', #21
                'pink', #22
                'pink' #23
    ]


    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[1],
                    'green':cmap.colors[3],
                    'pink': [1,1,1],
                    'red':cmap.colors[5],
                    }

    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]# [1x24x3]
    verts_colors = weights[:,:,None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors

class DiffusedSkinningTable:
    def __init__(self, dataset_name,
                 data_path='data/diffused_skinning',
                 device='cuda'):

        cpose_smpl_file = 'data/data_templates/%s/%s_minimal_cpose.obj' % (dataset_name, dataset_name)
        cpose_smpl_mesh = trimesh.load(cpose_smpl_file, process=False)
        cpose_verts = torch.from_numpy(np.array(cpose_smpl_mesh.vertices)).float().to(device)[:, :3]
        bbox_data_min = cpose_verts.min(0).values
        bbox_data_max = cpose_verts.max(0).values
        bbox_data_extend = (bbox_data_max - bbox_data_min).max()
        bbox_grid_extend = bbox_data_extend * 1.1
        center = (bbox_data_min + bbox_data_max) / 2
        
        # cpose_weight_grid_path = os.path.join(data_path, 'smplx_cano_lbs_weights_grid_float32.npy')
        cpose_weight_grid_path = 'data/data_templates/%s/%s_cano_lbs_weights_grid_float32.npy' % (dataset_name, dataset_name)
        grid_pt = torch.from_numpy(np.load(cpose_weight_grid_path)).float().to(device)

        self.bbox_grid_extend = bbox_grid_extend
        self.bbox_grid_center = center
        self.weight_grid = grid_pt

        lhand_id, rhand_id = np.loadtxt(os.path.join(data_path, 'smpl_lhand_id.txt')), \
            np.loadtxt(os.path.join(data_path, 'smpl_rhand_id.txt'))
        cpose_lhand_verts, cpose_rhand_verts = cpose_verts[lhand_id], cpose_verts[rhand_id]

        bbox_data_min = cpose_lhand_verts.min(0).values
        bbox_data_max = cpose_lhand_verts.max(0).values
        bbox_data_extend = (bbox_data_max - bbox_data_min).max()
        bbox_grid_extend = bbox_data_extend * 1.1
        center = (bbox_data_min + bbox_data_max) / 2
        
        tmp_list = cpose_weight_grid_path.split('_')
        tmp_list.insert(-1, 'lhand')
        tmp_fpath = '_'.join(tmp_list)
        if os.path.exists(tmp_fpath):
            grid_pt = torch.from_numpy(np.load(tmp_fpath)).float().to(device)
        else:
            grid_pt = None

        self.lhand_bbox_grid_extend = self.bbox_grid_extend / bbox_grid_extend
        self.lhand_bbox_grid_center = (self.bbox_grid_center - center) / bbox_grid_extend * 2
        self.lhand_weight_grid = grid_pt

        bbox_data_min = cpose_rhand_verts.min(0).values
        bbox_data_max = cpose_rhand_verts.max(0).values
        bbox_data_extend = (bbox_data_max - bbox_data_min).max()
        bbox_grid_extend = bbox_data_extend * 1.1
        center = (bbox_data_min + bbox_data_max) / 2
        
        tmp_list = cpose_weight_grid_path.split('_')
        tmp_list.insert(-1, 'rhand')
        tmp_fpath = '_'.join(tmp_list)
        if os.path.exists(tmp_fpath):
            grid_pt = torch.from_numpy(np.load(tmp_fpath)).float().to(device)
        else:
            grid_pt = None

        self.rhand_bbox_grid_extend = self.bbox_grid_extend / bbox_grid_extend
        self.rhand_bbox_grid_center = (self.bbox_grid_center - center) / bbox_grid_extend * 2
        self.rhand_weight_grid = grid_pt

    def query(self, p_xc):
        """
        points: [B, N, 3]
        return: [B, N, J]
        """
        def get_w(p_xc, p_mask, p_grid, p_lgrid, p_rgrid):

            def to_left(p_xc):
                p_xcl = p_xc * self.lhand_bbox_grid_extend + self.lhand_bbox_grid_center
                return p_xcl, ((p_xcl >= -1) & (p_xcl <= 1)).all(dim=-1)
            
            def to_right(p_xc):
                p_xcr = p_xc * self.rhand_bbox_grid_extend + self.rhand_bbox_grid_center
                return p_xcr, ((p_xcr >= -1) & (p_xcr <= 1)).all(dim=-1)

            n_batch, n_point, n_dim = p_xc.shape

            if n_batch * n_point == 0:
                return p_xc

            # reshape to [N,?]
            p_xc = p_xc.reshape(n_batch * n_point, n_dim)
            if p_mask is not None:
                p_xc = p_xc[p_mask]   # (n_b*n_p, n_dim)

            x = F.grid_sample(p_grid[None],
                              p_xc[None, None, None],
                              align_corners=False,
                              padding_mode='border')[0, :, 0, 0].T  # [Nv, 24]

            if p_lgrid is not None:
                p_xcl, lmask = to_left(p_xc)
                if lmask.sum() > 0:
                    x[lmask] = F.grid_sample(p_lgrid[None],
                                    p_xcl[lmask][None, None, None],
                                    align_corners=False,
                                    padding_mode='border')[0, :, 0, 0].T  # [Nv, 24]

            if p_rgrid is not None:
                p_xcr, rmask = to_right(p_xc)
                if rmask.sum() > 0:
                    x[rmask] = F.grid_sample(p_rgrid[None],
                                    p_xcr[rmask][None, None, None],
                                    align_corners=False,
                                    padding_mode='border')[0, :, 0, 0].T  # [Nv, 24]

            # add placeholder for masked prediction
            if p_mask is not None:
                x_full = torch.zeros(n_batch * n_point, x.shape[-1], device=x.device)
                x_full[p_mask] = x
            else:
                x_full = x

            return x_full.reshape(n_batch, n_point, -1)
        
        def inv_transform_v(v, scale_grid, transl):
            """
            v: [b, n, 3]
            """
            v = v - transl[None, None]
            v = v / scale_grid
            v = v * 2

            return v



        v_cano_in_grid_coords = inv_transform_v(p_xc, self.bbox_grid_extend, self.bbox_grid_center)
        out = get_w(v_cano_in_grid_coords, None, self.weight_grid, self.lhand_weight_grid, self.rhand_weight_grid)
        # out = F.grid_sample(grid_pt[None], v_cano_in_grid_coords[None, None], align_corners=False, padding_mode='border')[0, :, 0, 0].T  # [Nv, 24]
        w = out

        return w

class ForwardDiffusedSkinning(nn.Module):
    def __init__(self, FLAGS, kintree, rest_joints, cano_pose):
        super().__init__()

        self.kintree = kintree
        self.rest_joints = rest_joints
        self.jnum = len(self.rest_joints)
        self.rest_bones = torch.stack([self.rest_joints[self.kintree[1:]], self.rest_joints[1:]], dim=0)
        self.diffused_skinning_table = DiffusedSkinningTable(dataset_name=FLAGS.dataset_name)

        self.cano_tfs_inv = self.get_tfs(cano_pose, False).squeeze(0).inverse() # [J, 4, 4]

    def compute_skinning_weights(self, verts):
        """
        verts: [B, N, 3]
        return: [B, N, J]
        """
        return self.diffused_skinning_table.query(verts)
    
    def get_tfs(self, poses, compute_inv=True):

        batch_size = poses.shape[0]
        # poses[:, :3] *= 0
        rot_mats = batch_rodrigues(poses.view(-1, 3)).view(batch_size, -1, 3, 3) # [B, J, 3, 3]

        rel_joints = self.rest_joints.clone()
        rel_joints[1:] -= self.rest_joints[self.kintree[1:]]
        rel_joints = rel_joints[None].expand(batch_size, -1, -1)
        transform_mats = torch.cat([
            torch.cat([rot_mats, rel_joints[..., None]], dim=-1),
            torch.zeros((batch_size, self.jnum, 1, 4)).to(rot_mats.device)
        ], dim=-2) # [B, J, 4, 4]
        transform_mats[:, :, 3, 3] = 1

        transform_chain = [transform_mats[:, 0]]
        for i in range(1, self.jnum):
            transform_chain.append(torch.matmul(transform_chain[self.kintree[i]], transform_mats[:, i]))
        transforms = torch.stack(transform_chain, dim=1)

        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(self.rest_joints[None, ..., None].expand(batch_size, -1, -1, -1), [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [B, J, 4, 4]
        
        if compute_inv:
            rel_transforms = torch.einsum('bnij,njk->bnik', rel_transforms, self.cano_tfs_inv)

        return rel_transforms


    def forward(self, verts, poses, cano_verts=None):
        """
        verts: [B, N, 3]
        poses: [B, J, 3]
        cano_verts: [1, N, 3] or [N, 3], given if we want identical skinning weights among batches
        return: [B, N, 3]
        """

        batch_size, num_verts = verts.shape[:2]

        rel_transforms = self.get_tfs(poses)

        if cano_verts is None:
            weights = self.compute_skinning_weights(verts) # [B, N, J]
        else:
            if len(cano_verts.shape) == 2:
                cano_verts = cano_verts[None]
            weights = self.compute_skinning_weights(cano_verts).expand(batch_size, -1, -1)
        
        weighted_transforms = torch.matmul(weights, rel_transforms.reshape(batch_size, -1, 16)) \
            .reshape(batch_size, -1, 4, 4) # [B, N, 4, 4]
        posed_verts = (weighted_transforms[..., :3, :3] @ verts[..., None])[..., 0] \
              + weighted_transforms[..., :3, 3]
        
        return posed_verts