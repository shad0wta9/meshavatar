import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn
from smplx.lbs import batch_rodrigues

from .embedder import get_embedder
from . import mesh
from . import util
from . import diffused_skinning
from .unet import UNet
from . import render_posmap

class FeatureBlendingNetwork(nn.Module):
    def __init__(self, fb_map_fpath, feat_dim=64, net_type='unet'):
        super().__init__()

        self.fb_map_fpath = fb_map_fpath
        self.feat_dim = feat_dim
        self.AABB = torch.tensor([[-1.0, 1.0], [-1.4, 0.6]]).cuda()

        npz = np.load(self.fb_map_fpath)
        self.fpnts, self.fweights = torch.from_numpy(npz['fpnts']).float().cuda(), torch.from_numpy(npz['fweights']).float().cuda()
        self.bpnts, self.bweights = torch.from_numpy(npz['bpnts']).float().cuda(), torch.from_numpy(npz['bweights']).float().cuda()

        self.fpnts = torch.cat([self.fpnts, torch.ones_like(self.fpnts[..., [0]])], dim=-1)
        self.bpnts = torch.cat([self.bpnts, torch.ones_like(self.bpnts[..., [0]])], dim=-1)

        self.fpnts, self.fweights = self.fpnts[None], self.fweights[None]
        self.bpnts, self.bweights = self.bpnts[None], self.bweights[None]

        self.fvns, self.bvns = torch.from_numpy(npz['fvns']).float().cuda(), torch.from_numpy(npz['bvns']).float().cuda()
        self.fvns, self.bvns = self.fvns[None], self.bvns[None]

        const_feat_map = torch.randn(*self.fpnts.shape[:-1], 16) / 512.
        self.const_feat_map = torch.nn.Parameter(const_feat_map, requires_grad=True)
        self.register_parameter('const_feat_map', self.const_feat_map)

        if net_type == 'unet':
            self.net = UNet(n_channels=4*3+16, n_classes=feat_dim, bilinear=True)
        else:
            assert False, "Unknown Feature Network Type: [%s]" % net_type

        self.feat_map = None

    def skinning(self, transforms, pnts, vns, weights):
        """
        transforms: [B, 55, 4, 4]
        pnts: [1, 512, 512, 4]
        weights: [1, 512, 512, 55]
        return: [B, 512, 512, 3]
        """
        ptransforms = weights @ transforms.reshape(*transforms.shape[:-2], 16)[:, None]
        ptransforms = ptransforms.reshape(*ptransforms.shape[:-1], 4, 4) # [B, 512, 512, 4, 4]

        posed_pnts = (ptransforms @ pnts[..., None])[..., :3, 0]
        posed_vns = (ptransforms[..., :3, :3] @ vns[..., None])[..., 0]
        return posed_pnts, posed_vns
    
    @torch.no_grad()
    def compute_pca(self, transforms, n_components=20):
        """
        transforms: [B, 55, 4, 4]
        """
        n_frames = transforms.shape[0]
        print("Computing PCA for %d transformations" % n_frames)

        transforms_splits = torch.split(transforms, 8, dim=0)
        feat_splits = []
        msk = self.fweights[0].sum(-1) == 0

        for batch_transforms in transforms_splits:
            fposed, fn = self.skinning(batch_transforms, self.fpnts, self.fvns, self.fweights)
            bposed, bn = self.skinning(batch_transforms, self.bpnts, self.bvns, self.bweights)

            feat = torch.cat([fposed[:, ~msk], fn[:, ~msk], bposed[:, ~msk], bn[:, ~msk]], dim=-1) # [B, P, 4*3]

            feat_splits.append(feat)

        feat = torch.cat(feat_splits, dim=0)
        feat = feat.reshape(n_frames, -1) # [B, P * 4*3]

        self.feat_center = feat.mean(dim=0)
        feat_centered = feat - self.feat_center
        
        u, s, v = torch.pca_lowrank(feat_centered, n_components)
        self.pca_components = v
        self.pca_sigma = s / np.sqrt(n_frames - 1)

    def project_pca(self, feat):
        """
        feat: [B, feat_dim]
        """
        feat_centered = feat - self.feat_center
        coe = torch.matmul(feat_centered, self.pca_components)
        coe = torch.clamp(coe, -2 * self.pca_sigma, 2 * self.pca_sigma)

        return torch.matmul(coe, self.pca_components.T) + self.feat_center
    
    def visualize_position_map(self, fposed, fn, bposed, bn):
        import cv2
        img = ((fposed[0] + 1.0) / 2.0).detach().cpu().numpy()
        img = cv2.blur(img, (5, 5))
        # img = ((fn[0] + 1.0) / 2.0).detach().cpu().numpy()
        # img = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.imwrite('debug/fn.jpg', img * 255)

        img = ((bposed[0] + 1.0) / 2.0).detach().cpu().numpy()
        img = cv2.blur(img, (5, 5))
        # img = ((bn[0] + 1.0) / 2.0).detach().cpu().numpy()
        # img = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.imwrite('debug/bn.jpg', img * 255)


        msk = self.fweights.sum(-1) == 0
        points = torch.cat([fposed[[0]][~msk].reshape(-1, 3), bposed[[0]][~msk].reshape(-1, 3)], dim=0).detach().cpu().numpy()
        colors = torch.cat([fn[[0]][~msk].reshape(-1, 3), bn[[0]][~msk].reshape(-1, 3)], dim=0).detach().cpu().numpy()
        colors = (colors + 1.) / 2.

        util.save_point_cloud("./debug/rast_posed.ply", points, colors)

    def update(self, transforms):
        """
        transforms: [B, 55, 4, 4]
        return: [B, feat_dim, 512, 512]
        """
        fposed, fn = self.skinning(transforms, self.fpnts, self.fvns, self.fweights)
        bposed, bn = self.skinning(transforms, self.bpnts, self.bvns, self.bweights)

        # self.visualize_position_map(fposed, fn, bposed, bn)

        if hasattr(self, 'pca_components'):
            fmap = torch.zeros_like(fposed)
            fmap = torch.cat([fmap, fmap, fmap, fmap], dim=-1)

            msk = self.fweights[0].sum(-1) == 0
            feat = torch.cat([fposed[:, ~msk], fn[:, ~msk], bposed[:, ~msk], bn[:, ~msk]], dim=-1) # [B, P, 4*3]
            feat = feat.reshape(transforms.shape[0], -1)
            feat = self.project_pca(feat)

            fmap[:, ~msk] = feat.reshape(transforms.shape[0], -1, 4*3)
            pmap = torch.cat([fmap, self.const_feat_map.expand(fposed.shape[0], -1, -1, -1)], dim=-1).permute(0, 3, 1, 2)
        else:
            pmap = torch.cat([fposed, fn, bposed, bn, self.const_feat_map.expand(fposed.shape[0], -1, -1, -1)], dim=-1).permute(0, 3, 1, 2)
        # self.feat_map = pmap
        # return
        self.feat_map = self.net(pmap.contiguous())
    
    def forward(self, verts, poses=None):
        """
        verts: [B, N, 3]
        return: [B, N, feat_dim]
        p.s. I expect that the feature map can be automatically updated by given [poses],
            but it's not implemented
        """
        assert self.feat_map is not None, "please first apply [update] method to get feature map"
        normalized_verts = (verts[..., :2] - self.AABB[:, 0]) / (self.AABB[:, 1] - self.AABB[:, 0])
        normalized_verts = normalized_verts * 2. - 1. # normalize to [-1, 1] x [-1, 1]

        motion_feat = F.grid_sample(self.feat_map, normalized_verts.unsqueeze(0)).squeeze(0)
        motion_feat = motion_feat.permute(1, 2, 0)

        return motion_feat

    @torch.no_grad()
    def update_posmap(self, glctx, mesh, weights):
        """
        weights: [N, 55], the skinning weights for mesh vertices
        """
        res = [512, 512]
        near, far = 1e-3, 1e3
        device = self.fpnts.device

        proj = np.asarray([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -2 / (far - near), -(near + far) / (far - near)],
                        [0, 0, 0, 1]])
        mv = util.lookAt(torch.tensor([0, -0.4, 2]).float(), 
                 torch.tensor([0, -0.4, 0]).float(), 
                 torch.tensor([0, 1, 0]).float())

        mv, proj = mv.float(), torch.from_numpy(proj).float()
        mvp = proj @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        mvp, campos = mvp[None], campos[None]

        buffers = render_posmap.render_mesh(glctx, mesh, weights, mvp.to(device), 
                    res, spp=1, num_layers=1, msaa=True, background=None)
        self.fpnts, self.fvns = buffers['gb_pos'][..., :3], buffers['gb_normal'][..., :3]
        self.fweights = buffers['gb_weights'][..., :-1]
        self.fpnts = torch.cat([self.fpnts, torch.ones_like(self.fpnts[..., [0]])], dim=-1)

        ### back
        proj = np.asarray([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -2 / (far - near), -(near + far) / (far - near)],
                        [0, 0, 0, 1]])
        mv = util.lookAt(torch.tensor([0, -0.4, -2]).float(), 
                 torch.tensor([0, -0.4, 0]).float(), 
                 torch.tensor([0, 1, 0]).float())

        mv, proj = mv.float(), torch.from_numpy(proj).float()
        mvp = proj @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        mvp, campos = mvp[None], campos[None]

        buffers = render_posmap.render_mesh(glctx, mesh, weights, mvp.to(device), 
                    res, spp=1, num_layers=1, msaa=True, background=None)
        self.bpnts, self.bvns = buffers['gb_pos'][..., :3], buffers['gb_normal'][..., :3]
        self.bweights = buffers['gb_weights'][..., :-1]
        self.bpnts = torch.cat([self.bpnts, torch.ones_like(self.bpnts[..., [0]])], dim=-1)


class CanoDeformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, feat_dim=64, depth=5, embed_freq=6, return_scale=5e-2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.depth = depth
        self.return_scale = return_scale

        self.coord_embedder, self.coord_ebddim = get_embedder(embed_freq, input_dims=self.input_dim, include_input=True)
        self.net = [nn.Linear(self.coord_ebddim + self.feat_dim, self.hidden_dim)] + \
            [nn.ReLU() if i % 2 == 0 else nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.depth * 2 - 3)] + \
            [nn.Linear(self.hidden_dim, self.input_dim)]
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x, dyn_feat):
        """
        x, arti_x: [B, N, 3]
        dyn_feat: [B, F]
        return: [B, N, 3]
        """
        x_embed = self.coord_embedder(x)
        if isinstance(dyn_feat, torch.Tensor):
            input_dyn_feat = dyn_feat[:, None].expand(-1, x_embed.shape[1], -1)
        else:
            input_dyn_feat = dyn_feat(x)
        x_input = torch.cat([x_embed, input_dyn_feat], dim=-1)
        return self.net(x_input) * self.return_scale

    def reinitialize(self):
        for i in range(0, self.depth * 2 - 1, 2):
            torch.nn.init.kaiming_normal_(self.net[i].weight, nonlinearity='relu')
            torch.nn.init.normal_(self.net[i].bias, std=0.01)

class PosedDeformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, feat_dim=64, depth=2, embed_freq=6, return_scale=1e-2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.depth = depth
        self.return_scale = return_scale

        if embed_freq == 'hash':
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

            # gradient_scaling = 128.0
            self.coord_embedder = tcnn.Encoding(3, enc_cfg)
            self.coord_embedder2 = tcnn.Encoding(3, enc_cfg)
            self.embed_dim = self.coord_embedder.n_output_dims
        else:
            self.coord_embedder, self.embed_dim = get_embedder(embed_freq, input_dims=self.input_dim, include_input=True)
            self.coord_embedder2 = self.coord_embedder

        self.net = [nn.Linear(2 * self.embed_dim + self.feat_dim, self.hidden_dim)] + \
            [nn.ReLU() if i % 2 == 0 else nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.depth * 2 - 3)] + \
            [nn.Linear(self.hidden_dim, self.input_dim)]
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x, arti_x, dyn_feat):
        """
        x, arti_x: [B, N, 3]
        dyn_feat: [B, F]
        return: [B, N, 3]
        """
        B, N = x.shape[:2]
        x_embed = self.coord_embedder(x.reshape(-1, 3)).reshape(B, N, -1)
        arti_x_embed = self.coord_embedder2(arti_x.reshape(-1, 3)).reshape(B, N, -1)

        if isinstance(dyn_feat, torch.Tensor):
            input_dyn_feat = dyn_feat[:, None].expand(-1, x_embed.shape[1], -1)
        else:
            input_dyn_feat = dyn_feat(x)

        msk = x[..., [-1]] > 0
        # print(msk.shape, input_dyn_feat.shape)
        ret = msk * input_dyn_feat[..., :3] + (~msk) * input_dyn_feat[..., 3:6]
        return ret * self.return_scale

    def reinitialize(self):
        for i in range(0, self.depth * 2 - 1, 2):
            torch.nn.init.kaiming_normal_(self.net[i].weight, nonlinearity='relu')
            torch.nn.init.normal_(self.net[i].bias, std=0.01)

class ForwardDeformer(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()

        smplx = np.load(FLAGS.smpl_file_path)
        self.jnum = smplx['J_regressor'].shape[0]

        fb_map_fpath = 'data/data_templates/%s/fb_map.npz' % FLAGS.dataset_name
        self.motion_feat_net = FeatureBlendingNetwork(fb_map_fpath=fb_map_fpath, 
                                                        feat_dim=FLAGS.feat_dim)

        self.deformer_type = FLAGS.deformer_type
        return_scale = 1e-2
        if FLAGS.deformer_type == "cano":
            self.deformation_net = CanoDeformer(feat_dim=FLAGS.feat_dim, return_scale=return_scale)
        elif FLAGS.deformer_type == "posed":
            self.deformation_net = PosedDeformer(feat_dim=FLAGS.feat_dim, return_scale=return_scale)

        verts = smplx['v_template']
        shapedirs = smplx['shapedirs']
        verts = verts + (shapedirs[..., :10] @ FLAGS.beta[:, None])[..., 0]

        leg_angle = 15.0
        self.cano_pose = torch.zeros((1, 165)).cuda()
        self.cano_pose[:, 5] =  leg_angle / 180 * torch.pi
        self.cano_pose[:, 8] = -leg_angle / 180 * torch.pi

        self.skinning_model = diffused_skinning.ForwardDiffusedSkinning(FLAGS,
                        kintree=torch.from_numpy(smplx['kintree_table'][0]).cuda(), 
                        rest_joints=torch.from_numpy(smplx['J_regressor'] @ verts).cuda().float(),
                        cano_pose=self.cano_pose)

    def get_motion_feat(self, poses=None, t=None, tfs=None):
        if poses is None:
            poses = self.cano_pose.clone()

        if tfs is None:
            tfs = self.skinning_model.get_tfs(poses)
            tfs = torch.einsum('bnij,njk->bnik', tfs, self.skinning_model.cano_tfs_inv)
        self.motion_feat_net.update(tfs)
        pose_feat = self.motion_feat_net

        return pose_feat
    
    def compute_pca(self, training_poses):
        if not isinstance(self.motion_feat_net, FeatureBlendingNetwork):
            return
        zero_poses = torch.cat([torch.zeros_like(training_poses[..., :3]), training_poses[..., 3:]], dim=-1)
        training_transforms = self.skinning_model.get_tfs(zero_poses)
        self.motion_feat_net.compute_pca(training_transforms)

    def visualize_skinning_weights(self, verts, faces):
        import open3d as o3d
        skinning_weights = self.skinning_model.compute_skinning_weights(verts)
        skinning_weights[..., 15] += skinning_weights[..., 22] + skinning_weights[..., 23]
        skinning_weights[..., 22] = skinning_weights[..., 25:40].sum(-1)
        skinning_weights[..., 23] = skinning_weights[..., 40:55].sum(-1)
        vert_colors = diffused_skinning.weights2colors(skinning_weights[0, :, :24].detach().cpu().numpy())

        util.save_triangle_mesh('debug/mesh_cano_weights.ply', verts, faces, vert_colors)

    def update_posmap(self, glctx, shape):
        skinning_weights = self.skinning_model.compute_skinning_weights(shape.v_pos)
        # poses = torch.from_numpy(np.loadtxt('debug/pose60.txt')).float().to(shape.v_pos.device)[None]
        # zero_poses = torch.cat([torch.zeros_like(poses[..., :3]), poses[..., 3:]], dim=-1)
        # transforms = self.skinning_model.get_tfs(zero_poses)
        self.motion_feat_net.update_posmap(glctx, shape, skinning_weights)

    def forward(self, shape, poses, t=None, rots=None, trans=None):

        verts = shape.v_pos
        zero_poses = torch.cat([torch.zeros_like(poses[..., :3]), poses[..., 3:]], dim=-1)
        transforms = self.skinning_model.get_tfs(zero_poses)
        pose_feat = self.get_motion_feat(poses, t, transforms)

        # self.visualize_skinning_weights(verts, shape.t_pos_idx)

        if self.deformer_type == "cano":
            deformation = self.deformation_net(x=verts, dyn_feat=pose_feat)
            deformed_verts = verts + deformation

            result_verts = self.skinning_model(deformed_verts, poses, cano_verts=verts)

        elif self.deformer_type == "posed":
            zero_articulated_verts = self.skinning_model(verts, zero_poses)
            articulated_verts = self.skinning_model(verts, poses)

            # util.save_triangle_mesh('debug/mesh_posed.ply', articulated_verts, shape.t_pos_idx)

            deformation = self.deformation_net(x=verts, arti_x=zero_articulated_verts, dyn_feat=pose_feat)
            # deformation = torch.zeros_like(articulated_verts)
            deformation = (batch_rodrigues(poses[..., :3])[:, None] @ deformation.unsqueeze(-1))[..., 0]
            result_verts = articulated_verts + deformation

        if rots is not None:
            result_verts = torch.matmul(batch_rodrigues(rots), result_verts.transpose(-1, -2)).transpose(-1, -2)
        if trans is not None:
            result_verts += trans[:, None]

        return mesh.make_mesh(result_verts, shape.t_pos_idx, shape.v_tex, shape.t_tex_idx, shape.material), pose_feat, deformation

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            AABB,
            feature_vector_size = 0,
            d_in = 3,
            d_out = 1,
            dims = [256, 256, 256, 256, 256, 256],
            geometric_init=True,
            bias=0.5,
            skip_in=[4],
            weight_norm=True,
            multires=10
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires == 'hash':
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

            # gradient_scaling = 128.0
            self.embed_fn = tcnn.Encoding(3, enc_cfg)
            input_ch = self.embed_fn.n_output_dims
            dims[0] = input_ch
        elif multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[..., :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)