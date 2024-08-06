
import os
import time
import argparse
import json

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

import torch.nn.functional as F

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Perceptual loss
###############################################################################

import lpips

def sample_patch_bbox(msk, patch_size):
    rids, cids = torch.nonzero(torch.abs(msk-1) < 1e-6, as_tuple=True)
    i = np.random.randint(0, len(rids))
    r, c = rids[i], cids[i]
    r = torch.clip(r - patch_size // 2, 0, msk.shape[0] - patch_size - 1)
    c = torch.clip(c - patch_size // 2, 0, msk.shape[1] - patch_size - 1)
    return c, r, c + patch_size, r + patch_size

def extract_bbox(msk):
    rcids = torch.nonzero(torch.abs(msk-1) < 1e-6)

    minr, minc = torch.min(rcids, dim=0)[0]
    maxr, maxc = torch.max(rcids, dim=0)[0]

    return minc, minr, maxc+1, maxr+1

class PercLoss(torch.nn.Module):
    def __init__(self, patch_size, device='cuda'):
        super().__init__()

        self.patch_size = patch_size
        self.loss_fn = lpips.LPIPS(net='vgg').to(device)

    def forward(self, img, ref, msk, tonemap=True):
        if tonemap is True:
            img, ref = util.rgb_to_srgb(img), util.rgb_to_srgb(ref)
        img, ref = img.permute(0, 3, 1, 2), ref.permute(0, 3, 1, 2)

        img_patch, ref_patch = [], []
        for i in range(img.shape[0]):
            ### sample a patch of patch_size
            # bbox = sample_patch_bbox(msk[i], self.patch_size)
            # img_patch.append(img[i, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            # ref_patch.append(ref[i, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])

            ### extract the maximal patch, and resize it to patch_size
            bbox = extract_bbox(msk[i])
            bimg = img[i, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
            bref = ref[i, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]

            H, W = bimg.shape[-2:]
            if H > W:
                pad0 = (H - W) // 2
                pad1 = H - W - pad0
                bimg = F.pad(bimg, (pad0, pad1))
                bref = F.pad(bref, (pad0, pad1))
            else:
                pad0 = (W - H) // 2
                pad1 = W - H - pad0
                bimg = F.pad(bimg, (0, 0, pad0, pad1))
                bref = F.pad(bref, (0, 0, pad0, pad1))

            bimg = F.interpolate(bimg[None], (self.patch_size, self.patch_size), mode="bilinear")[0]
            bref = F.interpolate(bref[None], (self.patch_size, self.patch_size), mode="bilinear")[0]
            img_patch.append(bimg)
            ref_patch.append(bref)

        img_patch = torch.stack(img_patch) * 2. - 1.
        ref_patch = torch.stack(ref_patch) * 2. - 1.


        # img_patch = img
        # ref_patch = ref

        # H, W = img_patch.shape[-2:]
        # if H > W:
        #     pad0 = (H - W) // 2
        #     pad1 = H - W - pad0
        #     img_patch = F.pad(img_patch, (pad0, pad1))
        #     ref_patch = F.pad(ref_patch, (pad0, pad1))
        # else:
        #     pad0 = (W - H) // 2
        #     pad1 = W - H - pad0
        #     img_patch = F.pad(img_patch, (0, 0, pad0, pad1))
        #     ref_patch = F.pad(ref_patch, (0, 0, pad0, pad1))
        
        # img_patch = F.interpolate(img_patch, (self.patch_size, self.patch_size), mode="bilinear") * 2. - 1.
        # ref_patch = F.interpolate(ref_patch, (self.patch_size, self.patch_size), mode="bilinear") * 2. - 1.

        # import cv2
        # img_patch_host = img_patch.detach().permute(0, 2, 3, 1).cpu().numpy()[0]
        # ref_patch_host = ref_patch.detach().permute(0, 2, 3, 1).cpu().numpy()[0]
        # cv2.imwrite('debug/img_patch.jpg', (img_patch_host + 1.) / 2. * 255.)
        # cv2.imwrite('debug/ref_patch.jpg', (ref_patch_host + 1.) / 2. * 255.)
        # import ipdb; ipdb.set_trace()

        ret_loss = torch.mean(self.loss_fn(img_patch, ref_patch))

        if ret_loss < 0.:
            import ipdb; ipdb.set_trace()

        return ret_loss

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    # target['mv'] = target['mv'].cuda()
    # target['mvp'] = target['mvp'].cuda()
    # target['campos'] = target['campos'].cuda()
    # target['img'] = target['img'].cuda()
    target['background'] = background
    
    for key in target.keys():
        if key == 'background':
            continue
        if isinstance(target[key], torch.Tensor): 
            target[key] = target[key].cuda()

    w = target['img'][..., 3:4].clone()
    w[w > 1] = 0
    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], w), target['img'][..., 3:4]), dim=-1)
    if 'nml' in target:
        target['nml'] = torch.cat((torch.lerp(background, target['nml'][..., 0:3], w), target['nml'][..., 3:4]), dim=-1)

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS, pose=None, debug=False):
    eval_mesh = geometry.getMesh(mat)

    if debug:
        util.save_triangle_mesh('debug/mesh_cano.ply', eval_mesh.v_pos, eval_mesh.t_pos_idx)

    cano_pos = None
    if pose is not None:
        cano_pos = eval_mesh.v_pos
        eval_mesh, cond, _ = geometry.forward_deformer(eval_mesh, pose['poses'], pose['idx'], pose['rots'], pose['trans'])

        if debug:
            util.save_triangle_mesh('debug/mesh_deformed.ply', eval_mesh.v_pos, eval_mesh.t_pos_idx)
    else:
        cond = geometry.forward_deformer.get_motion_feat()
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos[0].detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx[0].detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs[None], t_tex_idx=faces[None], base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'], cond, cano_pos)
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        if FLAGS.pbr == True:
            mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max], cond_dim=FLAGS.feat_dim if FLAGS.static_texture is False else 0)
            mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
        else:
            mlp_map_opt = mlptexture.MLPRadiance3D(geometry.getAABB(), min_max=[kd_min[0:3], kd_max[0:3]], cond_dim=FLAGS.feat_dim)
            # mlp_map_opt = mlptexture.FeatMapRadiance(geometry.getAABB(), min_max=[kd_min[0:3], kd_max[0:3]], cond_dim=64)
            mat = material.Material({'radiance': mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        if FLAGS.pbr == True:
            mat['bsdf'] = 'pbr'
            if FLAGS.mc == True:
                mat['bsdf'] += '-optix'
        else:
            mat['bsdf'] = 'radiance'

    return mat

def get_flags():
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-w', '--num-workers', type=int, default=0)
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-dd', '--data_dir', type=str)
    parser.add_argument('--validate', type=bool, default=True)
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True
    FLAGS.pbr                 = True
    FLAGS.mc                  = False
    FLAGS.deformer_type       = "posed"
    FLAGS.use_mlp_sdf         = False
    FLAGS.perc_patch_size     = 0
    FLAGS.beta                = np.zeros(10)
    FLAGS.warmup_iter         = 100
    FLAGS.log_interval        = 10
    FLAGS.resume              = None
    FLAGS.normal_supervised   = False
    FLAGS.dataset_name        = "avatarrex_zzr"
    FLAGS.start_iter          = 0
    FLAGS.static_texture      = False
    FLAGS.feat_dim            = 64
    FLAGS.lambda_kd           = 0.1
    FLAGS.lambda_ks           = 0.05
    FLAGS.lambda_nrm          = 0.025
    FLAGS.posmap_update_interval = 0

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    FLAGS.out_dir = 'out/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    return FLAGS