# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import time

import numpy as np
import torch
import nvdiffrast.torch as dr

# Import data readers / generators
from dataset.dataset_synthetic import DatasetSynthetic, TestDatasetSynthetic

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry

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

from denoiser.denoiser import BilateralDenoiser
import train_utils

RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, bbox=None, relight=None, denoiser=None):
    result_dict = {}
    with torch.no_grad():
        if not FLAGS.mc:
            lgt.build_mips()
            if FLAGS.camera_space_light:
                lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material, denoiser)

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        if bbox is not None:
            result_dict['ref'] = result_dict['ref'][bbox[1]:bbox[3], bbox[0]:bbox[2]]
            result_dict['opt'] = result_dict['opt'][bbox[1]:bbox[3], bbox[0]:bbox[2]]
        result_image = torch.cat([result_dict['ref'], result_dict['opt']], axis=1)

        display = [
                {"bsdf" : "normal"} # , {"normal": True}
            ]
        if FLAGS.pbr:
            display += [{"bsdf": "kd"}, {"bsdf": "ks"}]

        if relight is not None:
            display += relight

            mv, mvp = target['mv'], target['mvp']
            proj_mtx = torch.matmul(mvp, torch.linalg.inv(mv))[0]
            h, w = target['resolution']
            fx, fy = proj_mtx[0, 0] * w / 2, -proj_mtx[1, 1] * h / 2
            cx, cy = (w - proj_mtx[0, 2] * w) / 2., (h - proj_mtx[1, 2] * h) / 2.
            fx, fy, cx, cy = fx.item(), fy.item(), cx.item(), cy.item()

            i, j = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
            dirs = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1)

            # Rotate ray directions from camera frame to the world frame
            rays_d = np.sum(dirs[..., np.newaxis, :] * np.linalg.inv(mv[0, :3, :3].detach().cpu().numpy()), -1)

            rays_d = torch.from_numpy(rays_d).cuda()
            rays_d = F.normalize(rays_d, dim=-1)
            phi = torch.arccos(rays_d[..., 1])
            theta = torch.atan2(rays_d[..., 0], -rays_d[..., 2])

            # normalize to [-1, 1]
            query_y = (phi / np.pi) * 2 - 1
            query_x = theta / np.pi
            grid = torch.stack((query_x, query_y), dim=-1).unsqueeze(0) # [1, h, w, 2]
            # import ipdb; ipdb.set_trace()
            # light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

        relight_cnt = 0
        render_envmap = True

        if display is not None:
            white_bg = torch.ones_like(target['background'])
            for layer in display:
                if 'latlong' in layer and layer['latlong']:
                    assert False
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    else:
                        result_dict['light_image'] = lgt.generate_image(FLAGS.display_res)
                        result_dict['light_image'] = util.rgb_to_srgb(result_dict['light_image'] / (1 + result_dict['light_image']))
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if isinstance(layer['relight'], str):
                        layer['relight'] = light.load_env(layer['relight'])
                    buffers = geometry.render(glctx, target, layer['relight'], opt_material)
                    if render_envmap:
                        background = F.grid_sample(layer['relight'].base.permute(2, 0, 1)[None], grid, align_corners=True).permute(0, 2, 3, 1)
                        img = torch.lerp(background, buffers['shaded'][..., 0:3], buffers['shaded'][..., -1:])
                    else:
                        img = buffers['shaded'][..., 0:3]
                    result_dict['relight%d' % relight_cnt] = util.rgb_to_srgb(img)[0]
                    if bbox is not None:
                        result_dict['relight%d' % relight_cnt] = result_dict['relight%d' % relight_cnt][bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    result_image = torch.cat([result_image, result_dict['relight%d' % relight_cnt]], axis=1)

                    result_dict['relight%d_vis' % relight_cnt] = util.rgb_to_srgb(buffers['diffuse_light'][..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight%d_vis' % relight_cnt]], axis=1)

                    relight_cnt += 1
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    if bbox is not None:
                        result_dict[layer['bsdf']] = result_dict[layer['bsdf']][bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    # result_image = torch.cat([result_dict[layer['bsdf']], result_image], axis=0)
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
                elif 'normal' in layer and layer['normal'] and 'nml' in target:
                    normal = (target['nml'][...,0:3][0] + 1.) * 0.5
                    result_dict['ref_nml'] = normal
                    if bbox is not None:
                        normal = normal[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    result_image = torch.cat([result_image, normal], axis=1)
                elif 'diffuse-optix' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf='diffuse-optix')
                    result_dict['diffuse-optix'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                    if bbox is not None:
                        result_dict['diffuse-optix'] = result_dict['diffuse-optix'][bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    result_image = torch.cat([result_image, result_dict['diffuse-optix']], axis=1) 

        return result_image, result_dict

def extract_bbox(msk, hw = None):
    rcids = torch.nonzero(torch.abs(msk-1) < 1e-6)

    minr, minc = torch.min(rcids, dim=0)[0]
    maxr, maxc = torch.max(rcids, dim=0)[0]

    if hw is not None:
        maxr, maxc = maxr + 1, maxc + 1

        pad0 = (hw[0] - (maxr-minr)) // 2
        pad1 = hw[0] - (maxr-minr) - pad0
        minr -= pad0
        maxr += pad1

        pad0 = (hw[1] - (maxc-minc)) // 2
        pad1 = hw[1] - (maxc-minc) - pad0
        minc -= pad0
        maxc += pad1

        return minc, minr, maxc, maxr

    return minc-100, minr-50, maxc+101, maxr+51

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    FLAGS = train_utils.get_flags()

    train_first_frame, train_last_frame = FLAGS.first_frame, FLAGS.last_frame
    FLAGS.first_frame, FLAGS.last_frame = 0, 2000
    glctx = dr.RasterizeGLContext()
    if os.path.isfile(os.path.join(FLAGS.data_dir, 'calibration_full.json')):
        FLAGS.cam_ids_to_use = list(range(FLAGS.cam_num))
        dataset = TestDatasetSynthetic(FLAGS.data_dir, glctx, FLAGS, real=False)
    training_poses = dataset.poses[train_first_frame:train_last_frame]
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate)

    denoiser = None
    if FLAGS.mc:
        denoiser = BilateralDenoiser().cuda()

    light_usage = 'pbr' if FLAGS.mc is False else 'pbr-optix'
    lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5, usage=light_usage)
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
    mat = train_utils.initial_guess_material(geometry, True, FLAGS)
    # import ipdb; ipdb.set_trace()

    model_path = os.path.join(FLAGS.out_dir, 'model.pt')
    if not os.path.exists(model_path):
        print('No model.pt, using best_model.pt!')
        model_path = os.path.join(FLAGS.out_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        print('No best_model.pt, using latest_model.pt!')
        model_path = os.path.join(FLAGS.out_dir, 'latest_model.pt')
    print(model_path)
    model = torch.load(model_path)
    lgt.load_state_dict(model['lighting'])
    geometry.load_state_dict(model['geometry'])
    mat.load_state_dict(model['material'])
    geometry.buildMesh(mat)

    if FLAGS.posmap_update_interval > 0:
        geometry.update_posmap(glctx)

    save_dir = os.path.join(FLAGS.out_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    start_frame, end_frame, freq = 0, 100, 21
    test = False
    export_img = False
    export_per_img = True
    export_video = False
    export_mesh = False
    relighting = True
    cam_ids = [12, 15, 19]

    fps = 25 // freq

    if relighting is True:
        synthetic_human_pp_dir = ''
        relight = [{"relight": light.load_env(synthetic_human_pp_dir + 'lighting/16x32/gym_entrance.hdr', usage=light_usage)},
                   {"relight": light.load_env(synthetic_human_pp_dir + 'lighting/16x32/olat0000-0027.hdr', usage=light_usage)},
                   {"relight": light.load_env(synthetic_human_pp_dir + 'lighting/16x32/olat0004-0017.hdr', usage=light_usage)},
                   {"relight": light.load_env(synthetic_human_pp_dir + 'lighting/16x32/olat0004-0019.hdr', usage=light_usage)},
                   {"relight": light.load_env(synthetic_human_pp_dir + 'lighting/16x32/peppermint_powerplant_blue.hdr', usage=light_usage)},
                   {"relight": light.load_env(synthetic_human_pp_dir + 'lighting/16x32/shanghai_bund.hdr', usage=light_usage)}]
        # relight = relight[:1] + relight[-1:]

        for rel in relight:
            rel["relight"].convert_for_synthetic()
    else:
        relight = None

    if test is True:
        geometry.forward_deformer.compute_pca(training_poses.cuda())

    if export_per_img:
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)

    import tqdm, cv2
    import imageio
    for cid in cam_ids:

        # prepare canvas
        fid = 0
        target = dataset.collate([dataset[fid * len(dataset.cam_ids_to_use) + cid]])
        target = train_utils.prepare_batch(target, FLAGS.background)
        # bbox = extract_bbox(target['img'][0, ..., -1], hw=(900, 500)) if export_per_img is False else None
        bbox = None
        print("cam %d: " % cid, bbox)
        # bbox = None

        result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, FLAGS, bbox=bbox, relight=relight, denoiser=denoiser)
        canvas = result_image.detach().cpu().numpy()
        
        if export_video:
            videoWriter = cv2.VideoWriter(os.path.join(save_dir, 'video_cam%02d.mp4' % cid), cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas.shape[1], canvas.shape[0]))
        for fid in tqdm.tqdm(range(start_frame, end_frame+1, freq)):
        
            target = dataset.collate([dataset[fid * len(dataset.cam_ids_to_use) + cid]])
            target = train_utils.prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, FLAGS, bbox=bbox, relight=relight, denoiser=denoiser)
            np_result_image = torch.clamp(result_image, 0, 1).detach().cpu().numpy()
                
            if export_img:
                util.save_image(os.path.join(save_dir, 'img_frame%04d_cam%02d.png' % (fid, cid)), np_result_image)
            if export_per_img:
                for k in result_dict.keys():
                    if 'normal' not in k:
                        np_img = util.srgb_to_rgb(result_dict[k]).detach().cpu().numpy()
                        imageio.imwrite(save_dir + '/images/' + ('img_frame%04d_cam%02d_%s.exr' % (fid, cid, k)), np_img)
                        if 'kd' in k:
                            np.save(save_dir + '/images/' + ('img_frame%04d_cam%02d_%s.npy' % (fid, cid, k)), np_img)
                    else:
                        np_img = torch.clamp(result_dict[k], 0, 1).detach().cpu().numpy()
                        util.save_image(save_dir + '/images/' + ('img_frame%04d_cam%02d_%s.png' % (fid, cid, k)), np_img)
            if export_video:
                np_result_image = (np_result_image[..., [2, 1, 0]] * 255).astype(np.uint8)
                videoWriter.write(np_result_image)

            if cid == cam_ids[0] and export_mesh is True:
                val_mesh = train_utils.xatlas_uvmap(glctx, geometry, mat, FLAGS, pose=target)
                os.makedirs(os.path.join(save_dir, "frame%04d/" % fid), exist_ok=True)
                obj.write_obj(os.path.join(save_dir, "frame%04d/" % fid), val_mesh)
        
        if export_video:
            videoWriter.release()
