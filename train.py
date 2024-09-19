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
from dataset.dataset_smpl import DatasetSMPL
# from dataset.dataset_actorshq import DatasetActorsHQ
from dataset.dataset_synthetic import DatasetSynthetic
from dataset.block_sampler import BlockSampler

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

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, denoiser=None):
    result_dict = {}
    with torch.no_grad():
        if not FLAGS.mc:
            lgt.build_mips()
            if FLAGS.camera_space_light:
                lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material, denoiser)

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        if FLAGS.display is not None:
            white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    else:
                        result_dict['light_image'] = lgt.generate_image(FLAGS.display_res)
                        result_dict['light_image'] = util.rgb_to_srgb(result_dict['light_image'] / (1 + result_dict['light_image']))
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.render(glctx, target, layer['relight'], opt_material)
                    result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
                elif 'normal' in layer and layer['normal'] and 'nml' in target:
                    normal = (target['nml'][...,0:3][0] + 3.) * 0.25
                    result_dict['ref_nml'] = normal
                    result_image = torch.cat([result_image, normal], axis=1)
   
        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            target = train_utils.prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        if not self.FLAGS.mc and not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.mat_params = list(self.material.parameters())
        self.lgt_params = list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

        if self.FLAGS.perc_patch_size > 0:
            self.perc_loss_fn = train_utils.PercLoss(self.FLAGS.perc_patch_size)
        else:
            self.perc_loss_fn = None

    def forward(self, target, it, denoiser):
        if not self.FLAGS.mc and self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, it, denoiser, self.perc_loss_fn)

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    denoiser,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    optimize_light=True,
    optimize_geometry=True
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_lgt = learning_rate[2] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate * 3.0

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter + FLAGS.start_iter - warmup_iter)*(1./min(FLAGS.iter, 50000)))) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = train_utils.createLoss(FLAGS)

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, FLAGS)
    betas = (0.9, 0.999)

    if FLAGS.multi_gpu: 
        # Multi GPU training mode
        import apex
        from apex.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp)
        trainer.train()
        if optimize_geometry:
            optimizer_mesh = apex.optimizers.FusedAdam(trainer_noddp.geo_params, lr=learning_rate_pos, betas=betas)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        if optimize_light:
            optimizer_light = apex.optimizers.FusedAdam(trainer_noddp.lgt_params, lr=learning_rate_lgt, betas=betas)
            scheduler_light = torch.optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        optimizer = apex.optimizers.FusedAdam(trainer_noddp.mat_params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 
    else:
        # Single GPU training mode
        trainer = trainer_noddp
        if optimize_geometry:
            optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos, betas=betas)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        if optimize_light:
            optimizer_light = torch.optim.Adam(trainer_noddp.lgt_params, lr=learning_rate_lgt, betas=betas)
            scheduler_light = torch.optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        optimizer = torch.optim.Adam(trainer_noddp.mat_params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    losses_vec = None
    iter_dur_vec = []
    save_freq, best_loss = 2000, 1e3

    train_sampler = BlockSampler(dataset_train, block_size=FLAGS.batch)
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, num_workers=FLAGS.num_workers, sampler=train_sampler)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):

        if it + FLAGS.start_iter > FLAGS.iter:
            break

        # Mix randomized background into dataset image
        target = train_utils.prepare_batch(target, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                result_image, result_dict = validate_itr(glctx, train_utils.prepare_batch(next(v_it), FLAGS.background), geometry, opt_material, lgt, FLAGS, denoiser)
                np_result_image = result_image.detach().cpu().numpy()
                if display_image:
                    util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                if save_image:
                    util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)
                    img_cnt = img_cnt+1

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()
        if optimize_light:
            optimizer_light.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        if optimize_light and FLAGS.mc:
            lgt.update_pdf()
        
        losses = trainer(target, it + FLAGS.start_iter, denoiser)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = 0
        for k in losses.keys():
            total_loss += losses[k]

        if losses_vec is None:
            # register losses
            losses_vec = {}
            for k in losses.keys():
                losses_vec[k] = []

        for k in losses.keys():
            losses_vec[k].append(losses[k].item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
            lgt.base.grad *= 64
        if 'kd_ks_normal' in opt_material:
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        log_grad = torch.log2(geometry.mesh_verts.grad.max())

        if torch.abs(log_grad - log_grad.long()) < 1e-6:
            print('flaw gradient! skip optimizer.step()!')
        else:
            optimizer.step()
            scheduler.step()

            if optimize_light:
                optimizer_light.step()
                scheduler_light.step()

            if optimize_geometry:
                torch.nn.utils.clip_grad_norm_(trainer_noddp.geo_params, 1e-2)
                optimizer_mesh.step()
                scheduler_mesh.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.01)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if (it + FLAGS.start_iter) % log_interval == 0 and FLAGS.local_rank == 0:
            loss_avg = ''
            for k in losses_vec.keys():
                loss_avg += ' %s=%.6f,' % (k, np.mean(losses_vec[k][-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-FLAGS.start_iter-it)*iter_dur_avg
            print("iter=%5d,%s lr=%.5f, time=%.1f ms, rem=%s" % 
                (it + FLAGS.start_iter, loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

        if (it + FLAGS.start_iter) % save_freq == 0 and FLAGS.local_rank == 0:
            cur_loss = np.mean(losses_vec['img_loss'][-save_freq:])
            if cur_loss < best_loss:
                torch.save({'geometry': geometry.state_dict(),
                            'material': opt_material.state_dict(),
                            'lighting': lgt.state_dict()}, os.path.join(FLAGS.out_dir, 'best_model.pt'))
                
        if FLAGS.posmap_update_interval > 0 and it + FLAGS.start_iter > warmup_iter and \
            (it + FLAGS.start_iter) % FLAGS.posmap_update_interval == 0 and FLAGS.local_rank == 0:
            geometry.update_posmap(glctx)

    return geometry, opt_material

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    FLAGS = train_utils.get_flags()

    assert FLAGS.pbr or not FLAGS.mc

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    os.system('cp %s %s' % (FLAGS.config, os.path.join(FLAGS.out_dir, 'exp.json')))

    glctx = dr.RasterizeGLContext()

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    if os.path.isfile(os.path.join(FLAGS.data_dir, 'calibration_full.json')):
        if os.path.isfile(os.path.join(FLAGS.data_dir, 'smpl_params.npz')):
            dataset_train    = DatasetSMPL(FLAGS.data_dir, glctx, FLAGS, validate=False)
            dataset_validate = DatasetSMPL(FLAGS.data_dir, glctx, FLAGS, validate=True)
        elif os.path.isfile(os.path.join(FLAGS.data_dir, 'merged.pkl')):
            dataset_train    = DatasetSynthetic(FLAGS.data_dir, glctx, FLAGS, validate=False)
            dataset_validate = DatasetSynthetic(FLAGS.data_dir, glctx, FLAGS, validate=True)
        else:
            assert False, "No SMPL-X poses in the data directory!"
    # if os.path.isfile(os.path.join(FLAGS.data_dir, 'calibration.csv')):
    #     dataset_train    = DatasetActorsHQ(FLAGS.data_dir, glctx, FLAGS, validate=False)
    #     dataset_validate = DatasetActorsHQ(FLAGS.data_dir, glctx, FLAGS, validate=True)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    
    light_usage = 'pbr' if FLAGS.mc is False else 'pbr-optix'
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5, usage=light_usage)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale, usage=light_usage)

    # ==============================================================================================
    #  Setup denoiser
    # ==============================================================================================

    denoiser = None
    if FLAGS.mc:
        denoiser = BilateralDenoiser().cuda()

    # ==============================================================================================
    #  Use DMtets to create geometry
    # ==============================================================================================

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)

    # Setup textures, make initial guess from reference if possible
    mat = train_utils.initial_guess_material(geometry, True, FLAGS)

    if FLAGS.resume is not None:
        model = torch.load(FLAGS.resume)
        lgt.load_state_dict(model['lighting'])
        geometry.load_state_dict(model['geometry'], strict=False)
        mat.load_state_dict(model['material'])

        # geometry.forward_deformer.deformation_net.reinitialize()
        # geometry.subdivide()

    # Run optimization
    geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, 
                    FLAGS, denoiser, pass_idx=0, pass_name="dmtet_pass1", warmup_iter=FLAGS.warmup_iter, 
                    log_interval=FLAGS.log_interval, optimize_light=FLAGS.learn_light)

    if FLAGS.local_rank == 0 and FLAGS.validate:
        validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS)

    if FLAGS.local_rank == 0:
        torch.save({'geometry': geometry.state_dict(),
                    'material': mat.state_dict(),
                    'lighting': lgt.state_dict()}, os.path.join(FLAGS.out_dir, 'model.pt'))

    # Create textured mesh from result
    base_mesh = train_utils.xatlas_uvmap(glctx, geometry, mat, FLAGS)

    if FLAGS.local_rank == 0:
        # Dump mesh for debugging.
        os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

#----------------------------------------------------------------------------
