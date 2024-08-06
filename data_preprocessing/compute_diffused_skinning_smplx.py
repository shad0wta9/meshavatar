import os, sys
from os.path import join
import math
import numpy as np
import trimesh
import torch
import yaml
from glob import glob

import array
import tqdm

import smplx
from smplx.lbs import vertices2joints
from .lbs import lbs
import pickle

if __name__ == '__main__':

    ### NOTE useful options
    # opt = {}
    # with open(join('configs', 'common.yaml'), 'r') as common_opt_f:
    #     common_opt = yaml.safe_load(common_opt_f)
    #     opt.update(common_opt)
    # with open(join('configs', f'step1_smplx.yaml'), 'r') as step_opt_f:
    #     step_opt = yaml.safe_load(step_opt_f)
    #     opt.update(step_opt)

    data_templates_path = 'data/data_templates'
    subject = sys.argv[-2]
    smpl_model_path = sys.argv[-1]
    num_joints = 55
    leg_angle = 15.0
    point_interpolant_exe = 'PoissonRecon/Bin/Linux/PointInterpolant'
    skinning_grid_depth = 8
    lbs_surf_grad_exe = 'data_preprocessing/diffused_skinning/lbs_surf_grad'
    ask_before_os_system = False

    tmp_folder_constraints = '/tmp/data_tmp_constraints'
    tmp_folder_skinning_grid = '/tmp/data_tmp_skinning_grid'

    specify_hands = False
    gender = 'neutral'

    if not os.path.exists(tmp_folder_constraints):
        os.makedirs(tmp_folder_constraints)
    if not os.path.exists(tmp_folder_skinning_grid):
        os.makedirs(tmp_folder_skinning_grid)



    # ### NOTE get a canonical-pose SMPL template
    smpl_tpose_mesh_path = join(data_templates_path, subject, f'{subject}_minimal_tpose.ply')
    # with open(join(data_templates_path, 'gender_list.yaml') ,'r') as f:
    #     gender = yaml.safe_load(f)[subject]

    cpose_param = torch.zeros(1, 72)
    cpose_param[:, 5] =  leg_angle / 180 * math.pi
    cpose_param[:, 8] = -leg_angle / 180 * math.pi

    tpose_mesh = trimesh.load(smpl_tpose_mesh_path, process=False)
    smpl_model = smplx.create(smpl_model_path, model_type='smpl', gender=gender)
    smplx_model = smplx.create(smpl_model_path, model_type='smplx', gender=gender)

    tpose_verts = torch.from_numpy(tpose_mesh.vertices).float()[None]
    tpose_joints = vertices2joints(smpl_model.J_regressor, tpose_verts)

    out = lbs(tpose_verts, tpose_joints, cpose_param, smpl_model.parents, smpl_model.lbs_weights[None])
    cpose_verts = out['v_posed'][0].cpu().numpy()

    # np.savetxt('cano_data_grad_constraints.xyz', out['v_posed'][0], fmt="%.8f")
    cpose_mesh = trimesh.Trimesh(cpose_verts, smpl_model.faces, process=False)
    cpose_mesh.export(join(data_templates_path, subject, f'{subject}_minimal_cpose.obj'))
    cpose_mesh.export(join(data_templates_path, subject, f'{subject}_minimal_cpose.ply'))

    with open(join(smpl_model_path, 'model_transfer/smplx_to_smpl.pkl'), 'rb') as f:
        smplx2smpl_mat = pickle.load(f, encoding='latin1')['matrix']
    np.savetxt(join(data_templates_path, subject, f'{subject}_lbs_weights.txt'), smplx2smpl_mat @ smplx_model.lbs_weights.numpy(), fmt="%.8f")

    ### NOTE compute the along-surface gradients of skinning
    cmd = f'{lbs_surf_grad_exe} ' + \
          f'{join(data_templates_path, subject, subject + "_minimal_cpose.obj")} ' + \
          f'{join(data_templates_path, subject, subject + "_lbs_weights.txt")} ' + \
          f'{join(data_templates_path, subject, subject + "_cpose_lbs_grads.txt")} '
    
    if ask_before_os_system:
        go_on = input(f'\n[WILL EXECUTE with os.system] {cmd}\nContinue? (y/n)')
    else:
        go_on = 'y'
    if go_on == 'y':
        os.system(cmd)

    ### NOTE reorganize data
    data = np.loadtxt(join(data_templates_path, subject, subject + "_cpose_lbs_grads.txt"))

    position = data[:, 0:3]
    normals = data[:, 3:6]
    tx = data[:, 6:9]
    ty = data[:, 9:12]
    lbs_w = data[:, 12:12+num_joints]
    lbs_tx = data[:, 12+num_joints:12+2*num_joints]
    lbs_ty = data[:, 12+2*num_joints:12+3*num_joints]

    if not os.path.exists(tmp_folder_constraints):
        os.mkdir(tmp_folder_constraints)

    for jid in tqdm.tqdm(range(num_joints)):
        out_fn_grad = os.path.join(tmp_folder_constraints, f"cano_data_lbs_grad_{jid:02d}.xyz")
        out_fn_val = os.path.join(tmp_folder_constraints, f"cano_data_lbs_val_{jid:02d}.xyz")

        grad_field = lbs_tx[:, jid:jid+1] * tx + lbs_ty[:, jid:jid+1] * ty

        out_data_grad = np.concatenate([position, grad_field], 1)
        out_data_val = np.concatenate([position, lbs_w[:, jid:jid+1]], 1)
        np.savetxt(out_fn_grad, out_data_grad, fmt="%.8f")
        np.savetxt(out_fn_val, out_data_val, fmt="%.8f")

    if not os.path.exists(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_float32.npy')):

        ### NOTE solve for the diffused skinning fields
        for jid in range(num_joints):
            cmd = f'{point_interpolant_exe} ' + \
                f'--inValues {join(tmp_folder_constraints, f"cano_data_lbs_val_{jid:02d}.xyz")} ' + \
                f'--inGradients {join(tmp_folder_constraints, f"cano_data_lbs_grad_{jid:02d}.xyz")} ' + \
                f'--gradientWeight 0.05 --dim 3 --verbose ' + \
                f'--grid {join(tmp_folder_skinning_grid, f"grid_{jid:02d}.grd")} ' + \
                f'--depth {skinning_grid_depth} '
            
            if ask_before_os_system:
                go_on = input(f'\n[WILL EXECUTE with os.system] {cmd}\nContinue? (y/n)')
            else:
                go_on = 'y'
            if go_on == 'y':
                os.system(cmd)

        ### NOTE concatenate all grids
        fn_list = sorted(list(glob(join(tmp_folder_skinning_grid, 'grid_*.grd'))))
        print(fn_list)

        grids = []
        for fn in fn_list:
            with open(fn, 'rb') as f:
                bytes = f.read()
            grid_res = 2 ** skinning_grid_depth
            grid_header_len = len(bytes) - grid_res ** 3 * 8
            grid_np = np.array(array.array('d', bytes[grid_header_len:])).reshape(256, 256, 256)
            grids.append(grid_np)


        grids_all = np.stack(grids, 0)
        grids_all = np.clip(grids_all, 0.0, 1.0)
        grids_all = grids_all / grids_all.sum(0)[None]
        np.save(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_float32.npy'), grids_all.astype(np.float32))

    if not specify_hands:
        exit()

    ### HAND
    hand_info_path = 'data/diffused_skinning'
    smplx_lhand_to_mano_rhand_data = np.load(join(hand_info_path, 'smplx_lhand_to_mano_rhand.npz'), allow_pickle=True)
    smplx_rhand_to_mano_rhand_data = np.load(join(hand_info_path, 'smplx_rhand_to_mano_rhand.npz'), allow_pickle=True)
    smpl_lhand_vert_id = np.copy(smplx_lhand_to_mano_rhand_data['smpl_vert_id_to_mano'])
    smpl_rhand_vert_id = np.copy(smplx_rhand_to_mano_rhand_data['smpl_vert_id_to_mano'])

    lhand, rhand = np.zeros(10475), np.zeros(10475)
    lhand[smpl_lhand_vert_id] = 1
    rhand[smpl_rhand_vert_id] = 1
    lhand_id = np.where((smplx2smpl_mat @ lhand[:, None])[:, 0] > 0)[0]
    rhand_id = np.where((smplx2smpl_mat @ rhand[:, None])[:, 0] > 0)[0]

    for jid in tqdm.tqdm(range(num_joints)):
        out_fn_grad_lhand = os.path.join(tmp_folder_constraints, f"cano_data_lbs_grad_lhand_{jid:02d}.xyz")
        out_fn_val_lhand = os.path.join(tmp_folder_constraints, f"cano_data_lbs_val_lhand_{jid:02d}.xyz")
        out_fn_grad_rhand = os.path.join(tmp_folder_constraints, f"cano_data_lbs_grad_rhand_{jid:02d}.xyz")
        out_fn_val_rhand = os.path.join(tmp_folder_constraints, f"cano_data_lbs_val_rhand_{jid:02d}.xyz")

        grad_field = lbs_tx[:, jid:jid+1] * tx + lbs_ty[:, jid:jid+1] * ty

        out_data_grad = np.concatenate([position, grad_field], 1)
        out_data_val = np.concatenate([position, lbs_w[:, jid:jid+1]], 1)

        np.savetxt(out_fn_grad_lhand, out_data_grad[lhand_id], fmt="%.8f")
        np.savetxt(out_fn_val_lhand, out_data_val[lhand_id], fmt="%.8f")
        np.savetxt(out_fn_grad_rhand, out_data_grad[rhand_id], fmt="%.8f")
        np.savetxt(out_fn_val_rhand, out_data_val[rhand_id], fmt="%.8f")

    hand_skinning_grid_depth = skinning_grid_depth - 2

    if not os.path.exists(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_lhand_float32.npy')):
        ### LEFT HAND
        ### NOTE solve for the diffused skinning fields
        for jid in range(num_joints):
            cmd = f'{point_interpolant_exe} ' + \
                f'--inValues {join(tmp_folder_constraints, f"cano_data_lbs_val_lhand_{jid:02d}.xyz")} ' + \
                f'--inGradients {join(tmp_folder_constraints, f"cano_data_lbs_grad_lhand_{jid:02d}.xyz")} ' + \
                f'--gradientWeight 0.05 --dim 3 --verbose ' + \
                f'--grid {join(tmp_folder_skinning_grid, f"grid_{jid:02d}.grd")} ' + \
                f'--depth {hand_skinning_grid_depth} '
            
            if ask_before_os_system:
                go_on = input(f'\n[WILL EXECUTE with os.system] {cmd}\nContinue? (y/n)')
            else:
                go_on = 'y'
            if go_on == 'y':
                os.system(cmd)

        # ### NOTE concatenate all grids
        fn_list = sorted(list(glob(join(tmp_folder_skinning_grid, 'grid_*.grd'))))
        print(fn_list)

        grids = []
        for fn in fn_list:
            with open(fn, 'rb') as f:
                bytes = f.read()
            grid_res = 2 ** hand_skinning_grid_depth
            grid_header_len = len(bytes) - grid_res ** 3 * 8
            grid_np = np.array(array.array('d', bytes[grid_header_len:])).reshape(2 ** hand_skinning_grid_depth, 2 ** hand_skinning_grid_depth, 2 ** hand_skinning_grid_depth)
            grids.append(grid_np)

        grids_all = np.stack(grids, 0)
        grids_all = np.clip(grids_all, 0.0, 1.0)
        grids_all = grids_all / grids_all.sum(0)[None]
        np.save(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_lhand_float32.npy'), grids_all.astype(np.float32))
    
    if not os.path.exists(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_rhand_float32.npy')):
        ### RIGHT HAND
        ### NOTE solve for the diffused skinning fields
        for jid in range(num_joints):
            cmd = f'{point_interpolant_exe} ' + \
                f'--inValues {join(tmp_folder_constraints, f"cano_data_lbs_val_rhand_{jid:02d}.xyz")} ' + \
                f'--inGradients {join(tmp_folder_constraints, f"cano_data_lbs_grad_rhand_{jid:02d}.xyz")} ' + \
                f'--gradientWeight 0.05 --dim 3 --verbose ' + \
                f'--grid {join(tmp_folder_skinning_grid, f"grid_{jid:02d}.grd")} ' + \
                f'--depth {hand_skinning_grid_depth} '
            
            if ask_before_os_system:
                go_on = input(f'\n[WILL EXECUTE with os.system] {cmd}\nContinue? (y/n)')
            else:
                go_on = 'y'
            if go_on == 'y':
                os.system(cmd)

        ### NOTE concatenate all grids
        fn_list = sorted(list(glob(join(tmp_folder_skinning_grid, 'grid_*.grd'))))
        print(fn_list)

        grids = []
        for fn in fn_list:
            with open(fn, 'rb') as f:
                bytes = f.read()
            grid_res = 2 ** hand_skinning_grid_depth
            grid_header_len = len(bytes) - grid_res ** 3 * 8
            grid_np = np.array(array.array('d', bytes[grid_header_len:])).reshape(2 ** hand_skinning_grid_depth, 2 ** hand_skinning_grid_depth, 2 ** hand_skinning_grid_depth)
            grids.append(grid_np)

        grids_all = np.stack(grids, 0)
        grids_all = np.clip(grids_all, 0.0, 1.0)
        grids_all = grids_all / grids_all.sum(0)[None]
        np.save(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_rhand_float32.npy'), grids_all.astype(np.float32))
