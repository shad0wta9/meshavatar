import os, sys

import pickle
import open3d as o3d
import numpy as np
import tqdm
import torch
from .smpl_np import SMPLModel as SmplNumpy

dataset_name, smpl_dir = sys.argv[-2], sys.argv[-1]

with open(os.path.join(smpl_dir, 'smplx_10/SMPLX_NEUTRAL.pkl'), 'rb') as f:
    model = pickle.load(f, encoding='latin1')

gender = 'NEUTRAL'

# gender = 'FEMALE'
smpl_file_path = os.path.join(smpl_dir, 'smplx/SMPLX_%s.npz' % gender)
smplx_np = SmplNumpy(smpl_file_path)

# smpl_param_path = '/data/yushuo/data/zzr_fullbody_20221130_01_2k/data.5/whole.pt'
# betas, poses, trans = load_smplx_params(smpl_param_path, body_only=False)

smplx_verts = smplx_np.set_params(
    pose= np.zeros((1, 165)),
    beta= np.load('/tmp/beta.npy'),
    rot=None,
    trans=None
)
joints = smplx_np.J_regressor.dot(smplx_verts)
lines = model['kintree_table'].T


with open(os.path.join(smpl_dir, 'model_transfer/smplx_to_smpl.pkl'), 'rb') as f:
    transfer = pickle.load(f, encoding='latin1')

with open(os.path.join(smpl_dir, 'smpl/SMPL_NEUTRAL.pkl'), 'rb') as f:
    smpl_model = pickle.load(f, encoding='latin1')

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(transfer['matrix'] @ smplx_verts)
mesh.triangles = o3d.utility.Vector3iVector(smpl_model['f'])
o3d.io.write_triangle_mesh("data/data_templates/%s/%s_minimal_tpose.ply" % 
                            (dataset_name, dataset_name), mesh)
