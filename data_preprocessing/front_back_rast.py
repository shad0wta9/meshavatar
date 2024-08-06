import pickle
import open3d as o3d
import numpy as np
import tqdm
import time

from scipy.spatial import cKDTree as KDTree
import sys

with open('../smpl_models/smplx_10/SMPLX_NEUTRAL.pkl', 'rb') as f:
    model = pickle.load(f, encoding='latin1')

# import ipdb; ipdb.set_trace()
pnts = model['v_template']
faces = model['f']
skinning_weights = model['weights']

dataset_name = sys.argv[-1]
cpose_smpl_file = 'data/data_templates/%s/%s_minimal_cpose.obj' % (dataset_name, dataset_name)
npz_fpath = 'data/data_templates/%s/fb_map.npz' % dataset_name
cpose_mesh = o3d.io.read_triangle_mesh(cpose_smpl_file)
cpose_mesh.compute_vertex_normals()
pnts = np.asarray(cpose_mesh.vertices)
vns = np.asarray(cpose_mesh.vertex_normals)
faces = np.asarray(cpose_mesh.triangles).astype(np.int32)
skinning_weights_file = 'data/data_templates/%s/%s_lbs_weights.txt' % (dataset_name, dataset_name)
skinning_weights = np.loadtxt(skinning_weights_file)

for i in range(3):
    print(pnts[:, i].max(), pnts[:, i].min())
print(faces.shape)

face_verts = pnts[faces]
face_vns = vns[faces]
face_weights = skinning_weights[faces]

h, w = 512, 512
k = 10 # some problems on the 'eyes' and 'nose', dismiss here
x, y = np.linspace(-1, 1, h), np.linspace(-1.4, 0.6, w)
x, y = np.meshgrid(x, y)

x, y = x.reshape(-1), y.reshape(-1)


mid_points = face_verts.sum(axis=-2) / 3.0
kdtree = KDTree(mid_points[:, :2])

st_time = time.time()
dist, face_idx = kdtree.query(np.stack([x, y], axis=-1), k)
print('kdtree: %.6fs' % (time.time() - st_time))

selected_face_verts = face_verts[face_idx]
v1, v2, v3 = selected_face_verts[..., 0, :], selected_face_verts[..., 1, :] - selected_face_verts[..., 0, :], selected_face_verts[..., 2, :] - selected_face_verts[..., 0, :]

selected_face_vns = face_vns[face_idx]
vn1, vn2, vn3 = selected_face_vns[..., 0, :], selected_face_vns[..., 1, :] - selected_face_vns[..., 0, :], selected_face_vns[..., 2, :] - selected_face_vns[..., 0, :] 

selected_face_weights = face_weights[face_idx]
w1, w2, w3 = selected_face_weights[..., 0, :], selected_face_weights[..., 1, :] - selected_face_weights[..., 0, :], selected_face_weights[..., 2, :] - selected_face_weights[..., 0, :]

st_time = time.time()
mat = np.stack([v2[..., 0], v3[..., 0], v2[..., 1], v3[..., 1]], axis=-1).reshape(h*w, k, 2, 2)
vec = np.stack([x[:, None] - v1[..., 0], y[:, None] - v1[..., 1]], axis=-1) # 512x512, 5, 2
mat_inv = np.linalg.inv(mat)
alpha_beta = (mat_inv @ vec[..., None])[..., 0]
alpha, beta = alpha_beta[..., [0]], alpha_beta[..., [1]]
print('solving alpha & beta: %.6fs' % (time.time() - st_time))

traced_pnts = v1 + v2 * alpha + v3 * beta # 512x512, 5, 3
traced_vns = vn1 + vn2 * alpha + vn3 * beta
traced_weights = w1 + w2 * alpha + w3 * beta
msk = (alpha >= 0) * (alpha <= 1) * (beta >= 0) * (beta <= 1) * (alpha + beta <= 1)
msk = msk.squeeze(-1)
traced_pnts[~msk, 2] = -np.inf
traced_msk = msk.sum(-1) > 0

argz = np.argmax(traced_pnts[..., 2], axis=1)
ftraced_pnt = np.take_along_axis(traced_pnts, argz[:, None, None], axis=1).squeeze(1)
ftraced_vn = np.take_along_axis(traced_vns, argz[:, None, None], axis=1).squeeze(1)
ftraced_weight = np.take_along_axis(traced_weights, argz[:, None, None], axis=1).squeeze(1)
ftraced_pnt[~traced_msk] = 0
ftraced_vn[~traced_msk] = 0
ftraced_weight[~traced_msk] = 0


traced_pnts[~msk, 2] = np.inf

argz = np.argmin(traced_pnts[..., 2], axis=1)
btraced_pnt = np.take_along_axis(traced_pnts, argz[:, None, None], axis=1).squeeze(1)
btraced_vn = np.take_along_axis(traced_vns, argz[:, None, None], axis=1).squeeze(1)
btraced_weight = np.take_along_axis(traced_weights, argz[:, None, None], axis=1).squeeze(1)
btraced_pnt[~traced_msk] = 0
btraced_vn[~traced_msk] = 0
btraced_weight[~traced_msk] = 0

traced_msk = traced_msk.reshape(w, h)
ftraced_pnt = ftraced_pnt.reshape(w, h, 3)
btraced_pnt = btraced_pnt.reshape(w, h, 3)
ftraced_vn = ftraced_vn.reshape(w, h, 3)
btraced_vn = btraced_vn.reshape(w, h, 3)
ftraced_weight = ftraced_weight.reshape(w, h, 55)
btraced_weight = btraced_weight.reshape(w, h, 55)

np.savez(npz_fpath, fpnts=ftraced_pnt, fvns=ftraced_vn, fweights=ftraced_weight,
                    bpnts=btraced_pnt, bvns=btraced_vn, bweights=btraced_weight)

# pcd = o3d.geometry.PointCloud()

# points = np.concatenate([ftraced_pnt[traced_msk], btraced_pnt[traced_msk]], axis=0)
# colors = (np.concatenate([ftraced_vn[traced_msk], btraced_vn[traced_msk]], axis=0) + 1.) / 2.

# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.io.write_point_cloud("./debug/rast.ply", pcd)

# npz = np.load('output/rast.npz')
# pnts, weights = npz['pnts'], npz['weights']
# msk = weights.sum(-1) == 0
# pnts, weights = pnts[~msk], weights[~msk]

# import smplx
# import torch
# from smpl_np import SMPLModel as SmplNumpy

# def load_smplx_params_from_easymocap(smpl_param_ckpt_fpath, body_only=False):
#     d = torch.load(smpl_param_ckpt_fpath, map_location='cpu')

#     betas = d['beta'].detach()
#     # orient = d['orient']
#     # body_pose = d['body_pose']
#     # jaw_pose = d['jaw_pose']
#     # leye_pose = torch.zeros_like(jaw_pose)
#     # reye_pose = torch.zeros_like(jaw_pose)
#     # left_hand_pose = d['lhand']
#     # right_hand_pose = d['rhand']
#     # poses = torch.cat([orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose], dim=1)
#     poses = d['poses'][:, 0].detach()
#     if body_only:
#         poses[:, 66:] *= 0.0
#     trans = d['trans'][:, 0].detach()
#     rots = d['rots'][:, 0].detach()
#     # print(betas.shape, poses.shape, trans.shape, rots.shape)
#     return betas, poses, trans, rots

# smpl_params_fpath = r'D:\workspace\avatar\data\data0424\20230424-142858\data.5\whole.pt'
# smpl_file_path = r'D:\workspace\avatar\smplx\models\smplx\SMPLX_NEUTRAL.npz'
# betas, poses, trans, rots = load_smplx_params_from_easymocap(smpl_params_fpath, body_only=False)
# smplx_np = SmplNumpy(smpl_file_path)

# frame_id=482
# smplx_verts = smplx_np.set_params(
#     pose=poses[frame_id].detach().cpu().numpy(),
#     beta=betas.detach().cpu().numpy(),
#     rot=rots[frame_id].detach().cpu().numpy(),
#     trans=trans[frame_id].detach().cpu().numpy()
# )
# joints = smplx_np.J_regressor.dot(smplx_verts)
# lines = model['kintree_table'].T

# pcd = o3d.geometry.PointCloud()

# pcd.points = o3d.utility.Vector3dVector(pnts)
# # pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.io.write_point_cloud("./output/rast.ply", pcd)

# transform = (weights @ smplx_np.G.reshape(-1, 16)).reshape(-1, 4, 4)
# pnts_one = np.concatenate([pnts, np.ones_like(pnts[..., [0]])], axis=-1)
# posed_pnts = transform @ pnts_one[..., None]
# posed_pnts = posed_pnts[..., :3, 0]

# pcd = o3d.geometry.PointCloud()

# pcd.points = o3d.utility.Vector3dVector(posed_pnts)
# # pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.io.write_point_cloud("./output/rast_posed.ply", pcd)