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
import os
import json
import scipy

from render import util
from render import mesh
from render import render
from render import light
from render import obj

from .dataset import Dataset

import cv2
import imageio

def cam2world(pts, cam_rotmat, cam_transl):
    return np.matmul(pts - cam_transl.reshape(1, 3), cam_rotmat)

def load_smplx_params(smpl_param_ckpt_fpath, body_only=True):
    npz = np.load(smpl_param_ckpt_fpath)
    d = {}
    for x in npz.keys():
        d.update({x: torch.from_numpy(npz[x]).float()})

    betas = d['betas']
    orient = d['global_orient']
    body_pose = d['body_pose']
    jaw_pose = d['jaw_pose']
    leye_pose = torch.zeros_like(jaw_pose)
    reye_pose = torch.zeros_like(jaw_pose)
    left_hand_pose = d['left_hand_pose']
    right_hand_pose = d['right_hand_pose']
    poses = torch.cat([orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose], dim=1)
    if body_only:
        poses[:, body_pose.shape[1]:] *= 0.0
    trans = d['transl']
    # logging.info('Loaded pre-trained SMPL parameters from ' + smpl_param_ckpt_fpath)
    return betas, poses, trans

def get_cams(root_dir):
    with open(os.path.join(root_dir, 'calibration_full.json'), 'r') as fp:
        cam_data = json.load(fp)

    return cam_data


def get_cam_ssns(dataset_fpath):
    cam_ssns = []   # sorted according to their spatial position
    with open(os.path.join(dataset_fpath, 'cam_ssns.txt'), 'r') as fp:
        lns = fp.readlines()
    for ln in lns:
        ln = ln.strip().split(' ')
        if len(ln) > 0:
            cam_ssns.append(ln[0])
    return cam_ssns

def projection_matrix_from_intrinsics(w, h, fx, fy, cx, cy, znear, zfar, device=None):
    return torch.tensor([
        [2 * fx / w, 0., 0., 0.],
        [0., -2 * fy / h, 0., 0.],
        [(w - 2 * cx) / w, (h - 2 * cy) / h, -(zfar + znear) / (zfar - znear), -1.],
        [0., 0., -2 * zfar * znear / (zfar - znear), 0.]
    ], dtype=torch.float32, device=device).transpose(0, 1)

def get_frame_id_offset(dataset_path, suffix=''):
    try:
        a = np.loadtxt(os.path.join(dataset_path, 'frame_id_offset' + suffix + '.txt'))
    except:
        return 0
    a = int(a)
    return a

def geodesic_path(t, H0, H1):
    H = H1 @ np.linalg.inv(H0)
    S = scipy.linalg.logm(H)
    Ht = scipy.linalg.expm(t * S) @ H0

    return Ht

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class DatasetSMPL(Dataset):

    def __init__(self, data_dir, glctx, FLAGS, validate=False):
        # Init 
        self.glctx              = glctx
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]

        self.cam_num = FLAGS.cam_num
        self.ratio = FLAGS.ratio
        self.h, self.w = FLAGS.display_res
        self.first_frame, self.last_frame = FLAGS.first_frame, FLAGS.last_frame
        self.cam_ids_to_use = FLAGS.cam_ids_to_use.copy()

        self.data_dir = data_dir
        self.betas, self.poses, self.trans = load_smplx_params(os.path.join(data_dir, 'smpl_params.npz'), body_only=False)
        self.frame_id_offset = get_frame_id_offset(self.data_dir)
        self.rots = None
        self.cam_ssn_list = get_cam_ssns(self.data_dir)
        cam_data = get_cams(self.data_dir)
        FLAGS.beta = self.betas.detach().cpu().numpy()[0] # [10]

        cams = []
        for cam_id in range(self.cam_num):
            cam_ssn = self.cam_ssn_list[cam_id]
            K = np.array(cam_data[cam_ssn]['K']).astype(np.float32).reshape((3, 3))
            D = np.array(cam_data[cam_ssn]['distCoeff']).astype(np.float32).reshape((-1,))
            R = np.array(cam_data[cam_ssn]['R']).astype(np.float32).reshape((3, 3))
            t = np.array(cam_data[cam_ssn]['T']).astype(np.float32).reshape((3,))

            K[0] *= int(self.w*self.ratio) / self.w
            K[1] *= int(self.h*self.ratio) / self.h

            P = np.zeros([3, 4], dtype=np.float32)
            P[:3, :3] = R
            P[:3, 3] = t
            P = np.matmul(K, P)

            mv = np.zeros((4, 4))
            mv[:3, :3] = R
            mv[:3, 3] = t
            mv[3, 3] = 1
            mv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ mv
            mv = torch.from_numpy(mv).float()

            proj_mtx = projection_matrix_from_intrinsics(
                self.w, self.h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
            mvp = proj_mtx @ mv
            campos = torch.linalg.inv(mv)[:3, 3]

            cam = {
                'id': cam_id,
                'ssn': cam_ssn,
                'R': np.float32(R),
                't': np.float32(t),
                'K': np.float32(K),
                'P': np.float32(P),
                'D': np.float32(D),
                'height': np.float32(int(self.h*self.ratio)),
                'width': np.float32(int(self.w*self.ratio)),
                'mv': mv,
                'mvp': mvp,
                'campos': campos
            }
            cams.append(cam)
        self.cams = cams

        if self.FLAGS.local_rank == 0:
            print('The following cameras will be loaded: ')
            print([self.cam_ssn_list[cam_id] for cam_id in self.cam_ids_to_use])

        if os.path.exists(os.path.join(self.data_dir, 'valid.txt')) and self.FLAGS.normal_supervised:
            valid_list = np.loadtxt(os.path.join(self.data_dir, 'valid.txt')) + self.frame_id_offset
            j = 0
            while self.frame_id_offset+self.first_frame > valid_list[j]: j += 1
            self.valid_frame_list = []
            # manual_remove = [[1090, 1200], [1428, 1592]]
            manual_remove = []
            for i in range(self.frame_id_offset+self.first_frame, self.frame_id_offset+self.last_frame+1):
                if i == valid_list[j]:
                    removed = False
                    for interval in manual_remove:
                        if interval[0] <= i and i <= interval[1]:
                            removed = True
                            break
                    if removed is False:
                        self.valid_frame_list.append(i)
                    j += 1
        else:
            self.valid_frame_list = list(range(self.frame_id_offset+self.first_frame, self.frame_id_offset+self.last_frame+1))
        
        if self.FLAGS.local_rank == 0:
            print("Totally %d valid frames" % len(self.valid_frame_list))

    def _rotate_scene(self, itr):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        ang    = (itr / 50) * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp

    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.
        mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation_translation(0.25)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp # Add batch dimension

    @property
    def frame_num(self):
        return len(self.valid_frame_list)
    
    @property
    def view_num(self):
        return len(self.cam_ids_to_use)
    
    @property
    def n_images(self):
        return self.frame_num * self.view_num
    
    def __len__(self):
        # return 50 if self.validate else self.frame_num * self.view_num
        return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch
    
    def get_mask(self, msk_fpath):
        """Follow the same preprocessing as NeuralBody"""

        msk = cv2.imread(msk_fpath, cv2.IMREAD_UNCHANGED)
        if len(msk.shape) > 2:
            msk = msk[:, :, 0]

        thres = 128
        msk_bk = np.copy(msk)
        msk[msk <= thres] = 0
        msk[msk > thres] = 1

        # border = 5
        # kernel = np.ones((border, border), np.uint8)
        # msk_erode = cv2.erode(msk.copy(), kernel)
        # msk_dilate = cv2.dilate(msk.copy(), kernel)
        # msk[(msk_dilate - msk_erode) == 1] = 100
        # msk[np.logical_and(msk_bk > 5, msk_bk < 250)] = 100

        # image boundary pixels
        row_ids, col_ids = np.where(msk > 0)
        row_min, row_max = np.min(row_ids), np.max(row_ids)
        col_min, col_max = np.min(col_ids), np.max(col_ids)
        pad = max(msk.shape[0], msk.shape[1]) // 30
        row_min = max(0, row_min - pad)
        row_max = min(msk.shape[0]-1, row_max + pad)
        col_min = max(0, col_min - pad)
        col_max = min(msk.shape[1]-1, col_max + pad)
        msk[:row_min] += 100
        msk[row_max:] += 100
        msk[:, :col_min] += 100
        msk[:, col_max:] += 100

        return msk
    
    def get_normal(self, frame_id, cam_id, H, W):
        assert False, "this function is not implemented in the released code"
        cam = self.cams[cam_id]
        nml_dir = 'xxx/eval/%s/nml' % self.FLAGS.dataset_name
        nml_fpath = os.path.join(nml_dir, 'frame%08d_view%02d.png' % (frame_id, cam_id))
        nml = cv2.imread(nml_fpath) / 255. * 2. - 1.
        # nml_fpath = os.path.join(nml_dir, 'frame%08d_view%02d.exr' % (frame_id, cam_id))
        # nml = imageio.imread(nml_fpath) * 2. - 1.

        msk_fpath = os.path.join(nml_dir, '../msk', 'frame%08d_view%02d.png' % (frame_id, cam_id))
        msk = cv2.imread(msk_fpath) / 255.
        if len(msk.shape) > 2:
            msk = msk[..., 0]

        if nml.shape[0] > H:
            pad0 = (nml.shape[0] - H) // 2
            pad1 = nml.shape[0] - H - pad0
            nml = nml[pad0:-pad1]
            msk = msk[pad0:-pad1]
        elif nml.shape[1] > W:
            pad0 = (nml.shape[1] - W) // 2
            pad1 = nml.shape[1] - W - pad0
            nml = nml[:, pad0:-pad1]
            msk = msk[:, pad0:-pad1]

        cam_K, cam_R, cam_t = cam['K'], cam['R'], cam['t']

        nml = nml * np.array([1, -1, -1], dtype=nml.dtype).reshape([1, 3])
        nml = cam2world(nml, cam_R, np.zeros_like(cam_t)).reshape(H, W, 3)

        return torch.from_numpy(np.concatenate([nml, msk[..., None]], axis=-1)).float()
    
    def _parse_data(self, item):
        cam_id = self.cam_ids_to_use[item % len(self.cam_ids_to_use)]
        frame_id = item // len(self.cam_ids_to_use)
        if self.validate:
            frame_id = (item * 5) % self.frame_num # skip some consecutive frames in validation
        frame_id = self.valid_frame_list[frame_id]

        cam = self.cams[cam_id]
        mv, mvp, campos = cam['mv'][None], cam['mvp'][None], cam['campos'][None]
        iter_res, iter_spp = self.FLAGS.display_res, self.FLAGS.spp

        img_fpath = os.path.join(self.data_dir, '%s/%08d.jpg' % (self.cam_ssn_list[cam_id], frame_id))
        msk_fpath = os.path.join(self.data_dir, self.cam_ssn_list[cam_id], 'mask-rvm', '%08d.jpg' % frame_id)
        if not os.path.exists(msk_fpath):
            msk_fpath = os.path.join(self.data_dir, self.cam_ssn_list[cam_id], 'mask/pha', '%08d.jpg' % frame_id)
        img, msk = cv2.imread(img_fpath), self.get_mask(msk_fpath)

        img = np.float32(img[:, :, ::-1]) / 255.
        
        if abs(self.ratio-1) > 1e-6:
            H, W = int(img.shape[0] * self.ratio), int(img.shape[1] * self.ratio)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(np.concatenate([img, msk[..., None]], axis=-1))[None]
        img[..., :3] = util.srgb_to_rgb(img[..., :3])

        if self.FLAGS.normal_supervised:
            nml = self.get_normal(frame_id, cam_id, img.shape[1], img.shape[2])
        else:
            nml = None

        return_dict = {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : img,
            'pose': self.poses[frame_id - self.frame_id_offset].detach(),
            'trans': self.trans[frame_id - self.frame_id_offset].detach(),
            'idx': torch.LongTensor([frame_id]),
            'c_idx': torch.LongTensor([cam_id])
        }
        if self.rots is not None:
            return_dict.update({'rot': self.rots[frame_id - self.frame_id_offset].detach(),})
        if nml is not None:
            return_dict.update({'nml': nml})

        return return_dict

    def __getitem__(self, itr):
        return self._parse_data(itr % self.n_images)


class TestDatasetSMPL(DatasetSMPL):
    def __init__(self, data_dir, glctx, FLAGS):
        super().__init__(data_dir, glctx, FLAGS, 
                            validate=False, real=True)
    
    def __len__(self):
        return self.n_images