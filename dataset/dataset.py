# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return_dict = {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),
            'rots': torch.stack(list([item['rot'] for item in batch]), dim=0) if 'rot' in batch[0] else None,
            'trans': torch.stack(list([item['trans'] for item in batch]), dim=0) if 'trans' in batch[0] else None,
            'idx': torch.stack(list([item['idx'] for item in batch]), dim=0) if 'idx' in batch[0] else None,
            'c_idx': torch.stack(list([item['c_idx'] for item in batch]), dim=0) if 'c_idx' in batch[0] else None,
        }

        if 'pose' in batch[0]:
            return_dict['poses'] = torch.stack(list([item['pose'] for item in batch]), dim=0)
        if 'nml' in batch[0]:
            return_dict['nml'] = torch.stack(list([item['nml'] for item in batch]), dim=0)
        return return_dict