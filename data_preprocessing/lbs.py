import torch
import torch.nn

from smplx.lbs import batch_rodrigues, batch_rigid_transform

def lbs(v_shaped, j_shaped, pose, parents, lbs_weights):
    """
    This implementation is based on https://github.com/vchoutas/smplx/smplx/lbs.py.

    weights: [bs, n_pts, 24]
    body_lbs_weights: [6890, 24], used for similarity weighting between cloth and smpl verts
    posedirs: will add pose correctives if not None, shape: [207, 20670]
    """
    batch_size = max(v_shaped.shape[0], pose.shape[0])
    device, dtype = v_shaped.device, v_shaped.dtype

    v_posed = v_shaped  # [N, 6890, 3]
    J = j_shaped  # [N, 24, 3]
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])   # [N, 24, 3, 3]
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # do skinning:
    W = lbs_weights   # [N, 6890, 24]
    T = torch.einsum('bvj,bjrc->bvrc', W, A)

    homogen_coord = torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device) # [N, 6890, 1], all ones
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)   # [N, 6890, 4]
    v_homo = torch.einsum('bvrc,bvc->bvr', T, v_posed_homo)

    verts = v_homo[:, :, :3]# / v_homo[:, :, 3:]

    return {'v_posed': verts, 'j_posed': J_transformed, 'v_tfs': T}


def inv_lbs(v_posed, j_unposed, pose, parents, lbs_weights):
    """
    This implementation is based on https://github.com/vchoutas/smplx/smplx/lbs.py.
    """

    batch_size = max(v_posed.shape[0], pose.shape[0])
    device, dtype = v_posed.device, v_posed.dtype

    v_posed = v_posed  # [N, 6890, 3]
    J = j_unposed  # [N, 24, 3]

    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape([batch_size, -1, 3, 3])   # [N, 24, 3, 3]
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    ### NOTE Do skinning:
    W = lbs_weights   # [N, 6890, 24]
    T = torch.einsum('bvj,bjrc->bvrc', W, A)
    T_inv = torch.linalg.inv(T)

    homogen_coord = torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device) # [N, 6890, 1], all ones
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)   # [N, 6890, 4]
    v_homo = torch.einsum('bvrc,bvc->bvr', T_inv, v_posed_homo)

    verts = v_homo[:, :, :3] # / v_homo[:, :, 3:]

    return {'v_unposed': verts, 'j_posed': J_transformed, 'v_tfs_inv': T_inv}
