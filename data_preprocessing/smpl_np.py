import numpy as np
import pickle
import scipy.sparse


def _convert_to_sparse(mat):
    if isinstance(mat, np.ndarray):
        row_ind, col_ind = np.asarray(mat > 0).nonzero()
        data = [mat[r, c] for (r, c) in zip(row_ind, col_ind)]
        mat_sparse = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=mat.shape)
        return mat_sparse
    else:
        return mat


class SMPLModel():
    def __init__(self, model_path='./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', beta_dim=10):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                params = pickle.load(f, encoding='iso-8859-1')
            print('SmplNumpy: Loading SMPL data from %s' % model_path)
        elif model_path.endswith('.npz'):
            params = np.load(model_path)
            print('SmplNumpy: Loading SMPL data from %s' % model_path)
        else:
            raise ValueError('Unknown file extension: %s' % model_path)

        self.v_num = len(params['v_template'])
        self.joint_num = params['weights'].shape[1]
        self.beta_dim = beta_dim

        self.J_regressor = _convert_to_sparse(params['J_regressor'])
        self.weights = np.asarray(params['weights'])
        self.posedirs = np.asarray(params['posedirs'])
        self.v_template = np.asarray(params['v_template'])
        self.shapedirs = np.asarray(params['shapedirs'][:, :, :self.beta_dim])
        self.faces = np.asarray(params['f'])
        self.kintree_table = np.asarray(params['kintree_table'])

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [self.joint_num, 3]
        self.beta_shape = [self.beta_dim]
        self.rot_shape = [3]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.rot = np.zeros(self.rot_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None
        self.G = None

        self.update()

    def set_params(self, pose=None, beta=None, rot=None, trans=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        rot: Global rotation of shape [3].

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose is not None:
            self.pose = np.reshape(pose, self.pose_shape)
        else:
            self.pose = np.zeros(self.pose_shape)
        if beta is not None:
            self.beta = np.reshape(beta, self.beta_shape)
        else:
            self.beta = np.zeros(self.beta_shape)
        if rot is not None:
            self.rot = np.reshape(rot, self.rot_shape)
        else:
            self.rot = np.zeros(self.rot_shape)
        if trans is not None:
            self.trans = np.reshape(trans, self.trans_shape)
        else:
            self.trans = np.zeros(self.trans_shape)
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([self.joint_num, 1])]).reshape([self.joint_num, 4, 1])
            )
        )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        rot_mat = self.rodrigues(self.rot.reshape(1, 1, 3)).reshape(3, 3) # 3x3
        self.verts = np.matmul(rot_mat, v.transpose()).transpose()
        self.verts = self.verts + self.trans.reshape([1, 3])
        self.G = G

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float32).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class HandPca:
    def __init__(self, model_path='./data/MANO_LEFT.pkl', ncomps=6):
        """
        Hand PCA Layer.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                params = pickle.load(f, encoding='iso-8859-1')
            print('SmplNumpy: Loading SMPL data from %s' % model_path)
        elif model_path.endswith('.npz'):
            params = np.load(model_path)
            print('SmplNumpy: Loading SMPL data from %s' % model_path)
        else:
            raise ValueError('Unknown file extension: %s' % model_path)

        self.ncomps = ncomps
        self.selected_components = np.vstack(params['hands_components'][:ncomps])
        self.hands_components = np.vstack(params['hands_components'][:])
        self.hands_mean = np.asarray(params['hands_mean'])

    def get_hand_posevec(self, rot, pca_coeffs, flat_mean=False):
        if flat_mean:
            pose_param = np.concatenate([
                rot, pca_coeffs[:self.ncomps].dot(self.selected_components)
            ])
        else:
            pose_param = np.concatenate([
                rot, self.hands_mean + pca_coeffs[:self.ncomps].dot(self.selected_components)
            ])
        return pose_param


class HandTexture:
    def __init__(self, model_path='./data/html/model_sr/model.pkl', uv_path='./data/html/uvs.pkl'):

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.tex_mean = np.asarray(self.model['mean'])  # the mean texture
        self.tex_basis = np.asarray(self.model['basis'])  # 101 PCA comps
        self.vec2img_index = np.asarray(self.model['index_map'])  # the index map, from a compact vector to a 2D texture image

        with open(uv_path,'rb') as f:
            self.uvs = pickle.load(f)
        self.faces_uvs = np.asarray(self.uvs['faces_uvs'])
        self.verts_uvs = np.asarray(self.uvs['verts_uvs'])

        self.ncomps = 101

    def vec2img(self, tex_code):
        img1d = np.zeros(1024*1024*3)
        img1d[self.vec2img_index] = tex_code
        return img1d.reshape((3, 1024,1024)).transpose(2,1,0)

    def check_alpha(self, alpha):
        # just for checking the alpha's length
        if alpha.size < self.ncomps :
            n_alpha = np.zeros(self.ncomps, 1)
            n_alpha[0:alpha.size, :] = alpha
        elif alpha.size > self.ncomps:
            n_alpha = alpha.reshape(alpha.size, 1)[0:self.ncomps, :]
        else:
            n_alpha = alpha.reshape(alpha.size, 1)
        return n_alpha

    def get_mano_texture(self, alpha):
        # first check the length of the input alpha vector
        alpha = self.check_alpha(alpha)
        offsets = np.dot(self.tex_basis, alpha)
        tex_code = offsets.reshape(-1) + self.tex_mean
        new_tex_img = self.vec2img(tex_code) / 255

        return new_tex_img


