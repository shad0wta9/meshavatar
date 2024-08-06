import numpy as np
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional encoding (section 5.1， NeRF)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. **torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. **0., 2. **max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FastEmbedder(nn.Module):
    def __init__(self, multires, input_dims=3, include_input=True, log_sampling=True):
        super(FastEmbedder, self).__init__()
        self.multires = multires
        self.input_dims = input_dims
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.out_dim = 2 * multires * input_dims + input_dims * int(include_input)

    def forward(self, xyz):
        if self.multires<= 0:
            return xyz

        xyz_shape = xyz.shape
        xyz_ = xyz.unsqueeze(-2)
        freq_bands = 2. ** torch.linspace(0., self.multires-1, steps=self.multires).to(xyz.device)
        freq_bands = freq_bands.reshape(*((1,)*(len(xyz_shape)-1)), -1, 1)
        xyz_sin = torch.sin(xyz_ * freq_bands)
        xyz_cos = torch.cos(xyz_ * freq_bands)
        xyz_ebd = torch.cat([xyz_sin.unsqueeze(-2), xyz_cos.unsqueeze(-2)], dim=-2)
        xyz_ebd = torch.cat([xyz, xyz_ebd.reshape(*xyz_shape[:-1], -1)], dim=-1)
        return xyz_ebd


def get_embedder(multires, input_dims=3, include_input=True, log_sampling=True):
    if multires <= 0:
        return nn.Identity(), input_dims

    # embed_kwargs = {
    #     'include_input' : include_input,
    #     'input_dims' : input_dims,
    #     'max_freq_log2' : multires -1,
    #     'num_freqs' : multires,
    #     'log_sampling' : log_sampling,
    #     'periodic_fns' : [torch.sin, torch.cos],
    # }
    #
    # embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj : eo.embed(x)
    # return embed, embedder_obj.out_dim
    embed = FastEmbedder(multires, input_dims, include_input, log_sampling)
    out_dim = embed.out_dim
    return embed, out_dim


def get_embedder_sin(multires, input_dims=3, include_input=True, log_sampling=True):
    if multires <= 0:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input' : include_input,
        'input_dims' : input_dims,
        'max_freq_log2' : multires -1,
        'num_freqs' : multires,
        'log_sampling' : log_sampling,
        'periodic_fns' : [torch.sin],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class ImgNormalizerForResnet(nn.Module):
    def __init__(self):
        super(ImgNormalizerForResnet, self).__init__()
        IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        IMG_NORM_STD = [0.229, 0.224, 0.225]
        img_mean = np.array(IMG_NORM_MEAN, dtype=np.float32).reshape((1, -1, 1, 1))
        img_std = np.array(IMG_NORM_STD, dtype=np.float32).reshape((1, -1, 1, 1))
        self.register_buffer('img_mean', torch.from_numpy(img_mean))
        self.register_buffer('img_std', torch.from_numpy(img_std))

    def forward(self, imgs):
        imgs_ = F.interpolate(imgs, [224, 224], mode='bilinear', align_corners=False)
        imgs_ = (imgs_ - self.img_mean) / self.img_std
        return imgs_


def sal_init(m):
    from torch.nn.init import _calculate_correct_fan

    if isinstance(m, nn.Linear):
        if hasattr(m, 'weight'):
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out'))

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)

    if isinstance(m, nn.Conv1d):
        if hasattr(m, 'weight'):
            g = m.groups
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out') / g)

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)


def sal_init_last_layer(m):
    from torch.nn.init import _calculate_correct_fan

    if isinstance(m, nn.Linear):
        if hasattr(m, 'weight'):
            val = np.sqrt(np.pi) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_in'))
            with torch.no_grad():
                m.weight.fill_(val)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)

    if isinstance(m, nn.Conv1d):
        if hasattr(m, 'weight'):
            g = m.groups
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out') / g)

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)


"""
The following code is needed in unet.py
"""

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def pad_and_add_tensor(target, t, dim=-1, pad_ahead=False):
    tgt_shape = target.shape
    src_shape = t.shape
    dim = dim if dim >= 0 else dim+len(target.shape)
    assert all(tgt_shape[i]==src_shape[i] for i in range(len(tgt_shape)) if i!=dim)
    assert tgt_shape[dim] >= src_shape[dim]
    if tgt_shape[dim] > src_shape[dim]:
        pad_shape = deepcopy(list(tgt_shape))
        pad_shape[dim] = tgt_shape[dim] - src_shape[dim]
        pad = torch.zeros(*pad_shape, device=target.device, dtype=target.dtype)
        if pad_ahead:
            t = torch.cat([pad, t], dim=dim)
        else:
            t = torch.cat([t, pad], dim=dim)
    return target + t
