import math
import copy
from functools import partial
from typing import Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# triton cross scan, 2x speed than pytorch implementation =========================
try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1


# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs

# HELP Function cross scan =============

def flip_odd_rows(x):
    odd_rows_flipped = torch.flip(x[:, :, 1::2, :], dims=[2, 3])
    even_rows = x[:, :, ::2, :]
    # Concatenate even rows followed by the odd rows in the original order
    result = torch.zeros(x.shape, device='cuda')
    result[:,:,::2, :] = even_rows  # Index every second row, starting from 0
    result[:,:,1::2, :] = torch.flip(odd_rows_flipped, dims=[2] ) # Index every second row, starting from 1 
    return result

def flip_odd_columns(x):
    even_cols_flipped = torch.flip(x[:, :, :, 1::2], dims=[2, 3])
    odd_cols = x[:, :, :, ::2]
    # Concatenate even columns followed by the odd columns in the original order
    result = torch.zeros(x.shape, device='cuda')
    result[:, :, :, ::2] = odd_cols # Index every second column, starting from 0
    result[:, :, :, 1::2] = torch.flip(even_cols_flipped, dims=[3]) # Index every second column, starting from 1
    return result

# pytorch cross scan snake=============
class CrossScanSnake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)

        # test = torch.tensor([[[[ 1,  2,  3,  4],[ 5,  6,  7,  8],[ 9, 10, 11, 12],[13, 14, 15, 16]]]])
        # b,c,h,w=test.shape

        # ts = test.new_empty((b, 4, c, h*w))

        # ts[:, 0] = flip_odd_rows(test).flatten(2, 3)
        # ts[:, 1] = flip_odd_columns(test).transpose(dim0=2, dim1=3).flatten(2, 3)

        # test_flipped = torch.flip(test, dims=[-2, -1])
        # ts[:, 2] = flip_odd_rows(test_flipped).flatten(2, 3)
        # ts[:, 3] = flip_odd_columns(test_flipped).transpose(dim0=2, dim1=3).flatten(2, 3)

        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = flip_odd_rows(x).flatten(-2, -1).contiguous()
        xs[:, 1] = flip_odd_columns(x).transpose(dim0=-2, dim1=-1).flatten(-2, -1).contiguous()

        x_flipped = torch.flip(x, dims=[-2, -1])
        xs[:, 2] = flip_odd_rows(x_flipped).flatten(-2, -1).contiguous()
        xs[:, 3] = flip_odd_columns(x_flipped).transpose(dim0=-2, dim1=-1).flatten(-2, -1).contiguous()

        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        ys = ys.view(B, 4, C, H, W)
        y = ys.new_empty((B, 4, C, H*W))
        y[:,0] = flip_odd_rows(ys[:,0]).flatten(-2, -1).contiguous()
        y[:,1] = flip_odd_columns(ys[:,1].transpose(dim0=-2, dim1=-1)).flatten(-2, -1).contiguous()
        y[:,2] = flip_odd_rows(ys[:,2]).flatten(-2, -1).flip(-1).contiguous()
        y[:,3] = flip_odd_rows(ys[:,3].flip(-2)).transpose(dim0=-2, dim1=-1).flatten(-2, -1).contiguous()

        y = y[:,0] + y[:,1] + + y[:,2] + y[:,3]
        return y.view(B, C, H, W)


class CrossMergeSnake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape #([4, 4, 192, 64, 64])
        ctx.shape = (H, W)

        # test = torch.tensor([[[1,  2,  3,  4,  8,  7,  6,  5, 9, 10, 11, 12, 16, 15, 14, 13]],
        #                     [[ 1,  5,  9, 13, 14, 10,  6,  2,  3,  7, 11, 15, 16, 12,  8,  4]],
        #                     [[16, 15, 14, 13,  9, 10, 11, 12,  8,  7,  6,  5,  1,  2,  3,  4]],
        #                     [[16, 12,  8,  4,  3,  7, 11, 15, 14, 10,  6,  2,  1,  5,  9, 13]]]).reshape(1, 4, 1, 4, 4)
        # print(test)
        # b, k, d, h, w=test.shape
        # ts = test.new_empty((b, k, d, h*w))
        # ts[:,0] = flip_odd_rows(test[:,0]).flatten(-2, -1).contiguous()
        # ts[:,1] = flip_odd_columns(test[:,1].transpose(dim0=-2, dim1=-1)).flatten(-2, -1).contiguous()
        # ts[:,2] = flip_odd_rows(test[:,2]).flatten(-2, -1).flip(-1).contiguous()
        # ts[:,3] = flip_odd_rows(test[:,3].flip(-2)).transpose(dim0=-2, dim1=-1).flatten(-2, -1).contiguous()

        y = ys.new_empty((B, K, D, H*W))
        y[:,0] = flip_odd_rows(ys[:,0]).flatten(-2, -1).contiguous()
        y[:,1] = flip_odd_columns(ys[:,1].transpose(dim0=-2, dim1=-1)).flatten(-2, -1).contiguous()
        y[:,2] = flip_odd_rows(ys[:,2]).flatten(-2, -1).flip(-1).contiguous()
        y[:,3] = flip_odd_rows(ys[:,3].flip(-2)).transpose(dim0=-2, dim1=-1).flatten(-2, -1).contiguous()
        y = y[:,0] + y[:,1] + y[:,2] + y[:,3]
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape

        x = x.reshape(B, C, H, W)
        xs = x.new_empty((B, 4, C, H, W))

        xs[:, 0] = flip_odd_rows(x).flatten(-2, -1).contiguous().reshape(B, C, H, W)
        xs[:, 1] = flip_odd_columns(x).transpose(dim0=-2, dim1=-1).flatten(-2, -1).contiguous().reshape(B, C, H, W)

        x_flipped = torch.flip(x, dims=[-2, -1])
        xs[:, 2] = flip_odd_rows(x_flipped).flatten(-2, -1).contiguous().reshape(B, C, H, W)
        xs[:, 3] = flip_odd_columns(x_flipped).transpose(dim0=-2, dim1=-1).flatten(-2, -1).contiguous().reshape(B, C, H, W)
        return xs

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)

# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

# cross selective scan ===============================
class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

# =============

def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    out_norm_shape="v0",
    # ==============================
    to_dtype=True, # True: final out to dtype
    force_fp32=False, # True: input fp32
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
    # ==============================
    SelectiveScan=None,
    CrossScan=CrossScan,
    CrossMerge=CrossMerge,
    no_einsum=False, # replace einsum with linear or conv1d to raise throughput
    dt_low_rank=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
        
    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    if (not dt_low_rank):
        x_dbl = F.conv1d(x.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, -1, L), [D, 4 * N, 4 * N], dim=1)
        xs = CrossScan.apply(x)
        dts = CrossScan.apply(dts)
    elif no_einsum:
        xs = CrossScan.apply(x)
        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
    else:
        xs = CrossScan.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]: # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    else: # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes() # u
    N = inputs[2].type().sizes()[1] # A
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False, device='cuda')
        self.norm = norm_layer(4 * dim, device='cuda')

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x
    
class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = nn.Linear
        self.fc1 = Linear(in_features, hidden_features, device='cuda')
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, device='cuda')
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# =====================================================

class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model,
        d_state,
        ssm_ratio,
        dt_rank,
        act_layer,
        # dwconv ===============
        d_conv, # < 2 means no conv 
        conv_bias,
        # ======================
        dropout,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        # ======================
        **kwargs,
    ):
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type,
        )

        if forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
            return
        else:
            self.__initv2__(**kwargs)
            return

    def __initv2__(
        self,
        # basic dims ===========
        d_model,
        d_state,
        ssm_ratio,
        dt_rank,
        act_layer,
        # dwconv ===============
        d_conv, # < 2 means no conv 
        conv_bias,
        # ======================
        dropout,
        bias,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": 'cuda', "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        Linear = nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        self.out_norm_shape = "v1"
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False, device='cuda')
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            class SoftmaxSpatial(nn.Softmax):
                def forward(self, x: torch.Tensor):
                    B, C, H, W = x.shape
                    return super().forward(x.view(B, C, -1)).view(B, C, H, W)
            self.out_norm = SoftmaxSpatial(dim=-1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner, device='cuda')

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        
    def __initxv__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
        ):
            factory_kwargs = {"device":'cuda', "dtype": None}
            super().__init__()
            d_inner = int(ssm_ratio * d_model)
            dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
            self.d_conv = d_conv
            self.d_state = d_state
            self.dt_rank = dt_rank
            self.d_inner = d_inner
            Linear = nn.Linear
            self.forward = self.forwardxv

            # tags for forward_type ==============================
            def checkpostfix(tag, value):
                ret = value[-len(tag):] == tag
                if ret:
                    value = value[:-len(tag)]
                return ret, value

            # softmax | sigmoid | dwconv | norm ===========================
            self.out_norm_shape = "v1"
            if forward_type[-len("none"):] == "none":
                forward_type = forward_type[:-len("none")]
                self.out_norm = nn.Identity()
            elif forward_type[-len("dwconv3"):] == "dwconv3":
                forward_type = forward_type[:-len("dwconv3")]
                self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False, **factory_kwargs)
            elif forward_type[-len("softmax"):] == "softmax":
                forward_type = forward_type[:-len("softmax")]
                class SoftmaxSpatial(nn.Softmax):
                    def forward(self, x: torch.Tensor):
                        B, C, H, W = x.shape
                        return super().forward(x.view(B, C, -1)).view(B, C, H, W)
                self.out_norm = SoftmaxSpatial(dim=-1)
            elif forward_type[-len("sigmoid"):] == "sigmoid":
                forward_type = forward_type[:-len("sigmoid")]
                self.out_norm = nn.Sigmoid()
            else:
                self.out_norm_shape = "v0"
                self.out_norm = nn.LayerNorm(d_inner,**factory_kwargs)

            k_group = 4
            # in proj =======================================
            
            self.forward = partial(self.forwardxv, mode="xv1a")
            self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
            self.out_act = nn.GELU()

            # conv =======================================
            if d_conv > 1:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
                self.act: nn.Module = act_layer()

            # out proj =======================================
            self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
            self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

            if initialize in ["v0"]:
                # dt proj ============================
                self.dt_projs = [
                    self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                    for _ in range(k_group)
                ]
                self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
                self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
                del self.dt_projs
                
                # A, D =======================================
                self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
                self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
            elif initialize in ["v1"]:
                # simple init dt_projs, A_logs, Ds
                self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
                self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
                self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
                self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
            elif initialize in ["v2"]:
                # simple init dt_projs, A_logs, Ds
                self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
                self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
                self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
                self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

            if forward_type.startswith("xv2"):
                del self.dt_projs_weight


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device='cuda', merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device='cuda', merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward_corev2(self, x: torch.Tensor, cross_selective_scan=cross_selective_scan, **kwargs):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")

        return cross_selective_scan(
            x, x_proj_weight, None, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, delta_softplus=True,
            out_norm=out_norm,
            out_norm_shape=out_norm_shape,
            **kwargs,
        )

    def forwardv2(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        

        x = x.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        y = self.forward_core(x)

        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out
    
    def forwardxv(self, x: torch.Tensor, mode="xv1a", omul=False, **kwargs):
        B, H, W, C = x.shape
        L = H * W
        K = 4
        dt_projs_weight = getattr(self, "dt_projs_weight", None)
        A_logs = self.A_logs
        dt_projs_bias = self.dt_projs_bias
        force_fp32 = False
        delta_softplus = True
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        out_norm = self.out_norm
        to_dtype = True
        Ds = self.Ds

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

        x = x.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x = self.conv2d(x) # (b, d, h, w)
            x = self.act(x)
        x = self.in_proj(x)

        if mode in ["xv1", "xv2", "xv3", "xv7"]:
            print(f"ERROR: MODE {mode} will be deleted in the future, use {mode}a instead.")

        if mode in ["xv1"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, -1, L)
            dts = F.conv1d(dts, dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        elif mode in ["xv2"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts).contiguous().view(B, -1, L)
        elif mode in ["xv3"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton1b1.apply(dts.contiguous().view(B, K, -1, H, W))
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        else:
            ...

        if mode in ["xv1a"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            us, dts = us.contiguous().view(B, -1, L), dts
            _us = us.view(B, K, -1, H, W)[:, 0, :, :, :]
        elif mode in ["xv2a"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            us, dts = us.contiguous().view(B, -1, L), dts.contiguous().view(B, -1, L)
        elif mode in ["xv3a"]:
            # us, dtBCs = x.split([self.d_inner, 4 * self.dt_rank + 4 * self.d_state + 4 * self.d_state], dim=1)
            # _us = us
            # us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            # dtBCs = CrossScanTriton1b1.apply(dtBCs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            # dts, Bs, Cs = dtBCs.split([self.dt_rank, self.d_state, self.d_state], dim=2)
            # dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            # us, dts = us.contiguous().view(B, -1, L), dts
            
            us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton1b1.apply(dts.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            us, dts = us.contiguous().view(B, -1, L), dts
        else: 
            ...

        Bs, Cs = Bs.view(B, K, -1, L).contiguous(), Cs.view(B, K, -1, L).contiguous()
    
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float) # (K * c)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
            
        y: torch.Tensor = CrossMergeTriton.apply(ys)
        y = y.view(B, -1, H, W)

        # originally:
        # y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        # y = out_norm(y).view(B, H, W, -1)

        y = out_norm(y.permute(0, 2, 3, 1))

        y = (y.to(x.dtype) if to_dtype else y)
        y = self.out_act(y)
        out = self.dropout(self.out_proj(y))
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        norm_layer: nn.Module,
        # =============================
        ssm_d_state: int,
        ssm_ratio,
        ssm_dt_rank: Any,
        ssm_act_layer,
        ssm_conv: int,
        ssm_conv_bias,
        ssm_drop_rate: float,
        ssm_init,
        forward_type,
        # =============================
        mlp_ratio,
        mlp_act_layer,
        mlp_drop_rate: float,
        # =============================
        use_checkpoint: bool,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim, device='cuda')
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim, device='cuda')
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate)

    def _forward(self, input: torch.Tensor):
            if self.ssm_branch:
                if self.post_norm:
                    x = input + self.drop_path(self.norm(self.op(input)))
                else:
                    x = input + self.drop_path(self.op(self.norm(input)))
            if self.mlp_branch:
                if self.post_norm:
                    x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
                else:
                    x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
            return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSG(nn.Module):
    def __init__(
        self, 
        in_chans, 
        patch_size_global, 
        patch_size_local, 
        #depths=[2, 2, 9, 2], 
        gl_merge,
        depths=[2],
        #dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2", 
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", 
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.dim = 96
        self.gl_merge = gl_merge
        #self.scale = nn.Parameter(torch.ones(in_chans, 1, 1, device='cuda'))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.layers = nn.ModuleList()

        self.patch_embed_global = self._make_patch_embed(in_chans//2, self.dim//2, patch_size_global, patch_norm, norm_layer)
        self.patch_unembed_global = self._make_patch_unembed(self.dim//2, in_chans//2, patch_size_global)

        if self.gl_merge:
            self.patch_embed_local = self._make_patch_embed(in_chans//2, self.dim//2, patch_size_local, patch_norm, norm_layer)
            self.patch_unembed_local = self._make_patch_unembed(self.dim//2, in_chans//2, patch_size_local)

            for i_layer in range(self.num_layers):
                self.layers.append(GlobalLocalScan(
                    dim = self.dim//2,
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                ))
        else:
            for i_layer in range(self.num_layers):
                self.layers.append(GlobalScan(
                    dim = self.dim,
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer,
                    # =================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans, embed_dim, patch_size, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True, device='cuda'),
            #nn.MaxPool2d(kernel_size=patch_size, stride=patch_size),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim, device='cuda') if patch_norm else nn.Identity())
        )
    
    @staticmethod
    def _make_patch_unembed(embed_dim, out_chans, patch_size=4): 
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            #nn.Conv2d(embed_dim, out_chans, kernel_size=1, stride=1, bias=True, device='cuda'),
            #nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, out_chans*patch_size*patch_size, kernel_size=1, stride=1, bias=True, device='cuda'),
            nn.PixelShuffle(upscale_factor=patch_size),
        )

    def forward_gl(self, x: torch.Tensor):
        x_global, x_local = torch.chunk(x, 2, dim=1)
        # print(x_global.shape, x_local.shape)

        x_global = self.patch_embed_global(x_global) # bigger conv kernel, resulting in smaller feature map 
        x_local = self.patch_embed_local(x_local) # smaller conv kernel, resulting in bigger feature map 
        # print(x_global.shape, x_local.shape)

        for i, layer in enumerate(self.layers):
            x_global, x_local = layer(x_global, x_local)
            # print(x_global.shape, x_local.shape)

        x_global = self.patch_unembed_global(x_global)
        x_local = self.patch_unembed_local(x_local)
        # print(x_global.shape, x_local.shape)

        #----------------------------------
        # x = self.scale * x_global + x_local 
        #----------------------------------
        # x = x_global + x_local 
        #----------------------------------

        x = torch.cat([x_global, x_local], dim=1)
        # print(x.shape)
        return x
    
    def forward_g(self, x: torch.Tensor):
        x_global = self.patch_embed_global(x) 

        for i, layer in enumerate(self.layers):
            x_global = layer(x_global)

        x = self.patch_unembed_global(x_global)
        return x
    
    def forward(self, x: torch.Tensor):
        if self.gl_merge:
            return self.forward_gl(x)
        else:
            return self.forward_g(x)

    def flops(self, x):
        shape = x.shape[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

class GlobalLocalScan(nn.Module):
    def __init__(            
            self,
            dim, 
            drop_path, 
            use_checkpoint, 
            norm_layer,
            # ===========================
            ssm_d_state,
            ssm_ratio,
            ssm_dt_rank,       
            ssm_act_layer,
            ssm_conv,
            ssm_conv_bias,
            ssm_drop_rate, 
            ssm_init,
            forward_type,
            # ===========================
            mlp_ratio,
            mlp_act_layer,
            mlp_drop_rate,
            **kwargs,):
        super(GlobalLocalScan, self).__init__()
        depth = len(drop_path)

        blocks_global = []
        for d in range(depth):
            blocks_global.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        blocks_local = []
        for d in range(depth):
            blocks_local.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        self.seq_global = nn.Sequential(OrderedDict(blocks=nn.Sequential(*blocks_global)))
        self.seq_local = nn.Sequential(OrderedDict(blocks=nn.Sequential(*blocks_local)))

    def forward(self, x_global, x_local):
        #x_global = self.seq_global(x_global) + x_global
        #x_local = self.seq_local(x_local) + x_local
        x_global = self.seq_global(x_global)
        x_local = self.seq_local(x_local)
        return x_global, x_local


class GlobalScan(nn.Module):
    def __init__(            
            self,
            dim, 
            drop_path, 
            use_checkpoint, 
            norm_layer,
            # ===========================
            ssm_d_state,
            ssm_ratio,
            ssm_dt_rank,       
            ssm_act_layer,
            ssm_conv,
            ssm_conv_bias,
            ssm_drop_rate, 
            ssm_init,
            forward_type,
            # ===========================
            mlp_ratio,
            mlp_act_layer,
            mlp_drop_rate,
            **kwargs,):
        super(GlobalScan, self).__init__()
        depth = len(drop_path)

        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        self.seq_global = nn.Sequential(OrderedDict(blocks=nn.Sequential(*blocks)))

    def forward(self, x_global):
        #x_global = self.seq_global(x_global) + x_global
        x_global = self.seq_global(x_global)
        return x_global



