# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). All Bytedance Modifications are Copyright 2022 Bytedance Inc.


# Copyright 2023 Bytedance Inc.
# All rights reserved.
# Licensed under the BSD 3-Clause Clear License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://choosealicense.com/licenses/bsd-3-clause-clear/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.ops import ste_round
from compressai.layers import GDN, MaskedConv2d,NAFBlock
from compressai.models.utils import conv, deconv, update_registered_buffers,conv_con,deconv_con
from compressai.layers import conv3x3, subpel_conv3x3,UFONE,FFN,AttnFFN,AttnFFN_LN,ConvTransBlock
from ptflops import get_model_complexity_info
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from compressai.layers.stf_utils import *
from compressai.layers import PatchEmbed, PatchMerging, PatchSplitting, BasicLayer

from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    
)
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
eps = 1e-9
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192,  M=320, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        # self.h_a = nn.Sequential(
        #     conv(M, N, stride=1, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     conv(N, N),
        #     nn.ReLU(inplace=True),
        #     conv(N, N),
        # )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        # self.h_s = nn.Sequential(
        #     deconv(N, N),
        #     nn.ReLU(inplace=True),
        #     deconv(N, N),
        #     nn.ReLU(inplace=True),
        #     conv(N, M, stride=1, kernel_size=3),
        #     nn.ReLU(inplace=True),
        # )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.quantizer = Quantizer()
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18, 0.36, 0.72, 1.44]
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000,14.1421,20.0000,28.2842]), requires_grad=True)
        self.levels = len(self.lmbda)  # 8

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, noise=False, stage=3, s=1):
        if stage > 1:
            if s != 0:
                QuantizationRegulator = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
            else:
                s = 0
                QuantizationRegulator = self.Gain[s].detach()
        else:
            QuantizationRegulator = self.Gain[0].detach()

        ReQuantizationRegulator = 1.0 / QuantizationRegulator.clone().detach()

        if noise:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            scales_hat, means_hat = scales_hat.chunk(2, 1)
            y_hat = self.gaussian_conditional.quantize(y * QuantizationRegulator,
                                                       "noise" if self.training else "dequantize") * ReQuantizationRegulator
            # _, y_likelihoods = self.gaussian_conditional(y * QuantizationRegulator, scales_hat * QuantizationRegulator)
            _, y_likelihoods = self.gaussian_conditional(y*QuantizationRegulator, scales_hat*QuantizationRegulator,means_hat*QuantizationRegulator)
            x_hat = self.g_s(y_hat)
        else:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            scales_hat = self.h_s(z_hat)
            scales_hat, means_hat = scales_hat.chunk(2, 1)
            y_hat = self.quantizer.quantize((y-means_hat) * QuantizationRegulator, "ste") * ReQuantizationRegulator+means_hat
            _, y_likelihoods = self.gaussian_conditional(y * QuantizationRegulator, scales_hat * QuantizationRegulator,means_hat*QuantizationRegulator)
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = 192
        M = 320
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x, s, inputscale=0):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * QuantizationRegulator)
        y_strings = self.gaussian_conditional.compress((y-means_hat) * QuantizationRegulator, indexes,)
        # y_strings = self.gaussian_conditional.compress(y* QuantizationRegulator, indexes, means=means_hat* QuantizationRegulator)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * QuantizationRegulator)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes) * ReQuantizationRegulator+means_hat
        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat* QuantizationRegulator)* ReQuantizationRegulator
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



class SwinTAnalysisTransform(nn.Module):

    def __init__(self, embed_dim, embed_out_dim, depths, head_dim, window_size, input_dim):
        super().__init__()
        self.patch_embed = PatchEmbed(dim=input_dim, out_dim=embed_dim[0])
        # self.patch_embed = nn.Conv2d(input_dim, embed_dim[0], 2, 2)
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        head_dim=head_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchMerging if (i < num_layers - 1) else None)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)

class SwinTSynthesisTransform(nn.Module):

    def __init__(self, embed_dim, embed_out_dim, depths, head_dim, window_size):
        super().__init__()
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        head_dim=head_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchSplitting)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)

class SwinTHyperprior_TIC(ScaleHyperprior):
    """SwinT-Hyperprior
    Y. Zhu, Y. Yang, T. Cohen:
    "Transformer-Based Transform Coding"

    International Conference on Learning Representations (ICLR), 2022
    https://openreview.net/pdf?id=IDwN6xjHnK8
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        N = 192
        M = 320
        config={
            'g_a': {
                'input_dim': 3,
                'embed_dim': [96, 128, 192, 320],
                'embed_out_dim': [128, 192, 320, None],
                'depths': [2, 2, 6, 2],
                'head_dim': [32, 32, 32, 32],
                'window_size': [8, 8, 8, 8]
            },
            'g_s': {
                'embed_dim': [320, 192, 128, 96],
                'embed_out_dim': [192, 128, 96, 3],
                'depths': [2, 6, 2, 2],
                'head_dim': [32, 32, 32, 32],
                'window_size': [8, 8, 8, 8],
            }}
        self.g_a = SwinTAnalysisTransform(**config['g_a'])
        self.g_s = SwinTSynthesisTransform(**config['g_s'])
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )



class ScaleHyperprior_NAFNet(ScaleHyperprior):

    def __init__(self, *args, **kwargs):
        super().__init__()

        N = 192
        M = 320
        
        self.g_a = nn.Sequential(
            conv(3, N),
            NAFBlock(N),
            NAFBlock(N),
            conv(N, N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),
            conv(N, N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),
            conv(N, M),
            NAFBlock(M),
        )

        self.g_s = nn.Sequential(
            NAFBlock(M),
            deconv(M, N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),

            deconv(N, N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),
            NAFBlock(N),

            deconv(N, N),
            NAFBlock(N),
            NAFBlock(N),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )


class ConvNeXtBlockLN(nn.Module):
    default_embedding_dim = 256
    def __init__(self, dim, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        # self.norm.affine = False # for FLOPs computing
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        from timm.layers.mlp import Mlp
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual

    def forward(self, x):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # MLP
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x

class ScaleHyperprior_chengres(ScaleHyperprior):
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )
        self.g_s = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

 

        
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )

class ScaleHyperprior_Convres(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192,  M=320, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            conv(N, N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            conv(N, N),
            ConvNeXtBlockLN(N,kernel_size=5),
            ConvNeXtBlockLN(N,kernel_size=5),
            ConvNeXtBlockLN(N,kernel_size=5),
            ConvNeXtBlockLN(N,kernel_size=5),
            conv(N, M),
            ConvNeXtBlockLN(M,kernel_size=5),
            ConvNeXtBlockLN(M,kernel_size=5),
            ConvNeXtBlockLN(M,kernel_size=3),
            ConvNeXtBlockLN(M,kernel_size=3),
            ConvNeXtBlockLN(M,kernel_size=3),
        )



        self.g_s = nn.Sequential(
            ConvNeXtBlockLN(M,kernel_size=3),
            ConvNeXtBlockLN(M,kernel_size=3),
            ConvNeXtBlockLN(M,kernel_size=3),
            ConvNeXtBlockLN(M,kernel_size=5),
            ConvNeXtBlockLN(M,kernel_size=5),
            deconv(M, N),
            ConvNeXtBlockLN(N,kernel_size=5),
            ConvNeXtBlockLN(N,kernel_size=5),
            ConvNeXtBlockLN(N,kernel_size=5),
            ConvNeXtBlockLN(N,kernel_size=5),
            deconv(N, N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            deconv(N, N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            ConvNeXtBlockLN(N),
            deconv(N, 3),
        )



        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )


        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.quantizer = Quantizer()
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18, 0.36, 0.72, 1.44]
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000,14.1421,20.0000,28.2842]), requires_grad=True)
        self.levels = len(self.lmbda)  # 8

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, noise=False, stage=3, s=1):
        if stage > 1:
            if s != 0:
                QuantizationRegulator = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
            else:
                s = 0
                QuantizationRegulator = self.Gain[s].detach()
        else:
            QuantizationRegulator = self.Gain[0].detach()

        ReQuantizationRegulator = 1.0 / QuantizationRegulator.clone().detach()

        if noise:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            scales_hat, means_hat = scales_hat.chunk(2, 1)
            y_hat = self.gaussian_conditional.quantize(y * QuantizationRegulator,
                                                       "noise" if self.training else "dequantize") * ReQuantizationRegulator
            # _, y_likelihoods = self.gaussian_conditional(y * QuantizationRegulator, scales_hat * QuantizationRegulator)
            _, y_likelihoods = self.gaussian_conditional(y*QuantizationRegulator, scales_hat*QuantizationRegulator,means_hat*QuantizationRegulator)
            x_hat = self.g_s(y_hat)
        else:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            scales_hat = self.h_s(z_hat)
            scales_hat, means_hat = scales_hat.chunk(2, 1)
            y_hat = self.quantizer.quantize((y-means_hat) * QuantizationRegulator, "ste") * ReQuantizationRegulator+means_hat
            _, y_likelihoods = self.gaussian_conditional(y * QuantizationRegulator, scales_hat * QuantizationRegulator,means_hat*QuantizationRegulator)
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = 192
        M = 320
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x, s, inputscale=0):
        # print("true")
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * QuantizationRegulator)
        # y_strings = self.gaussian_conditional.compress((y-means_hat) * QuantizationRegulator, indexes,)
        y_strings = self.gaussian_conditional.compress(y* QuantizationRegulator, indexes, means=means_hat* QuantizationRegulator)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * QuantizationRegulator)
        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes) * ReQuantizationRegulator+means_hat
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat* QuantizationRegulator)* ReQuantizationRegulator
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class ScaleHyperprior_DITN(ScaleHyperprior):

    def __init__(self, *args, **kwargs):
        super().__init__()

        N = 192
        M = 320
        
        self.g_a = nn.Sequential(
            conv(3, N),
            UFONE(N),
            conv(N, N),
            UFONE(N),
            conv(N, N),
            UFONE(N),
            conv(N, M),
            UFONE(M),
        )

        self.g_s = nn.Sequential(
            UFONE(M),
            deconv(M, N),
            UFONE(N),
            deconv(N, N),
            UFONE(N),
            deconv(N, N),
            UFONE(N),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )



from torch import Tensor

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out

class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out



class ScaleHyperprior_elicres(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192,  M=320, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)


        self.g_a = nn.Sequential(
            conv(3, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            deconv(N, 3),
        )

 

        
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )

       


        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.quantizer = Quantizer()
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18, 0.36, 0.72, 1.44]
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000,14.1421,20.0000,28.2842]), requires_grad=True)
        self.levels = len(self.lmbda)  # 8

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, noise=False, stage=3, s=1):
        if stage > 1:
            if s != 0:
                QuantizationRegulator = torch.max(self.Gain[s], torch.tensor(1e-4)) + eps
            else:
                s = 0
                QuantizationRegulator = self.Gain[s].detach()
        else:
            QuantizationRegulator = self.Gain[0].detach()

        ReQuantizationRegulator = 1.0 / QuantizationRegulator.clone().detach()

        if noise:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            scales_hat, means_hat = scales_hat.chunk(2, 1)
            y_hat = self.gaussian_conditional.quantize(y * QuantizationRegulator,
                                                       "noise" if self.training else "dequantize") * ReQuantizationRegulator
            # _, y_likelihoods = self.gaussian_conditional(y * QuantizationRegulator, scales_hat * QuantizationRegulator)
            _, y_likelihoods = self.gaussian_conditional(y*QuantizationRegulator, scales_hat*QuantizationRegulator,means_hat*QuantizationRegulator)
            x_hat = self.g_s(y_hat)
        else:
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            scales_hat = self.h_s(z_hat)
            scales_hat, means_hat = scales_hat.chunk(2, 1)
            y_hat = self.quantizer.quantize((y-means_hat) * QuantizationRegulator, "ste") * ReQuantizationRegulator+means_hat
            _, y_likelihoods = self.gaussian_conditional(y * QuantizationRegulator, scales_hat * QuantizationRegulator,means_hat*QuantizationRegulator)
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        N = 192
        M = 320
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x, s, inputscale=0):
        # print("true")
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * QuantizationRegulator)
        # y_strings = self.gaussian_conditional.compress((y-means_hat) * QuantizationRegulator, indexes,)
        y_strings = self.gaussian_conditional.compress(y* QuantizationRegulator, indexes, means=means_hat* QuantizationRegulator)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        scales_hat, means_hat = scales_hat.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat * QuantizationRegulator)
        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes) * ReQuantizationRegulator+means_hat
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat* QuantizationRegulator)* ReQuantizationRegulator
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class ScaleHyperprior_EfficientV2(ScaleHyperprior):

    def __init__(self, *args, **kwargs):
        super().__init__()

        N = 192
        M = 320
        
        self.g_a = nn.Sequential(
            conv(3, N),
            FFN(N,mlp_ratio=2),
            FFN(N),
            conv(N, N),
            FFN(N),
            FFN(N),
            # FFN(N),
            FFN(N),
            conv(N, N),
            # AttnFFN(N),
            # AttnFFN(N),
            AttnFFN(N),
            AttnFFN(N),
            conv(N, M),
            AttnFFN(M),
        )

        self.g_s = nn.Sequential(
            AttnFFN(M),
            deconv(M, N),
            # AttnFFN(N),
            # AttnFFN(N),
            AttnFFN(N),
            AttnFFN(N),

            deconv(N, N),
            # FFN(N),
            FFN(N),
            FFN(N),
            FFN(N),

            deconv(N, N),
            FFN(N),
            FFN(N,mlp_ratio=2),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )


class ScaleHyperprior_EfficientV2_LN(ScaleHyperprior):

    def __init__(self, *args, **kwargs):
        super().__init__()

        N = 192
        M = 320
        
        self.g_a = nn.Sequential(
            conv(3, N),
            FFN(N,mlp_ratio=2),
            FFN(N),
            conv(N, N),
            FFN(N),
            FFN(N),
            # FFN(N),
            FFN(N),
            conv(N, N),
            # AttnFFN(N),
            # AttnFFN(N),
            AttnFFN_LN(N),
            AttnFFN_LN(N),
            conv(N, M),
            AttnFFN_LN(M),
        )

        self.g_s = nn.Sequential(
            AttnFFN_LN(M),
            deconv(M, N),
            # AttnFFN(N),
            # AttnFFN(N),
            AttnFFN_LN(N),
            AttnFFN_LN(N),

            deconv(N, N),
            # FFN(N),
            FFN(N),
            FFN(N),
            FFN(N),

            deconv(N, N),
            FFN(N),
            FFN(N,mlp_ratio=2),
            deconv(N, 3),
        )
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )



class ScaleHyperprior_TCM(ScaleHyperprior):

    def __init__(self, drop_path_rate=0.1,*args, **kwargs):
        super().__init__()

        N = 192
        M = 320
        config=[2, 2, 2, 2, 2, 2]
        head_dim=[8, 16, 32, 32, 16, 8]
        self.head_dim = head_dim
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0
        self.window_size = 8
        dim = N//2
        
        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(N, N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(N, N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(N, N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(N, N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)
        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, M*2),
        )




if __name__ == "__main__":
    model = ScaleHyperprior_NAFNet(N=192, M=320)
    input = torch.Tensor(1, 3, 256, 256)
    print(model)
    out = model(input,stage=1)
    print(out["x_hat"].shape)
    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False,
                                           print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #MACs per pixel
    print("MACs per pixel:{}".format(macs/256/256))
    #paramters(M)
    print("paramters(M):{}".format(params/1000000))