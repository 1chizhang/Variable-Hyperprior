import torch
import math
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from compressai.models.utils import conv, deconv, update_registered_buffers

import torch.nn as nn
from compressai.layers import GDN, MaskedConv2d,NAFBlock
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import conv3x3, subpel_conv3x3,InternImageLayer
from compressai.ops import ste_round
from compressai.models.priors import CompressionModel, GaussianConditional
from ptflops import get_model_complexity_info
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

eps = 1e-9

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Channelwise_NAFNet(CompressionModel):
    def __init__(self,N = 192,M = 320
                 ):
        super().__init__(entropy_bottleneck_channels=N)
        num_slices=5
        self.num_slices = num_slices
        self.max_support_slices = 5
        
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
        
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18, 0.36, 0.72, 1.44]
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000,14.1421,20.0000,28.2842]), requires_grad=True)
        self.levels = len(self.lmbda)  # 8


    

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
        """Forward function."""
        

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        if noise:
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
        else:
            _, z_likelihoods = self.entropy_bottleneck(z)
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_s(z_hat)
        latent_scales, latent_means = latent_scales.chunk(2, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if noise:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice * QuantizationRegulator,
                                                                  scale * QuantizationRegulator,
                                                                  mu * QuantizationRegulator)

                y_hat_slice = self.gaussian_conditional.quantize(y_slice * QuantizationRegulator,
                                                           "noise" if self.training else "dequantize") * ReQuantizationRegulator
            else:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice*QuantizationRegulator, scale*QuantizationRegulator, mu*QuantizationRegulator)
                y_hat_slice = ste_round((y_slice - mu)*QuantizationRegulator)*ReQuantizationRegulator + mu

            y_likelihood.append(y_slice_likelihood)
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

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
        net = cls()
        net.load_state_dict(state_dict)
        return net


    def compress(self, x, s, inputscale=0):
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        y = self.g_a(x)
        y_shape = y.shape[2:]


        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_s(z_hat)
        latent_scales, latent_means = latent_scales.chunk(2, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]*QuantizationRegulator

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice*QuantizationRegulator, "symbols", mu*QuantizationRegulator)
            y_hat_slice = y_q_slice*ReQuantizationRegulator + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale=0):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_s(z_hat)
        latent_scales, latent_means = latent_scales.chunk(2, 1)


        ReQuantizationRegulator = (torch.tensor(1.0) / QuantizationRegulator).to(z_hat.device)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
       

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]*QuantizationRegulator

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1]).to(z_hat.device)
            y_hat_slice = self.gaussian_conditional.dequantize(rv*ReQuantizationRegulator, mu)


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    

class Channelwise_DCNv4(CompressionModel):
    def __init__(self,N = 192,M = 320
                 ):
        super().__init__(entropy_bottleneck_channels=N)
        num_slices=5
        self.num_slices = num_slices
        self.max_support_slices = 5
        
        self.g_a = nn.Sequential(
            conv(3, N),
            InternImageLayer(N),
            InternImageLayer(N),
            conv(N, N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            conv(N, N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            conv(N, M),
            InternImageLayer(M,groups=20),
        )

        self.g_s = nn.Sequential(
            InternImageLayer(M,groups=20),
            deconv(M, N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            deconv(N, N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            InternImageLayer(N),
            deconv(N, N),
            InternImageLayer(N),
            InternImageLayer(N),
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
        
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18, 0.36, 0.72, 1.44]
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000,14.1421,20.0000,28.2842]), requires_grad=True)
        self.levels = len(self.lmbda)  # 8


    

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
        """Forward function."""
        

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        if noise:
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
        else:
            _, z_likelihoods = self.entropy_bottleneck(z)
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_s(z_hat)
        latent_scales, latent_means = latent_scales.chunk(2, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if noise:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice * QuantizationRegulator,
                                                                  scale * QuantizationRegulator,
                                                                  mu * QuantizationRegulator)

                y_hat_slice = self.gaussian_conditional.quantize(y_slice * QuantizationRegulator,
                                                           "noise" if self.training else "dequantize") * ReQuantizationRegulator
            else:
                _, y_slice_likelihood = self.gaussian_conditional(y_slice*QuantizationRegulator, scale*QuantizationRegulator, mu*QuantizationRegulator)
                y_hat_slice = ste_round((y_slice - mu)*QuantizationRegulator)*ReQuantizationRegulator + mu

            y_likelihood.append(y_slice_likelihood)
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

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
        net = cls()
        net.load_state_dict(state_dict)
        return net


    def compress(self, x, s, inputscale=0):
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        ReQuantizationRegulator = torch.tensor(1.0) / QuantizationRegulator

        y = self.g_a(x)
        y_shape = y.shape[2:]


        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_s(z_hat)
        latent_scales, latent_means = latent_scales.chunk(2, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]*QuantizationRegulator

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice*QuantizationRegulator, "symbols", mu*QuantizationRegulator)
            y_hat_slice = y_q_slice*ReQuantizationRegulator + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape, s, inputscale=0):
        assert isinstance(strings, list) and len(strings) == 2
        if inputscale != 0:
            QuantizationRegulator = inputscale
        else:
            assert s in range(0, self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
            QuantizationRegulator = torch.abs(self.Gain[s])

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_s(z_hat)
        latent_scales, latent_means = latent_scales.chunk(2, 1)


        ReQuantizationRegulator = (torch.tensor(1.0) / QuantizationRegulator).to(z_hat.device)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
       

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]*QuantizationRegulator

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1]).to(z_hat.device)
            y_hat_slice = self.gaussian_conditional.dequantize(rv*ReQuantizationRegulator, mu)


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

if __name__ == "__main__":
    model = Channelwise_DCNv4().cuda()
    input = torch.Tensor(1, 3, 64, 64).cuda()
    out = model(input)
    # print(model)
    print(out["x_hat"].shape)
    flops, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True, print_per_layer_stat=True,
                                              flops_units='Mac', param_units='')
    print('flops: ', flops, 'params: ', params)
