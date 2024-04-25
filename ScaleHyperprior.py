
import math
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.ops import ste_round
from compressai.layers import GDN
from compressai.models.utils import conv, deconv, update_registered_buffers
from ptflops import get_model_complexity_info
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from compressai.layers.stf_utils import *


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






if __name__ == "__main__":
    model = ScaleHyperprior(N=192, M=320)
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