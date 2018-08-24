import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from pose.models.lacorr2d import LocalAutoCorr2DCUDA, PadInfo
from pose.models.common import StrictNaNReLU
from utils.globals import config, hparams
from utils.log import log_i

class AutoCorr2D(nn.Module):
    def __init__(self, in_channels, out_channels, corr_channels, corr_kernel_size, corr_stride=0, pad=False, permute=True):
        super(AutoCorr2D, self).__init__()
        assert isinstance(corr_kernel_size, int) or isinstance(corr_kernel_size, tuple)
        if isinstance(corr_kernel_size, int):
            assert corr_kernel_size > 0
            corr_kernel_size = (corr_kernel_size, corr_kernel_size)
        else:
            assert len(corr_kernel_size) == 2 and isinstance(corr_kernel_size[0], int) and isinstance(corr_kernel_size[1], int)
            assert corr_kernel_size[0] > 0 and corr_kernel_size[1] > 0
        assert isinstance(corr_stride, int) or isinstance(corr_stride, tuple)
        if isinstance(corr_stride, int):
            assert corr_stride > 0
            corr_stride = (corr_stride, corr_stride)
        else:
            assert len(corr_stride) == 2 and isinstance(corr_stride[0], int) and isinstance(corr_stride[1], int)
            assert corr_stride[0] > 0 and corr_stride[1] > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.corr_channels = corr_channels
        self.corr_kernel_size = corr_kernel_size
        self.corr_stride = corr_stride
        if pad is False:
            pad = PadInfo()
        elif isinstance(pad, int):
            pad = PadInfo(pad, pad, pad, pad)
        elif isinstance(pad, tuple):
            if len(pad) == 2:
                pad = PadInfo(pad[0], pad[0], pad[1], pad[1])
            elif len(pad) == 4:
                pad = PadInfo(pad[0], pad[1], pad[2], pad[3])
            else:
                raise ValueError("length of pad tuple must be 2 or 4")
        elif pad == "k":
            pad = PadInfo(top=corr_kernel_size[0] // 2,
                          bottom=(corr_kernel_size[0] + 1) // 2 - 1,
                          left=corr_kernel_size[1] // 2,
                          right=(corr_kernel_size[1] + 1) // 2 - 1)
        elif not isinstance(pad, PadInfo):
            raise ValueError("pad must be one of values: False, int, tuple, PadInfo, 'k'")
        self.pad = pad
        self.permute = permute
        self.extract_input = nn.Sequential(nn.Conv2d(in_channels, corr_channels, kernel_size=3, padding=1),
                                           StrictNaNReLU(inplace=True))
        self.corr2d = LocalAutoCorr2DCUDA(corr_kernel_size,
                                          corr_stride,
                                          pad)
        if out_channels is not None:
            self.regressor = nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=True)
        else:
            self.regressor = None

        # TODO: improve initialization
        # For finetune
        if self.regressor is not None:
            self.regressor.weight.data.zero_()
            self.regressor.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.extract_input(x)
        # corrs shape: b x ch x cw x chan x kh x kw
        corrs = self.corr2d(x)

        if self.regressor is not None:
            n_corr_h = corrs.size(1)
            n_corr_w = corrs.size(2)
            corrs = corrs.view(-1, self.corr_channels, self.corr_kernel_size[0], self.corr_kernel_size[1])
            # regressor result shape: b*ch*cw x chan x 1 x 1
            corrs = self.regressor(corrs)
            corrs = corrs.view(batch_size, n_corr_h, n_corr_w, corrs.size(1))

            if self.permute:
                # back to b x chan x ch x cw size
                corrs = corrs.permute(0, 3, 1, 2)

        return corrs

class FriendlyGradArcCosine(Function):
    """
    Avoid potential NaN grad
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x.acos()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = -1 / (1 - x**2).sqrt()
        mask_xp1 = (x == 1)
        mask_xn1 = (x == -1)
        grad[mask_xp1 | mask_xn1] = 0
        # Define semi-derivative at +-1, assume minimzing cost/loss
        grad[mask_xp1 & (grad_output < 0)] = -3
        grad[mask_xn1 & (grad_output > 0)] = -3
        grad = grad.clamp(-3, 0)
        return grad_output * grad

class FriendlyGradCosine(Function):
    """
    Replace 0 grad by random
    """
    @staticmethod
    def forward(ctx, x):
        output = x.cos()
        ctx.save_for_backward(x, (output > 0))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, mask_cosgt0 = ctx.saved_tensors
        grad = -x.sin()
        mask_geq0 = (grad == 0)
        if mask_geq0.any():
            # Assume minimizing cost/loss
            mask_p = mask_geq0 & mask_cosgt0 & (grad_output > 0)
            mask_n = mask_geq0 & ~mask_cosgt0 & (grad_output < 0)
            mask_grad_inner_dir = mask_p | mask_n
            num_random = mask_grad_inner_dir.int().sum()
            if num_random > 0:
                grad[mask_grad_inner_dir] = torch.rand(num_random, dtype=torch.float32, device=grad.device) * 0.2 - 0.1
        return grad_output * grad

class FriendlyGradSine(Function):
    """
    Replace 0 grad by random
    """
    @staticmethod
    def forward(ctx, x):
        output = x.sin()
        ctx.save_for_backward(x, (output > 0))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, mask_singt0 = ctx.saved_tensors
        grad = x.cos()
        mask_geq0 = (grad == 0)
        if mask_geq0.any():
            # Assume minimizing cost/loss
            mask_p = mask_geq0 & mask_singt0 & (grad_output > 0)
            mask_n = mask_geq0 & ~mask_singt0 & (grad_output < 0)
            mask_grad_inner_dir = mask_p | mask_n
            num_random = mask_grad_inner_dir.int().sum()
            if num_random > 0:
                grad[mask_grad_inner_dir] = torch.rand(num_random, dtype=torch.float32, device=grad.device) * 0.2 - 0.1
        return grad_output * grad

class SelectBlocker(Function):
    @staticmethod
    def forward(ctx, x, sel, override_output, override_grad):
        ctx.block_sel = sel
        ctx.override_grad = override_grad
        if override_output is not None:
            x = x.clone()
            x.__setitem__(sel, override_output)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.override_grad is not None:
            grad_output = grad_output.clone()
            grad_output.__setitem__(ctx.block_sel, ctx.override_grad)
        return grad_output, None, None, None

fgcos = FriendlyGradCosine.apply
fgsin = FriendlyGradSine.apply
fgacos = FriendlyGradArcCosine.apply
selblock = SelectBlocker.apply

class LongRangeProj(nn.Module):
    def __init__(self,
                 channel_size=None,
                 radius_std_init=1.,
                 input_std=False,
                 mode="prob",
                 summary_mode="max",
                 use_conv_final=False,
                 samp_sigma=1.,
                 proj_data=False,
                 local_mask_sigma=0):
        super(LongRangeProj, self).__init__()
        assert mode in ["prob", "samp"]
        assert summary_mode in ["max", "sum"]
        self.mode = mode
        self.summary_mode = summary_mode
        self._float32_eps = np.finfo(np.float32).eps.item()
        self.input_std = input_std
        self.use_conv_final = use_conv_final
        if self.mode == "samp":
            self.proj_samp_sigma = float(samp_sigma)
        self.proj_data = proj_data
        self._local_mask_sigma = local_mask_sigma

        self._nh = None
        self._nw = None
        self._height = None
        self._width = None
        self._force_field = None
        self._force_norm = None
        self._local_mask = None
        self._origin_x = None
        self._origin_y = None

        if proj_data:
            raise NotImplementedError("Not completed")
        if use_conv_final:
            self.conv_final = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        if not input_std:
            self.radius_std = nn.Parameter(torch.FloatTensor(channel_size))
            self.angle_std = nn.Parameter(torch.FloatTensor(channel_size))

            # TODO: improve initialization
            self.radius_std.data.uniform_(float(radius_std_init) * (1-0.2), float(radius_std_init) * (1+0.2))
            self.angle_std.data.uniform_(np.pi * (1-0.2), np.pi * (1+0.2))

    def init_buffer(self, nh, nw, height, width, kernel_size, stride, pad, device=torch.device("cpu")):
        """
        Init:
            _force_field {torch.FloatTensor} -- [nh x nw x 2 x height x width]
            _force_norm {torch.FloatTensor} -- [nh x nw x height x width]
            _local_mask {torch.FloatTensor} -- [nh x nw x height x width]
            _origin_x {torch.FloatTensor} -- [nw]
            _origin_y {torch.FloatTensor} -- [nh]
        """
        if self._force_field is None or self._nh != nh or self._nw != nw or self._height != height or self._width != width:
            device_cpu = torch.device("cpu")
            self._nh = nh
            self._nw = nw
            self._height = height
            self._width = width

            x_orig = torch.arange(nw, dtype=torch.float32)
            x_orig = x_orig * stride[1] - pad[1] + kernel_size[1] // 2
            y_orig = torch.arange(nh, dtype=torch.float32)
            y_orig = y_orig * stride[0] - pad[0] + kernel_size[0] // 2
            self._origin_x = dict()
            self._origin_x[device_cpu] = x_orig
            self._origin_y = dict()
            self._origin_y[device_cpu] = y_orig

            origin_x = x_orig.view(1, -1, 1, 1)
            origin_y = y_orig.view(-1, 1, 1, 1)
            x_out = torch.arange(width, dtype=torch.float32).view(1, 1, 1, -1)
            y_out = torch.arange(height, dtype=torch.float32).view(1, 1, -1, 1)

            self._force_field = dict()
            self._force_field[device_cpu] = torch.stack([
                    (x_out-origin_x).expand(nh, -1, height, -1),
                    (y_out-origin_y).expand(-1, nw, -1, width)
                ], dim=2)
            self._force_norm = dict()
            self._force_norm[device_cpu] = torch.norm(self._force_field, dim=2)
            if self._local_mask_sigma > 0:
                self._local_mask = dict()
                self._local_mask[device_cpu] = 1 - torch.exp(- self._force_norm ** 2 / 2 / self._local_mask_sigma ** 2)

        if device not in self._force_field:
            device_cpu = torch.device("cpu")
            self._force_field[device] = self._force_field[device_cpu].to(device)
            self._force_norm[device] = self._force_norm[device_cpu].to(device)
            self._local_mask[device] = self._local_mask[device_cpu].to(device)

    def _proj_prob(self, force_field, force_norm, cx, cy, radius_mean, angle_mean, radius_std, angle_std):
        """
        Arguments:
            force_field {torch.FloatTensor} -- [2 x h x w]
            force_norm {torch.FloatTensor} -- [h x w]
            cx {int} -- center x
            cy {int} -- center y
            radius_mean {torch.FloatTensor} -- [batch_size x channel_size]
            angle_mean {torch.FloatTensor} -- [batch_size x channel_size]
            radius_std {torch.FloatTensor} -- [channel_size] or [batch_size x channel_size]
            angle_std {torch.FloatTensor} -- [channel_size] or [batch_size x channel_size]
        
        Return:
            {torch.FloatTensor} -- [batch_size x channel_size x h x w]
        """
        batch_size = radius_mean.size(0)
        channel_size = radius_mean.size(1)
        cx = int(cx.item())
        cy = int(cy.item())

        radius_mean = radius_mean.abs().view(batch_size, channel_size, 1, 1)
        radius_std = radius_std.view(1 if radius_std.dim() == 1 else batch_size, channel_size, 1, 1)
        radius_dist = torch.exp(-(force_norm.expand(batch_size, channel_size, -1, -1) - radius_mean)**2 / 2 / (radius_std**2 + 0.01))

        # batch_size x channel_size x 2
        angle_mean_force = torch.stack([fgcos(angle_mean), fgsin(angle_mean)], dim=2)
        # Compute cosine similarity between force_field and angle_mean vector
        # Use selblock to fix output and zero grad at the origin point
        ang_dis_cos = selblock(
            torch.mm(
                angle_mean_force.view(-1, 2),
                force_field.view(2, -1)
            ).view(angle_mean_force.size()[:-1]+force_field.size()[1:]),
            (slice(None), slice(None), cy, cx), 1, 0
        ) / selblock(force_norm, (cy, cx), 1, 0).expand(batch_size, channel_size, -1, -1)

        # sometimes rounding error can be twice the float32_eps
        assert not (ang_dis_cos.data > 1 + 2 * self._float32_eps).any() and not (ang_dis_cos.data < -1 - 2 * self._float32_eps).any()
        # This is intentional. The outsiders should only be caused by rounding error.
        # We don't want their gradient being eliminated.
        mask_upper = (ang_dis_cos.data > 1)
        mask_lower = (ang_dis_cos.data < -1)
        ang_dis_cos[mask_upper] -= ang_dis_cos.data[mask_upper] - 1
        ang_dis_cos[mask_lower] -= ang_dis_cos.data[mask_lower] + 1

        angle_std = angle_std.view(1 if angle_std.dim() == 1 else batch_size, channel_size, 1, 1)
        ang_dist = torch.exp(-fgacos(ang_dis_cos)**2 / 2 / (angle_std**2 + 0.0001))

        dist = radius_dist * ang_dist

        if config.debug_nan:
            def get_back_hook(hook_name):
                def back_hook(grad):
                    for g in grad:
                        if (g.data != g.data).any():
                            print("LongRangeProj." + hook_name + " contains NaN")
                            import ipdb; ipdb.set_trace()
                return back_hook

            ang_dis_cos.register_hook(get_back_hook("ang_dis_cos"))
            ang_dist.register_hook(get_back_hook("ang_dist"))
            radius_dist.register_hook(get_back_hook("radius_dist"))
            dist.register_hook(get_back_hook("dist"))

        return dist

    def _proj_samp(self, force_field, force_norm, cx, cy, radius_mean, angle_mean, radius_std, angle_std):
        """
        Arguments:
            force_field {torch.FloatTensor} -- [2 x h x w]
            force_norm {torch.FloatTensor} -- [h x w]
            cx {int} -- center x
            cy {int} -- center y
            radius_mean {torch.FloatTensor} -- [batch_size x channel_size]
            angle_mean {torch.FloatTensor} -- [batch_size x channel_size]
            radius_std {torch.FloatTensor} -- [channel_size] or [batch_size x channel_size]
            angle_std {torch.FloatTensor} -- [channel_size] or [batch_size x channel_size]
        
        Return:
            {torch.FloatTensor} -- [batch_size x channel_size x h x w]
        """
        batch_size = radius_mean.size(0)
        channel_size = radius_mean.size(1)

        if self.training:
            # radius_std = radius_std.view(1 if radius_std.dim() == 1 else batch_size, channel_size)
            # angle_std = angle_std.view(1 if angle_std.dim() == 1 else batch_size, channel_size)
            # radius = radius_mean.abs() + torch.randn(batch_size, channel_size, device=force_field.device) * radius_std
            # angle = angle_mean + torch.randn(batch_size, channel_size, device=force_field.device) * angle_std
            radius = radius_mean.abs()
            angle = angle_mean
        else:
            radius = radius_mean.abs()
            angle = angle_mean

        # nb x nc x 2
        force_target = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=2)
        # nb x nc x 2 x h x w
        force_field = force_field[None, None] - force_target[..., None, None]
        # TODO: parameterize std?
        return torch.exp(- force_field.norm(dim=2) ** 2 / 2 / self.proj_samp_sigma ** 2)

    def _sum_outsider(self, radius, angle):
        # b x nh x nw x chan
        proj_cen_x = radius * torch.cos(angle) + self._origin_x[radius.device].view(1, 1, -1, 1)
        proj_cen_y = radius * torch.sin(angle) + self._origin_y[radius.device].view(1, -1, 1, 1)

        left_outsd = (proj_cen_x.data < 0)
        right_outsd = (proj_cen_x.data >= self._width)
        top_outsd = (proj_cen_y.data < 0)
        bottom_outsd = (proj_cen_y.data >= self._height)
        loss_out_total = ((proj_cen_x[left_outsd] ** 2).sum() + ((proj_cen_x[right_outsd] - self._width) ** 2).sum() + \
                          (proj_cen_y[top_outsd] ** 2).sum() + ((proj_cen_y[bottom_outsd] - self._height) ** 2).sum())
        # count_out_total = (left_outsd | right_outsd | top_outsd | bottom_outsd).int().sum().float()

        return loss_out_total

    def forward(self, radius, angle, confidence, radius_std=None, angle_std=None, inp=None):
        """
        Arguments:
            radius {torch.FloatTensor} -- [nb x nh x nw x nchan], >= 0
            angle {torch.FloatTensor} -- [nb x nh x nw x nchan]
            confidence {torch.FloatTensor} -- [nb x nh x nw x nchan]
            radius_std {torch.FloatTensor} -- [nb x nh x nw x nchan] or None, > 0
            angle_std {torch.FloatTensor} -- [nb x nh x nw x nchan] or None, > 0
        """
        batch_size = radius.size(0)
        channel_size = radius.size(3)
        device = radius.device

        if self.proj_data or inp is not None:
            raise NotImplementedError("Not completed")

        if self.input_std:
            assert radius_std is not None and angle_std is not None
        else:
            radius_std = self.radius_std
            angle_std = self.angle_std

        projector = self._proj_prob if self.mode == "proj" else self._proj_samp

        out = None
        # TODO: random drop out to ease training (when random, should we disable confidence?)
        for iy in range(radius.size(1)):
            for ix in range(radius.size(2)):
                proj = confidence[:, iy, ix].view(batch_size, channel_size, 1, 1) * \
                       projector(self._force_field[device][iy, ix],
                                 self._force_norm[device][iy, ix],
                                 self._origin_x[device][ix],
                                 self._origin_y[device][iy],
                                 radius[:, iy, ix],
                                 angle[:, iy, ix],
                                 radius_std[:, iy, ix] if self.input_std else radius_std,
                                 angle_std[:, iy, ix] if self.input_std else angle_std)

                if self._local_mask_sigma > 0:
                    proj = proj * self._local_mask[device]

                if out is None:
                    out = proj
                else:
                    if self.summary_mode == "max":
                        out = torch.max(out, proj)
                    elif self.summary_mode == "sum":
                        out = out + proj
                    else:
                        raise ValueError("illegal summary_mode=" + self.summary_mode)

        if self.use_conv_final:
            out = self.conv_final(out)

        loss_out_total = self._sum_outsider(radius, angle)
        count_point = torch.tensor([radius.size(3) * self._nh * self._nw], dtype=torch.float, device=loss_out_total.device)
        loss_in_total = torch.relu(1 - radius).sum()

        return out, loss_out_total, loss_in_total, count_point

class AutoCorrProj(nn.Module):
    def __init__(self,
                 use_acorr,
                 in_channels,
                 out_channels,
                 inner_channels,
                 kernel_size,
                 stride,
                 regress_std,
                 proj_mode,
                 proj_summary_mode,
                 proj_use_conv_final,
                 proj_data=False,
                 pad=False,
                 proj_samp_sigma=1.,
                 radius_std_init=10.,
                 proj_local_mask_sigma=0):
        super(AutoCorrProj, self).__init__()
        self.use_acorr = use_acorr
        regressor_kwargs = {}
        if use_acorr:
            if proj_data:
                raise NotImplementedError("proj_data currently not implemented for use_acorr")
            log_i("Using acorr")
            self.acorr2d = AutoCorr2D(in_channels, None, inner_channels, kernel_size, stride, pad=pad, permute=False)
            self._kernel_size = self.acorr2d.corr_kernel_size
            self._stride = self.acorr2d.corr_stride
            self._pad_tl = (self.acorr2d.pad.top, self.acorr2d.pad.left)
        else:
            log_i("Using fake acorr")
            assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple)
            if isinstance(kernel_size, int):
                assert kernel_size > 0
                kernel_size = (kernel_size, kernel_size)
            else:
                assert len(kernel_size) == 2 and isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int)
                assert kernel_size[0] > 0 and kernel_size[1] > 0

            if pad is False:
                pad = (0, 0)
            elif pad is True:
                pad = (kernel_size[0] // 2, kernel_size[1] // 2)
            elif isinstance(pad, int):
                pad = (pad, pad)
            elif not isinstance(pad, tuple) and len(pad) == 2:
                raise ValueError("Illegal pad argument")

            assert isinstance(stride, int) or isinstance(stride, tuple)
            if isinstance(stride, int):
                assert stride > 0
                stride = (stride, stride)
            else:
                assert len(stride) == 2 and isinstance(stride[0], int) and isinstance(stride[1], int)
                assert stride[0] > 0 and stride[1] > 0

            groups = 1
            if proj_data:
                assert in_channels == inner_channels, "proj_data requires in_channels == inner_channels"
                groups = in_channels
            self.acorr2d_sim = nn.Sequential(nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, groups=groups),
                                             nn.Softplus())
            regressor_kwargs["stride"] = stride
            regressor_kwargs["padding"] = pad
            self._kernel_size = kernel_size
            self._stride = stride
            self._pad_tl = pad

        groups = 1
        if proj_data:
            assert out_channels == inner_channels, "proj_data requires out_channels == inner_channels"
            groups = out_channels
        self.radius_regressor = nn.Conv2d(inner_channels, out_channels, kernel_size=kernel_size, bias=True, groups=groups, **regressor_kwargs)
        self.angle_regressor = nn.Conv2d(inner_channels, out_channels, kernel_size=kernel_size, bias=True, groups=groups, **regressor_kwargs)
        if regress_std:
            self.radius_std_regressor = nn.Conv2d(inner_channels, out_channels, kernel_size=kernel_size, bias=True, groups=groups, **regressor_kwargs)
            self.angle_std_regressor = nn.Conv2d(inner_channels, out_channels, kernel_size=kernel_size, bias=True, groups=groups, **regressor_kwargs)
            # TODO: improve initialization
            self.projector = LongRangeProj(input_std=True,
                                           mode=proj_mode,
                                           summary_mode=proj_summary_mode,
                                           use_conv_final=proj_use_conv_final,
                                           proj_data=proj_data,
                                           local_mask_sigma=proj_local_mask_sigma)
        else:
            self.projector = LongRangeProj(input_std=False,
                                           channel_size=out_channels,
                                           radius_std_init=radius_std_init,
                                           mode=proj_mode,
                                           summary_mode=proj_summary_mode,
                                           use_conv_final=proj_use_conv_final,
                                           proj_data=proj_data,
                                           local_mask_sigma=proj_local_mask_sigma)
        self.conf_regressor = nn.Conv2d(inner_channels, out_channels, kernel_size=kernel_size, bias=True, groups=groups, **regressor_kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.regress_std = regress_std
        

        self._nh = None
        self._nw = None
        self._height = None
        self._width = None

        # For finetune
        # TODO: improve initialization
        # self.radius_regressor.weight.data.uniform_(2, 8)
        self.radius_regressor.bias.data.uniform_(1, 7)
        # self.angle_regressor.weight.data.uniform_(-1e-3, 1e-3)
        self.angle_regressor.bias.data.uniform_(-np.pi, np.pi)
        if regress_std:
            self.radius_std_regressor.bias.data.fill_(10)
            self.angle_std_regressor.bias.data.fill_(np.pi)
        # self.conf_regressor.weight.data.uniform_(0.1, 0.3)
        self.conf_regressor.bias.data.uniform_(0.1, 1.0)

    def _corr_regress(self, x):
        batch_size = x.size(0)
        # corrs shape: b x ch x cw x chan x kh x kw
        corrs = self.acorr2d(x)
        n_corr_h = corrs.size(1)
        n_corr_w = corrs.size(2)
        # corrs shape: b*ch*cw x chan x kh x kw
        corrs = corrs.view(batch_size*n_corr_h*n_corr_w, corrs.size(3), corrs.size(4), corrs.size(5))
        # radius shape: b*ch*cw x chan x 1 x 1
        # NOTE: Do not use nn.ReLU as of pytorch 0.4.1, as it won't propagate NaN
        radius = self.radius_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
        # NOTE: Use clamp instead of nn.Threshold as of pytorch 0.4.1, as the latter won't propagate NaN
        angle = self.angle_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
        if self.regress_std:
            radius_std = self.radius_std_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
            angle_std = self.angle_std_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
        else:
            radius_std = None
            angle_std = None
        conf = torch.relu(self.conf_regressor(corrs)).view(batch_size, n_corr_h, n_corr_w, self.out_channels)

        return radius, angle, radius_std, angle_std, conf, n_corr_h, n_corr_w

    def _conv_regress(self, x):
        batch_size = x.size(0)

        # nb x ninnerchan x h x w
        x = self.acorr2d_sim(x)

        # nb x nchan x nh x nw
        radius = self.radius_regressor(x)
        nh = radius.size(2)
        nw = radius.size(3)
        radius = radius.view(batch_size, self.out_channels, -1).transpose(1, 2).view(batch_size, nh, nw, self.out_channels)
        angle = self.angle_regressor(x).view(batch_size, self.out_channels, -1).transpose(1, 2).view(batch_size, nh, nw, self.out_channels)
        if self.regress_std:
            radius_std = self.radius_std_regressor(x).view(batch_size, self.out_channels, -1).transpose(1, 2).view(batch_size, nh, nw, self.out_channels)
            angle_std = self.angle_std_regressor(x).view(batch_size, self.out_channels, -1).transpose(1, 2).view(batch_size, nh, nw, self.out_channels)
        else:
            radius_std = None
            angle_std = None
        conf = torch.relu(self.conf_regressor(x)).view(batch_size, self.out_channels, -1).transpose(1, 2).view(batch_size, nh, nw, self.out_channels)

        return radius, angle, radius_std, angle_std, conf, nh, nw

    def forward(self, inp):
        height = inp.size(2)
        width = inp.size(3)

        if self.use_acorr:
            # b x nh x nw x chan
            radius, angle, radius_std, angle_std, conf, nh, nw = self._corr_regress(inp)
        else:
            # b x nh x nw x chan
            radius, angle, radius_std, angle_std, conf, nh, nw = self._conv_regress(inp)
        self.projector.init_buffer(nh, nw, height, width, self._kernel_size, self._stride, self._pad_tl, device=inp.device)

        # b x chan x h x w
        out, loss_out_total, loss_in_total, count_point = self.projector(radius, angle, conf, radius_std=radius_std, angle_std=angle_std)

        if config.debug:
            import ipdb; ipdb.set_trace()

        if config.vis and False:
            import matplotlib.pyplot as plt
            import cv2
            mean_out = out.data.mean()
            std_out = out.data.std()
            vmax = mean_out + 3 * std_out
            vmin = mean_out - 3 * std_out
            fig, axes = plt.subplots(out.size(0), 10, squeeze=False)
            for row, axes_row in enumerate(axes):
                # img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                fts = out.data[row].cpu().numpy()
                for col, ax in enumerate(axes_row):
                    ax.imshow(fts[col], vmin=vmin, vmax=vmax)
            plt.show()

        return out, loss_out_total, loss_in_total, count_point
