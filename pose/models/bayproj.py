import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from pose.models.lacorr2d import LocalAutoCorr2DCUDA, PadInfo
import pose.utils.config as config

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
        self.extract_input = nn.Conv2d(in_channels, corr_channels, kernel_size=3, padding=1)
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
    def __init__(self, channel_size=None, radius_std_init=1., input_std=False):
        super(LongRangeProj, self).__init__()
        self._float32_eps = np.finfo(np.float32).eps.item()
        self.input_std = input_std
        if not input_std:
            self.radius_std = nn.Parameter(torch.FloatTensor(channel_size))
            self.angle_std = nn.Parameter(torch.FloatTensor(channel_size))

            # TODO: improve initialization
            self.radius_std.data.uniform_(radius_std_init * (1-0.2), radius_std_init * (1+0.2))
            self.angle_std.data.uniform_(np.pi * (1-0.2), np.pi * (1+0.2))

    def _proj(self, force_field, force_norm, cx, cy, radius_mean, angle_mean, radius_std, angle_std):
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
        # TODO: range too long ? restrict the range, make it use visiblity
        batch_size = radius_mean.size(0)
        channel_size = radius_mean.size(1)
        height = force_field.size(1)
        width = force_field.size(2)
        cx = int(cx.item())
        cy = int(cy.item())

        radius_mean = radius_mean.abs().view(batch_size, channel_size, 1, 1)
        radius_std = radius_std.view(1 if radius_std.dim() == 1 else batch_size, channel_size, 1, 1)
        radius_dist = torch.exp(-(force_norm.expand(batch_size, channel_size, -1, -1) - radius_mean)**2 / 2 / radius_std**2)

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
        ang_dist = torch.exp(-fgacos(ang_dis_cos)**2 / 2 / angle_std**2)

        dist = radius_dist * ang_dist

        if config.debug_nan:
            def get_back_hook(hook_name):
                def back_hook(grad):
                    for g in grad:
                        if (g.data != g.data).any():
                            print("[LongRangeProj]" + hook_name + " contains NaN")
                            import ipdb; ipdb.set_trace()
                return back_hook

            ang_dis_cos.register_hook(get_back_hook("ang_dis_cos"))
            ang_dist.register_hook(get_back_hook("ang_dist"))
            radius_dist.register_hook(get_back_hook("radius_dist"))
            dist.register_hook(get_back_hook("dist"))

        return dist

    def forward(self, force_field, force_norm, origin_x, origin_y, radius, angle, confidence, radius_std=None, angle_std=None):
        """
        Arguments:
            force_field {torch.FloatTensor} -- [nh x nw x 2 x height x width]
            force_norm {torch.FloatTensor} -- [nh x nw x height x width]
            origin_x {torch.FloatTensor} -- [nw]
            origin_y {torch.FloatTensor} -- [nh]
            radius {torch.FloatTensor} -- [nb x nh x nw x nchan]
            angle {torch.FloatTensor} -- [nb x nh x nw x nchan]
            confidence {torch.FloatTensor} -- [nb x nh x nw x nchan]
            radius_std {torch.FloatTensor} -- [nb x nh x nw x nchan] or None
            angle_std {torch.FloatTensor} -- [nb x nh x nw x nchan] or None
        """
        batch_size = radius.size(0)
        channel_size = radius.size(3)

        if self.input_std:
            assert radius_std is not None and angle_std is not None

        out = None
        # TODO: random drop out to ease training (when random, should we disable confidence?)
        for iy in range(radius.size(1)):
            for ix in range(radius.size(2)):
                proj = confidence[:, iy, ix].view(batch_size, channel_size, 1, 1) * \
                       self._proj(force_field[iy, ix],
                                  force_norm[iy, ix],
                                  origin_x[ix],
                                  origin_y[iy],
                                  radius[:, iy, ix],
                                  angle[:, iy, ix],
                                  radius_std[:, iy, ix] if self.input_std else self.radius_std,
                                  angle_std[:, iy, ix] if self.input_std else self.angle_std)
                if out is None:
                    out = proj
                else:
                    out = torch.max(out, proj)

        return out

class AutoCorrProj(nn.Module):
    def __init__(self, in_channels, out_channels, corr_channels, corr_kernel_size, corr_stride, pad=False):
        super(AutoCorrProj, self).__init__()
        self.acorr2d = AutoCorr2D(in_channels, None, corr_channels, corr_kernel_size, corr_stride, pad=pad, permute=False)
        self.radius_regressor = nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=True)
        self.angle_regressor = nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=True)
        self.radius_std_regressor = nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=True)
        self.angle_std_regressor = nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=True)
        self.conf_regressor = nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=True)
        # TODO: improve initialization
        self.projector = LongRangeProj(input_std=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._n_corr_h = None
        self._n_corr_w = None
        self._height = None
        self._width = None
        self._force_field = None
        self._force_norm = None
        self._origin_x = None
        self._origin_y = None

        # For finetune
        # TODO: improve initialization
        self.radius_regressor.weight.data.zero_()
        self.radius_regressor.bias.data.uniform_(0, 0.5)
        self.angle_regressor.weight.data.uniform_(-1e-3, 1e-3)
        self.angle_regressor.bias.data.uniform_(-np.pi, np.pi)
        self.radius_std_regressor.bias.data.fill_(10)
        self.angle_std_regressor.bias.data.fill_(np.pi)
        self.conf_regressor.weight.data.zero_()
        self.conf_regressor.bias.data.zero_()

    def _init_force(self, n_corr_h, n_corr_w, height, width, device=torch.device("cpu")):
        if self._n_corr_h != n_corr_h or self._n_corr_w != n_corr_w or self._height != height or self._width != width or self._force_field.device != device:
            self._n_corr_h = n_corr_h
            self._n_corr_w = n_corr_w
            self._height = height
            self._width = width

            x_orig = torch.arange(n_corr_w, dtype=torch.float32, device=device)
            x_orig = x_orig * self.acorr2d.corr_stride[1] - self.acorr2d.pad.left + self.acorr2d.corr_kernel_size[1] // 2
            y_orig = torch.arange(n_corr_h, dtype=torch.float32, device=device)
            y_orig = y_orig * self.acorr2d.corr_stride[0] - self.acorr2d.pad.top + self.acorr2d.corr_kernel_size[0] // 2
            self._origin_x = x_orig
            self._origin_y = y_orig

            x_orig = x_orig.view(1, -1, 1, 1)
            y_orig = y_orig.view(-1, 1, 1, 1)
            x_out = torch.arange(width, dtype=torch.float32, device=device).view(1, 1, 1, -1)
            y_out = torch.arange(height, dtype=torch.float32, device=device).view(1, 1, -1, 1)

            self._force_field = torch.stack([
                    (x_out-x_orig).expand(n_corr_h, -1, height, -1),
                    (y_out-y_orig).expand(-1, n_corr_w, -1, width)
                ], dim=2)
            self._force_norm = torch.norm(self._force_field, dim=2)

    def _regress(self, x):
        batch_size = x.size(0)
        # corrs shape: b x ch x cw x chan x kh x kw
        corrs = self.acorr2d(x)
        n_corr_h = corrs.size(1)
        n_corr_w = corrs.size(2)
        # corrs shape: b*ch*cw x chan x kh x kw
        corrs = corrs.view(batch_size*n_corr_h*n_corr_w, corrs.size(3), corrs.size(4), corrs.size(5))
        # radius shape: b*ch*cw x chan x 1 x 1
        # TODO: radius could be negative, or too large
        radius = self.radius_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
        # TODO: angle periodicity
        angle = self.angle_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
        radius_std = self.radius_std_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
        angle_std = self.angle_std_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)
        conf = self.conf_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)

        return radius, angle, radius_std, angle_std, conf, n_corr_h, n_corr_w

    @staticmethod
    def _sum_outsider(radius, angle, origin_x, origin_y, height, width):
        # b x nh x nw x chan
        proj_cen_x = radius * torch.cos(angle) + origin_x.view(1, 1, -1, 1)
        proj_cen_y = radius * torch.sin(angle) + origin_y.view(1, -1, 1, 1)

        left_outsd = (proj_cen_x.data < 0)
        right_outsd = (proj_cen_x.data >= width)
        top_outsd = (proj_cen_y.data < 0)
        bottom_outsd = (proj_cen_y.data >= height)
        loss_out_total = ((proj_cen_x[left_outsd] ** 2).sum() + ((proj_cen_x[right_outsd] - width) ** 2).sum() + \
                          (proj_cen_y[top_outsd] ** 2).sum() + ((proj_cen_y[bottom_outsd] - height) ** 2).sum())
        count_out_total = (left_outsd | right_outsd | top_outsd | bottom_outsd).int().sum().float()

        return loss_out_total, count_out_total

    def forward(self, inp):
        height = inp.size(2)
        width = inp.size(3)
        # b x nh x nw x chan
        radius, angle, radius_std, angle_std, conf, n_corr_h, n_corr_w = self._regress(inp)
        self._init_force(n_corr_h, n_corr_w, height, width, device=inp.device)

        loss_out_total, count_out_total = self._sum_outsider(radius, angle, self._origin_x, self._origin_y, height, width)

        # b x chan x h x w
        out = self.projector(self._force_field, self._force_norm, self._origin_x, self._origin_y, radius, angle, conf, radius_std=radius_std, angle_std=angle_std)
        
        if config.debug:
            print(radius)
            print(angle)
            print(conf)
            import ipdb; ipdb.set_trace()

        if config.vis:
            import matplotlib.pyplot as plt
            import cv2
            fig, axes = plt.subplots(2, 10, squeeze=False)
            for row, axes_row in enumerate(axes):
                # img = (config.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                fts = out.data[row].cpu().numpy()
                for col, ax in enumerate(axes_row):
                    ax.imshow(fts[col])
            plt.show()

        return out, loss_out_total, count_out_total
