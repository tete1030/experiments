import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from pose.models.lacorr2d import LocalAutoCorr2DCUDA, PadInfo

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
            num_random = mask_grad_inner_dir.uint32().sum()
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
            num_random = mask_grad_inner_dir.uint32().sum()
            if num_random > 0:
                grad[mask_grad_inner_dir] = torch.rand(num_random, dtype=torch.float32, device=grad.device) * 0.2 - 0.1
        return grad_output * grad

class SelectBlocker(Function):
    @staticmethod
    def forward(ctx, x, sel, override_output, override_grad):
        ctx.block_sel = sel
        ctx.override_grad = override_grad
        if override_output is not None:
            x.__setitem__(sel, override_output)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.override_grad is not None:
            grad_output.__setitem__(ctx.block_sel, ctx.override_grad)
        return grad_output, None, None, None

fgcos = FriendlyGradCosine.apply
fgsin = FriendlyGradSine.apply
fgacos = FriendlyGradArcCosine.apply
selblock = SelectBlocker.apply

class LongRangeProj(nn.Module):
    def __init__(self, height=None, width=None, radius_std_init=1.):
        super(LongRangeProj, self).__init__()
        self._float32_eps = np.finfo(np.float32).eps.item()
        self.radius_std = nn.Parameter(torch.FloatTensor(1))
        self.angle_std = nn.Parameter(torch.FloatTensor(1))
        self.height = None
        self.width = None
        self._init_buffer(height, width)

        # TODO: improve initialization
        self.radius_std.data.fill_(radius_std_init)
        self.angle_std.data.fill_(np.pi)

    def _init_buffer(self, height, width, device=torch.device("cpu")):
        if self.width != width or (hasattr(self, "x_out") and self.x_out.device != device):
            self.width = width
            if width is not None:
                self.register_buffer("x_out", torch.arange(width, dtype=torch.float32, device=device).view(1, -1))

        if self.height != height or (hasattr(self, "y_out") and self.y_out.device != device):
            self.height = height
            if height is not None:
                self.register_buffer("y_out", torch.arange(height, dtype=torch.float32, device=device).view(-1, 1))

    def _proj(self, force_field, cx, cy, radius_mean, radius_std, angle_mean, angle_std):
        """
        Arguments:
            force_field {torch.FloatTensor} -- [2 x h x w]
            cx {int} -- center x
            cy {int} -- center y
            radius_mean {torch.FloatTensor} -- [batch_size x channel_size]
            radius_std {torch.FloatTensor} -- [1]
            angle_mean {torch.FloatTensor} -- [batch_size x channel_size]
            angle_std {torch.FloatTensor} -- [1]
        
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

        force_norm = force_field.norm(dim=0).view(1, 1, height, width).repeat(batch_size, channel_size, 1, 1)

        radius_mean = radius_mean.abs().view(batch_size, channel_size, 1, 1)
        radius_dist = torch.exp(-(force_norm - radius_mean)**2 / 2 / radius_std**2)

        # batch_size x channel_size x 2
        angle_mean_force = torch.stack([fgcos(angle_mean), fgsin(angle_mean)], dim=2)
        # Use selblock to fix output and zero grad at the origin point
        ang_dis_cos = selblock(torch.mm(angle_mean_force.view(-1, 2), force_field.view(2, -1)).view(force_norm.size()), (slice(None), slice(None), cy, cx), 1, 0) / \
                      selblock(force_norm, (slice(None), slice(None), cy, cx), 1, 0)
        # FIXME: TEST IF selblock CHANGED ORIGINAL DATA
        assert (force_norm.data[:, :, cy, cx] == 1).all()
        # sometimes rounding error can be twice the float32_eps
        assert not (ang_dis_cos.data > 1 + 2 * self._float32_eps).any() and not (ang_dis_cos.data < -1 - 2 * self._float32_eps).any()
        # This is intentional. The outsiders should only be caused by rounding error.
        # We don't want their gradient being eliminated.
        mask_upper = (ang_dis_cos.data > 1)
        mask_lower = (ang_dis_cos.data < -1)
        ang_dis_cos[mask_upper] -= ang_dis_cos.data[mask_upper] - 1
        ang_dis_cos[mask_lower] -= ang_dis_cos.data[mask_lower] + 1

        ang_dist = torch.exp(-fgacos(ang_dis_cos)**2 / 2 / angle_std**2)

        dist = radius_dist * ang_dist
        return dist

    def forward(self, origin_x, origin_y, radius, angle, confidence, height=None, width=None):
        """
        Arguments:
            origin_x {torch.FloatTensor} -- [nw]
            origin_y {torch.FloatTensor} -- [nh]
            radius {torch.FloatTensor} -- [nb x nh x nw x nchan]
            angle {torch.FloatTensor} -- [nb x nh x nw x nchan]
            confidence {torch.FloatTensor} -- [nb x nh x nw x nchan]
        """
        batch_size = radius.size(0)
        channel_size = radius.size(3)
        if height is not None and width is not None:
            self._init_buffer(height, width, device=radius.device)
        out = None
        # TODO: random drop out to ease training (when random, should we disable confidence?)
        for iy in range(radius.size(1)):
            force_y = self.y_out - origin_y[iy]
            for ix in range(radius.size(2)):
                force_x = self.x_out - origin_x[ix]
                force_field = torch.stack([force_x.repeat(self.height, 1), force_y.repeat(1, self.width)], dim=0)
                proj = confidence[:, iy, ix].view(batch_size, channel_size, 1, 1) * \
                       self._proj(force_field, origin_x[ix], origin_y[iy], radius[:, iy, ix], self.radius_std, angle[:, iy, ix], self.angle_std)
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
        self.conf_regressor = nn.Sequential(nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=True),
                                            nn.Sigmoid())
        # TODO: improve initialization
        self.projector = LongRangeProj(radius_std_init=10)
        self.n_corr_h = None
        self.n_corr_w = None
        self.in_channels = in_channels
        self.out_channels = out_channels

        # For finetune
        # TODO: improve initialization
        self.radius_regressor.weight.data.zero_()
        self.radius_regressor.bias.data.uniform_(0, 0.5)
        self.angle_regressor.weight.data.uniform_(-1e-3, 1e-3)
        self.angle_regressor.bias.data.uniform_(-np.pi, np.pi)
        self.conf_regressor[0].weight.data.zero_()
        self.conf_regressor[0].bias.data.zero_()

    def _init_buffer(self, n_corr_h, n_corr_w, device=torch.device("cpu")):
        if self.n_corr_w != n_corr_w or (hasattr(self, "origin_x") and self.origin_x.device != device):
            self.n_corr_w = n_corr_w
            origin_x = torch.arange(n_corr_w, dtype=torch.float32, device=device)
            origin_x = origin_x * self.acorr2d.corr_stride[1] - self.acorr2d.pad.left + self.acorr2d.corr_kernel_size[1] // 2
            self.register_buffer("origin_x", origin_x)

        if self.n_corr_h != n_corr_h or (hasattr(self, "origin_y") and self.origin_y.device != device):
            self.n_corr_h = n_corr_h
            origin_y = torch.arange(n_corr_h, dtype=torch.float32, device=device)
            origin_y = origin_y * self.acorr2d.corr_stride[0] - self.acorr2d.pad.top + self.acorr2d.corr_kernel_size[0] // 2
            self.register_buffer("origin_y", origin_y)

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
        conf = self.conf_regressor(corrs).view(batch_size, n_corr_h, n_corr_w, self.out_channels)

        return radius, angle, conf, n_corr_h, n_corr_w

    def forward(self, inp):
        radius, angle, conf, n_corr_h, n_corr_w = self._regress(inp)
        self._init_buffer(n_corr_h, n_corr_w, device=inp.device)

        out = self.projector(self.origin_x, self.origin_y, radius, angle, conf, height=inp.size(2), width=inp.size(3))
        return out
