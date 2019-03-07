import math
import numbers
import torch
from torch import nn
from torch.autograd import Function
from torch.distributions import Normal, Uniform
from utils.globals import config, globalvars
from utils.log import log_i
import torch.nn.functional as F
import numpy as np
from .displace import DisplaceFracCUDA, PositionalDisplace, PositionalGaussianDisplace
from lib.utils.lambdalayer import Lambda
import math

class Weighted(nn.Module):
    def __init__(self, num_channels, init=0.):
        super(Weighted, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_channels, dtype=torch.float))
        if isinstance(init, (float, int)):
            self.weight.data.fill_(float(init))
        elif isinstance(init, torch.FloatTensor):
            self.weight.data.copy_(init)
        else:
            raise ValueError("Wrong initialization")

    def forward(self, x):
        assert x.dim() == 2
        return x * self.weight[None]

class OffsetRegressor(nn.Module):
    def __init__(self, num_offsets):
        super(OffsetRegressor, self).__init__()
        self.num_offsets = num_offsets
        self.regressor = nn.Linear(self.num_offsets * 2, self.num_offsets, bias=False)
        self.regressor.weight.data.zero_()

    def forward(self, inp, atten):
        pos_inp = inp.abs().view(inp.size(0), inp.size(1), -1).argmax(dim=-1)
        pos_inp_x = (pos_inp % inp.size(-1)).float()
        pos_inp_y = (pos_inp / inp.size(-1)).float()
        pos_atten = atten.view(atten.size(0), atten.size(1), -1).argmax(dim=-1)
        pos_atten_x = (pos_atten % atten.size(-1)).float()
        pos_atten_y = (pos_atten / atten.size(-1)).float()
        return torch.stack([
            self.regressor(torch.cat([pos_inp_x, pos_atten_x], dim=-1)),
            self.regressor(torch.cat([pos_inp_y, pos_atten_y], dim=-1))], dim=2)

class TransformCoordinate(Function):
    @staticmethod
    def forward(ctx, offsets_x, offsets_y, angle_ksin, angle_kcos):
        offsets_x_new = ((angle_kcos * offsets_x) - (angle_ksin * offsets_y))
        offsets_y_new = ((angle_ksin * offsets_x) + (angle_kcos * offsets_y))
        ctx.save_for_backward(offsets_x, offsets_y, angle_ksin, angle_kcos)
        return offsets_x_new, offsets_y_new

    @staticmethod
    def backward(ctx, grad_offsets_x_new, grad_offsets_y_new):
        offsets_x, offsets_y, angle_ksin, angle_kcos = ctx.saved_tensors
        grad_offsets_x = grad_offsets_x_new * angle_kcos + grad_offsets_y_new * angle_ksin
        grad_offsets_y = -grad_offsets_x_new * angle_ksin + grad_offsets_y_new * angle_kcos
        grad_angle_ksin = -grad_offsets_x_new * offsets_y + grad_offsets_y_new * offsets_x
        grad_angle_kcos = grad_offsets_x_new * offsets_x + grad_offsets_y_new * offsets_y

        return grad_offsets_x, grad_offsets_y, grad_angle_ksin, grad_angle_kcos

class OffsetTransformer(nn.Module):
    def __init__(self, inplanes, num_offsets, bottleneck=None, num_regress=None, scale_grow_step=None, absolute_regressor=False):
        super(OffsetTransformer, self).__init__()
        self.inplanes = inplanes
        self.num_offsets = num_offsets
        self.bottleneck = bottleneck
        self.num_regress = num_regress
        self.use_absolute_regressor = absolute_regressor
        if not self.use_absolute_regressor:
            first_layer_channels = (self.num_regress if self.num_regress else num_offsets) if bottleneck is None else bottleneck
            use_bias = False if bottleneck is None else True

            scale_regressor_mods = []
            scale_regressor_mods.append(nn.Conv2d(inplanes, first_layer_channels, kernel_size=3, padding=1, bias=use_bias))
            if bottleneck is not None:
                scale_regressor_mods.append(nn.Conv2d(bottleneck, num_offsets, kernel_size=1, bias=False))
            scale_regressor_mods.append(nn.Softsign())
            self.scale_regressor = nn.Sequential(*scale_regressor_mods)

            angle_regressor_mods = []
            angle_regressor_mods.append(nn.Conv2d(inplanes, first_layer_channels, kernel_size=3, padding=1, bias=use_bias))
            if bottleneck is not None:
                angle_regressor_mods.append(nn.Conv2d(bottleneck, num_offsets, kernel_size=1, bias=False))
            angle_regressor_mods.append(nn.Softsign())
            self.angle_regressor = nn.Sequential(*angle_regressor_mods)
        else:
            num_last_inp_channels = inplanes if bottleneck is None else bottleneck
            num_last_out_channels = (self.num_regress if self.num_regress else num_offsets) if bottleneck is None else num_offsets

            scale_regressor_mods = []
            if bottleneck is not None:
                scale_regressor_mods.append(nn.Conv2d(inplanes, bottleneck, kernel_size=1, bias=False))
            scale_regressor_mods.append(nn.Conv2d(num_last_inp_channels, num_last_out_channels, kernel_size=1, bias=False))
            scale_regressor_mods.append(nn.BatchNorm2d(num_last_out_channels))
            scale_regressor_mods.append(nn.Softsign())
            self.scale_regressor = nn.Sequential(*scale_regressor_mods)

            if bottleneck is None:
                self.pre_angle_regressor = lambda x: x
                num_last_inp_channels = inplanes
            else:
                self.pre_angle_regressor = nn.Conv2d(inplanes, bottleneck * 2, kernel_size=1, bias=False)
                num_last_inp_channels = bottleneck * 2

            self.angle_x_regressor = nn.Sequential(
                nn.Conv2d(num_last_inp_channels, num_last_out_channels, kernel_size=1, bias=False))
            self.angle_y_regressor = nn.Sequential(
                nn.Conv2d(num_last_inp_channels, num_last_out_channels, kernel_size=1, bias=False))

            self.angle_bias = nn.Parameter(torch.zeros(num_offsets, dtype=torch.float))

        if scale_grow_step is not None:
            self.register_buffer("effect_scale", torch.zeros(1, dtype=torch.float))
            self.register_buffer("scale_grow_step", torch.tensor(scale_grow_step, dtype=torch.float))
        else:
            self.effect_scale = None
            self.scale_grow_step = None

    def visualize(self, scale, angle):
        import matplotlib.pyplot as plt
        batch_size = scale.size(0)
        assert globalvars.main_context.exp.data_source == "coco"
        img = globalvars.main_context.exp.val_dataset.restore_image(globalvars.cur_img.cpu().numpy())
        scale = scale.detach().cpu().numpy()
        angle = angle.detach().cpu().numpy()
        for i in range(batch_size):
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            axes[0].imshow(img[i])
            axes[1].imshow(scale[i, 0], vmin=-1, vmax=1)
            axes[2].imshow(angle[i, 0], vmin=-1, vmax=1)
            fig.show()
            plt.show()

    def forward(self, x, offsets, spatial_size, use_effect_scale=True):
        offset_dim = offsets.dim()
        offset_size = offsets.size()
        if offset_dim == 2:
            offsets_x = offsets[:, 0]
            offsets_y = offsets[:, 1]
        elif offset_dim == 3:
            offsets_x = offsets[:, :, 0]
            offsets_y = offsets[:, :, 1]

        use_effect_scale = bool(use_effect_scale and self.effect_scale is not None and (self.effect_scale < 1).all())

        assert len(spatial_size) == 2
        if x.size()[-2:] != spatial_size:
            height_ori = x.size(-2)
            width_ori = x.size(-1)
            height_new = spatial_size[0]
            width_new = spatial_size[1]
            assert height_new / height_ori == width_new / width_ori and height_new > height_ori
            x = F.interpolate(x, size=spatial_size, mode="bilinear", align_corners=True)

        if not self.use_absolute_regressor:
            scale = self.scale_regressor(x)
            angle = self.angle_regressor(x)
            if use_effect_scale:
                scale = scale * self.effect_scale
                angle = angle * self.effect_scale
            if config.vis:
                self.visualize(scale, angle)

            scale = 1 + scale
            angle = angle * np.pi
            angle_ksin = angle.sin() * scale
            angle_kcos = angle.cos() * scale

            scale_size = scale.size()
        else:
            scale = 1 + self.scale_regressor(x)
            pre_angle = self.pre_angle_regressor(x)
            angle_kcos = self.angle_x_regressor(pre_angle)
            angle_ksin = self.angle_y_regressor(pre_angle)
            angle_knorm = torch.stack([angle_kcos, angle_ksin], dim=0).norm(dim=0) / scale
            angle_kcos = angle_kcos / angle_knorm
            angle_ksin = angle_ksin / angle_knorm

            angle_bias_cos = self.angle_bias.cos().expand_as(offsets_x)
            angle_bias_sin = self.angle_bias.sin().expand_as(offsets_x)

            offsets_x, offsets_y = TransformCoordinate.apply(offsets_x, offsets_y, angle_bias_sin, angle_bias_cos)
            scale_size = scale.size()

        if angle_ksin.size(1) != offset_size[-2]:
            assert offset_size[-2] % angle_ksin.size(1) == 0, "Number of offsets should be interger times of number of regress results"
            expand_size = offset_size[-2] // angle_ksin.size(1)
            angle_ksin = angle_ksin[:, :, None].repeat(1, 1, expand_size, 1, 1).view(angle_ksin.size(0), -1, angle_ksin.size(2), angle_ksin.size(3))
            angle_kcos = angle_kcos[:, :, None].repeat(1, 1, expand_size, 1, 1).view(angle_kcos.size(0), -1, angle_kcos.size(2), angle_kcos.size(3))

        if offset_dim == 2:
            offsets_x = offsets_x.view(1, -1, 1, 1).expand(scale_size[0], -1, scale_size[2], scale_size[3])
            offsets_y = offsets_y.view(1, -1, 1, 1).expand(scale_size[0], -1, scale_size[2], scale_size[3])
        elif offset_dim == 3:
            offsets_x = offsets_x.view(offset_size[0], offset_size[1], 1, 1).expand(-1, -1, scale_size[2], scale_size[3])
            offsets_y = offsets_y.view(offset_size[0], offset_size[1], 1, 1).expand(-1, -1, scale_size[2], scale_size[3])

        new_offsets_x, new_offsets_y = TransformCoordinate.apply(offsets_x, offsets_y, angle_ksin, angle_kcos)
        if self.use_absolute_regressor and use_effect_scale:
            new_offsets_x = new_offsets_x * self.effect_scale + offsets_x * (1-self.effect_scale)
            new_offsets_y = new_offsets_y * self.effect_scale + offsets_y * (1-self.effect_scale)

        return new_offsets_x, new_offsets_y

class PositionalGaussianDisplaceModule(nn.Module):
    NUM_TOTAL_SAMPLE = 32
    def __init__(self,
            num_offset, num_sample,
            angle_std, scale_std,
            min_angle_std=math.atan2(0.5, 10) / 2, max_angle_std=np.pi,
            min_scale_std=0.2, max_scale_std=5.,
            learnable_sigma=True, transform_sigma=True,
            sampler="uniform", weight_dist="gaussian",
            fill=0,
            simple=False,
            grad_off_divide_distance=True):
        super().__init__()
        self.num_offset = num_offset
        self.num_sample = num_sample

        self.learnable_sigma = learnable_sigma
        self.transform_sigma = transform_sigma
        self.max_angle_std = max_angle_std
        self.min_angle_std = min_angle_std
        self.max_scale_std = max_scale_std
        self.min_scale_std = min_scale_std

        self.set_angle_std(angle_std)
        self.set_scale_std(scale_std)

        assert sampler in ["gaussian", "uniform"]
        assert weight_dist in ["gaussian", "uniform"]
        assert (weight_dist == "gaussian") or not learnable_sigma
        self.sampler = sampler
        self.weight_dist = weight_dist

        self.fill = fill
        self.simple = simple
        self.grad_off_divide_distance = grad_off_divide_distance
        assert not self.simple or (self.fill == 0)

    def _set_std(self, stdname, stdval, device=None):
        stdmin = getattr(self, "min_" + stdname + "_std")
        stdmax = getattr(self, "max_" + stdname + "_std")
        if isinstance(stdval, numbers.Number):
            stdval = torch.tensor([stdval], dtype=torch.float)
        elif not isinstance(stdval, torch.Tensor) or stdval.dtype != torch.float:
            raise TypeError(stdname + "_std should be number or torch tensor")
        if device is not None:
            stdval = stdval.to(device=device, non_blocking=True)
        elif hasattr(self, "_" + stdname + "_std"):
            stdval = stdval.to(device=getattr(self, "_" + stdname + "_std").device, non_blocking=True)

        if stdval.dim() == 0 or stdval.size() != (self.num_offset,):
            stdval = stdval.repeat(self.num_offset)
        assert stdval.size() == (self.num_offset,)
        assert ((stdmin <= stdval) & (stdval <= stdmax)).all(), stdval

        if self.learnable_sigma:
            if self.transform_sigma:
                stdnorm = (stdval - stdmin) / (stdmax - stdmin)
                stdval = torch.log(stdnorm/(1-stdnorm))
            if hasattr(self, "_" + stdname + "_std"):
                getattr(self, "_" + stdname + "_std").copy_(stdval, non_blocking=True)
            else:
                setattr(self, "_" + stdname + "_std", nn.Parameter(stdval))
        else:
            if hasattr(self, "_" + stdname + "_std"):
                getattr(self, "_" + stdname + "_std").copy_(stdval, non_blocking=True)
            else:
                self.register_buffer("_" + stdname + "_std", stdval)

    def _get_std(self, stdname):
        stdval = getattr(self, "_" + stdname + "_std")
        minval = getattr(self, "min_" + stdname + "_std")
        maxval = getattr(self, "max_" + stdname + "_std")
        if self.learnable_sigma and self.transform_sigma:
            return stdval.sigmoid() * (maxval - minval) + minval
        else:
            return stdval

    def set_angle_std(self, angle_std):
        self._set_std("angle", angle_std)

    def set_scale_std(self, scale_std):
        self._set_std("scale", scale_std)

    def angle_std(self):
        return self._get_std("angle")

    def scale_std(self):
        return self._get_std("scale")

    def forward(self, x, offsets_x, offsets_y, channel_per_off):
        angle_std = self.angle_std()
        scale_std = self.scale_std()
        if self.sampler == "gaussian":
            angle_sampler = Normal(loc=0, scale=angle_std)
            scale_sampler = Normal(loc=0, scale=scale_std)
        elif self.sampler == "uniform":
            _3angle_std_clamped = (angle_std * 3).clamp(max=np.pi)
            angle_sampler = Uniform(low=-_3angle_std_clamped, high=_3angle_std_clamped)
            # angle_sampler = Uniform(low=-angle_std * 3, high=angle_std * 3)
            scale_sampler = Uniform(low=-scale_std * 3, high=scale_std * 3)
        
        if self.weight_dist == "gaussian":
            angles = angle_sampler.sample(sample_shape=(max(self.num_sample, PositionalGaussianDisplaceModule.NUM_TOTAL_SAMPLE),)).t().contiguous()
            scales = scale_sampler.sample(sample_shape=(max(self.num_sample, PositionalGaussianDisplaceModule.NUM_TOTAL_SAMPLE),)).t().contiguous()
            weight = (- angles.pow(2) / 2 / (angle_std.pow(2)[:, None] + np.finfo(np.float32).eps.item()) - scales.pow(2) / 2 / (scale_std.pow(2)[:, None] + np.finfo(np.float32).eps.item())).exp()
            # weight * 3\sigma_1 x 3\sigma_2 / N / sqrt(2PI)\sigma_1 / sqrt(2PI)\sigma_2
            # weight = weight * (9. / (2. * np.pi * self.num_sample))
            # if self.sampler == "uniform":
            #     weight = weight * (_3angle_std_clamped / 3 / angle_std)[:, None]
            weight = weight / (weight.sum(dim=1, keepdim=True) + np.finfo(np.float32).eps.item())
            if self.num_sample < PositionalGaussianDisplaceModule.NUM_TOTAL_SAMPLE:
                weight = weight[:, :self.num_sample] * (PositionalGaussianDisplaceModule.NUM_TOTAL_SAMPLE / float(self.num_sample))
        elif self.weight_dist == "uniform":
            angles = angle_sampler.sample(sample_shape=(self.num_sample,)).t().contiguous()
            scales = scale_sampler.sample(sample_shape=(self.num_sample,)).t().contiguous()
            weight = torch.ones_like(angles) / self.num_sample

        return PositionalGaussianDisplace.apply(
            x, offsets_x, offsets_y, channel_per_off, angles, scales, weight, self.fill, self.simple, self.grad_off_divide_distance)

class DisplaceChannel(nn.Module):
    def __init__(self, height, width, num_channels, num_offsets,
                 disable_displace=False, learnable_offset=False, offset_scale=None,
                 regress_offset=False, transformer=None,
                 half_reversed_offset=False, previous_dischan=None,
                 arc_gaussian=None):
        super(DisplaceChannel, self).__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.num_offsets = num_offsets
        assert num_channels % num_offsets == 0, "num of channels cannot be divided by number of offsets"
        self.offset_scale = float(max(width, height)) if offset_scale is None else offset_scale
        self.disable_displace = disable_displace
        self.half_reversed_offset = half_reversed_offset
        self.regress_offset = regress_offset
        self.offset_transformer = transformer
        if arc_gaussian:
            self.arc_gaussian_displacer = arc_gaussian
        else:
            self.arc_gaussian_displacer = None
        if not disable_displace:
            self.previous_dischan = previous_dischan
            num_offsets_prev = 0
            if previous_dischan is not None:
                assert isinstance(previous_dischan, DisplaceChannel)
                assert self.random_offset_init is not None
                num_offsets_prev = self.previous_dischan.get_all_offsets_num()

            assert self.num_offsets >= num_offsets_prev
            self.offset = nn.parameter.Parameter(torch.zeros(self.num_offsets - num_offsets_prev, 2), requires_grad=True)
            self.switch_LO_state(learnable_offset)

            if regress_offset:
                self.offset_regressor = OffsetRegressor(self.num_offsets)
        else:
            self.switch_LO_state(False)

    def switch_LO_state(self, active):
        if active:
            log_i("Learnable offset is enabled")
        else:
            log_i("Learnable offset is disabled")
        self.learnable_offset = active

    def reset_outsider(self):
        if self.offset.size(0) == 0:
            return

        # At lease 2x2 pixels for receiving grad
        self.offset.data[:, 0].clamp_(min=(-self.width + 2.1) / self.offset_scale,
                                      max=(self.width - 2.1) / self.offset_scale)
        self.offset.data[:, 1].clamp_(min=(-self.height + 2.1) / self.offset_scale,
                                      max=(self.height - 2.1) / self.offset_scale)

    def get_all_offsets_num(self):
        offsets_num = self.offset.size(0)
        if self.previous_dischan is not None:
            offsets_num += self.previous_dischan.get_all_offsets_num()
        return offsets_num

    def get_all_offsets(self, detach=False, cat=True):
        if self.offset.size(0) == 0:
            all_offsets = []
        else:
            all_offsets = [self.offset.detach() if detach else self.offset]

        if self.previous_dischan is not None:
            all_offsets = self.previous_dischan.get_all_offsets(detach=detach, cat=False) + all_offsets
            if cat:
                all_offsets = torch.cat(all_offsets, dim=0)
        elif cat:
            all_offsets = all_offsets[0]

        return all_offsets

    def forward(self, inp, detach_offset=False, offset_plus_rel=None, transformer_source=None):
        batch_size = inp.size(0)
        num_channels = inp.size(1)
        height = inp.size(2)
        width = inp.size(3)
        device = inp.device
        # assert self.height == height and self.width == width and self.num_channels == num_channels
        assert offset_plus_rel is None or offset_plus_rel.size(1) == self.num_offsets

        if not self.learnable_offset:
            detach_offset = True

        if not self.disable_displace:
            offset_rel = self.get_all_offsets(detach=detach_offset)

            if offset_plus_rel is not None:
                offset_rel = offset_rel[None] + offset_plus_rel

            offset_abs = offset_rel * self.offset_scale

            if self.regress_offset:
                raise ValueError("No atten map provided")
                if offset_abs.dim() == 2:
                    offset_abs = offset_abs[None]
                offset_regressed_abs = self.offset_regressor(inp, None)
                offset_abs = offset_abs + offset_regressed_abs

            bind_chan = num_channels // self.num_offsets
            if self.offset_transformer is not None:
                offset_abs_x, offset_abs_y = self.offset_transformer(inp if transformer_source is None else transformer_source, offset_abs, inp.size()[-2:])
                
                if self.half_reversed_offset:
                    assert bind_chan % 2 == 0
                    bind_chan = bind_chan // 2
                    offset_abs_x = torch.cat([offset_abs_x, -offset_abs_x], dim=1)
                    offset_abs_y = torch.cat([offset_abs_y, -offset_abs_y], dim=1)
                
                if detach_offset:
                    offset_abs_x = offset_abs_x.detach()
                    offset_abs_y = offset_abs_y.detach()
                if self.arc_gaussian_displacer is None:
                    out = PositionalDisplace.apply(inp, offset_abs_x, offset_abs_y, bind_chan)
                else:
                    out = self.arc_gaussian_displacer(inp, offset_abs_x, offset_abs_y, bind_chan)
            else:
                offset_dim = offset_abs.dim()
                if self.half_reversed_offset:
                    assert bind_chan % 2 == 0
                    bind_chan = bind_chan // 2
                    if offset_dim in [2, 3]:
                        offset_abs = torch.cat([offset_abs, -offset_abs], dim=-2)
                    else:
                        raise ValueError()

                if detach_offset:
                    offset_abs = offset_abs.detach()

                if self.arc_gaussian_displacer:
                    if offset_dim == 2:
                        offset_abs_x = offset_abs[None, :, None, None, 0].repeat(batch_size, 1, height, width)
                        offset_abs_y = offset_abs[None, :, None, None, 1].repeat(batch_size, 1, height, width)
                    elif offset_dim == 3:
                        offset_abs_x = offset_abs[:, :, None, None, 0].repeat(1, 1, height, width)
                        offset_abs_y = offset_abs[:, :, None, None, 1].repeat(1, 1, height, width)
                    else:
                        raise ValueError()
                    out = self.arc_gaussian_displacer(inp, offset_abs_x, offset_abs_y, bind_chan)
                elif offset_dim == 2:
                    out = DisplaceFracCUDA.apply(inp, offset_abs, bind_chan)
                elif offset_dim == 3:
                    out = DisplaceFracCUDA.apply(inp.view(1, -1, height, width), offset_abs.view(-1, 2), bind_chan).view(batch_size, -1, height, width)
                else:
                    raise ValueError()

        else:
            out = inp

        if config.vis and False:
            import matplotlib.pyplot as plt
            import cv2
            fig, axes = plt.subplots(3, 30, figsize=(100, 12), squeeze=False)
            
            for row, axes_row in enumerate(axes):
                img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                fts = inp.data[row].cpu().numpy()
                for col, ax in enumerate(axes_row):
                    if col == 0:
                        ax.imshow(img)
                    else:
                        ax.imshow(fts[col-1])
            fig.suptitle("displace inp")

            fig, axes = plt.subplots(3, 30, figsize=(100, 12), squeeze=False)
            for row, axes_row in enumerate(axes):
                img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                fts = out.data[row].cpu().numpy()
                for col, ax in enumerate(axes_row):
                    if col == 0:
                        ax.imshow(img)
                    else:
                        ax.imshow(fts[col-1])
            fig.suptitle("displace out")
            plt.show()

        return out
