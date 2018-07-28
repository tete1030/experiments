import torch.nn as nn
from pose.models.lacorr2d import LocalAutoCorr2DCUDA, PadInfo

class AutoCorr2D(nn.Module):
    def __init__(self, in_channels, out_channels, corr_channels, corr_kernel_size):
        super(AutoCorr2D, self).__init__()
        assert isinstance(corr_kernel_size, int) or (isinstance(corr_kernel_size, tuple) and len(corr_kernel_size) == 2)
        if isinstance(corr_kernel_size, int):
            corr_kernel_size = (int(corr_kernel_size), int(corr_kernel_size))
        else:
            corr_kernel_size = (int(corr_kernel_size[0]), int(corr_kernel_size[1]))
        # assert isinstance(corr_stride, int) or (isinstance(corr_stride, tuple) and len(corr_stride) == 2)
        # if isinstance(corr_stride, int):
        #     corr_stride = (corr_stride, corr_stride)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.corr_channels = corr_channels
        self.corr_kernel_size = corr_kernel_size
        self.extract_input = nn.Conv2d(in_channels, corr_channels, kernel_size=3, padding=1)
        self.corr2d = LocalAutoCorr2DCUDA(corr_kernel_size,
                                          (1, 1),
                                          PadInfo(top=corr_kernel_size[0] // 2,
                                                  bottom=(corr_kernel_size[0] + 1) // 2 - 1,
                                                  left=corr_kernel_size[1] // 2,
                                                  right=(corr_kernel_size[1] + 1) // 2 - 1))
        self.regressor = nn.Conv2d(corr_channels, out_channels, kernel_size=corr_kernel_size, bias=False)

        # For finetune
        self.extract_input.do_not_init = True
        self.regressor.do_not_init = True
        # self.regressor.weight.requires_grad_(False)
        self.regressor.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.extract_input(x)
        # corrs shape: b x ch x cw x chan x kh x kw
        corrs = self.corr2d(x)
        n_corr_h = corrs.size(1)
        n_corr_w = corrs.size(2)
        corrs = corrs.view(-1, self.corr_channels, self.corr_kernel_size[0], self.corr_kernel_size[1])
        # vec shape: b*ch*cw x 4 x 1 x 1
        vec = self.regressor(corrs)
        vec = vec.view(batch_size, n_corr_h, n_corr_w, vec.size(1))

        vec = vec.permute(0, 3, 1, 2)

        return vec
