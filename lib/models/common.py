import torch.nn as nn

# Update: fixed in pytorch 1.0
# as of pytorch 0.4.1 , nn.ReLU still give output 0 when input NaN
# This is hard for debug, so we replace it
# https://github.com/pytorch/pytorch/issues/10238
# nn.functional.relu(_) is correctly behaved (fixed in pytorch/pytorch#8033),
# while nn.ReLU uses nn.functional.threshold(_), which is not fixed
class StrictNaNReLU(nn.Module):
    def __init__(self, inplace=False):
        super(StrictNaNReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return nn.functional.relu_(x)
        return nn.functional.relu(x)
