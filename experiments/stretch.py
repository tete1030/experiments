from __future__ import print_function, absolute_import
import torch
import pose.models as models
import pose.datasets as datasets
from pose.utils.transforms import fliplr_chwimg, fliplr_map
from pose.utils.evaluation import accuracy
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate

"""
1. Detect person location / head
2. Generate maps of part location and embedding
3. Input into network, output vector fields
4. Update part location
5. A: If iterating counter less than M, goto 2 (Stack Sharing/Non-sharing)
   B: If stable, goto 2 (Stack Non-sharing)

# Keypoint map fusion
- Could banlance semantic for image and keypoint maps

# Loss
- from loose to tight / equal

Input:
    A:
        image map: 128
        fuse map: 128
        keypoint map: 2 * 17
    B:
        image/fuse map: 256
        keypoint map: 2 * 17
    C:
        image/fuse map: 256
        keypoint map: 256 (after fusion)

Output: 
    offset field: 2 * 17
"""

class Experiment:
    def __init__(self, hparams):
        model = models.PoseNet(hparams["model"]["inp_dim"], )
