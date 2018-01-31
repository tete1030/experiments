import matplotlib.pyplot as plt
import pose.datasets as datasets
from pose.models import PoseMapGenerator
import numpy as np
import cv2
import torch

train_dataset = datasets.COCOPose('data/mscoco/images',
                                  'data/mscoco/person_keypoints_train2014.json',
                                  None,
                                  # 'data/mscoco/split.pth',
                                  None,
                                  # 'data/mscoco/mean_std.pth',
                                  train=True,
                                  single_person=False)

pose_gen = PoseMapGenerator(16, 17, (64, 64), 30)

ids = []
imgs = []
keypoints = []

i = 0
while True:
    img, target, mask, extra = train_dataset[i]
    kp_i = extra["keypoints_tf"]
    if kp_i is not None:
        ids.append(extra["index"])
        imgs.append(img)
        keypoints.append(kp_i)
        if len(imgs) >= 16:
            break
    i += 1

batch_ids, person_ids, part_ids, keypoints_flat = pose_gen.init_ids(keypoints)

map_res = pose_gen(batch_ids, person_ids, part_ids, keypoints_flat)

map_res = map_res.data.numpy()

for i, img in enumerate(imgs):

    fig, axs = plt.subplots(5, 8, figsize=(30, 15), gridspec_kw={"hspace": 0.1, "wspace": 0.02, "top": 0.97, "bottom": 0})

    img = img.numpy().transpose(1,2,0)

    for ax in axs.flat:
        ax.axis('off')

    axs.flat[0].imshow(img, vmin=0, vmax=1)

    for ijoint in range(datasets.mscoco.NUM_PARTS):
        axs.flat[1+ijoint*2].set_title(datasets.mscoco.PART_LABELS[ijoint], fontdict={"fontsize": 8})
        axs.flat[1+ijoint*2+1].set_title(datasets.mscoco.PART_LABELS[ijoint], fontdict={"fontsize": 8})
        # axs.flat[1+ijoint].imshow(img, vmin=0, vmax=1)
        axs.flat[1+ijoint*2].imshow(map_res[i, ijoint], vmin=0, vmax=1)
        axs.flat[1+ijoint*2+1].imshow(map_res[i, datasets.mscoco.NUM_PARTS+ijoint], vmin=0, vmax=30)

    plt.show()
