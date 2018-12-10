import matplotlib.pyplot as plt
import lib.datasets as datasets
import numpy as np
import cv2


train_dataset = datasets.COCOPose('data/mscoco/images',
                                  'data/mscoco/person_keypoints_train2014.json',
                                  None,
                                  # 'data/mscoco/split.pth',
                                  None,
                                  # 'data/mscoco/mean_std.pth',
                                  train=True,
                                  single_person=False)

spec_index = None #23002

for i in range(50):
    fig, axs = plt.subplots(4, 5, figsize=(20, 15), gridspec_kw={"hspace": 0.1, "wspace": 0.02, "top": 0.97, "bottom": 0})
    index = spec_index if spec_index else np.random.randint(0, len(train_dataset))
    img, target, mask, extra = train_dataset[index]
    print(index)

    img = img.numpy().transpose(1,2,0)

    target = target.numpy()
    target_tf = np.zeros((target.shape[0], img.shape[0], img.shape[1]))
    for ijoint in range(target.shape[0]):
        target_tf[ijoint] = cv2.resize((target[ijoint] * 255).clip(0,255).astype(np.uint8), target_tf.shape[-2:]).astype(np.float32) / 255

    mask = mask.numpy()
    mask = ~(mask.astype(np.uint8) > 0.5)
    mask = cv2.resize((mask.astype(np.uint8) * 255), img.shape[:2]).astype(np.float32) / 255
    
    for iax in range(axs.size):
        axs.flat[iax].axis('off')
    
    masked_img = img.copy()
    masked_img[mask > 0.5] = np.array([0, 0, 1], dtype=np.float32)
    axs.flat[0].imshow(masked_img, vmin=0, vmax=1)
    for ijoint in range(target.shape[0]):
        axs.flat[1+ijoint].set_title(datasets.mscoco.PART_LABELS[ijoint], fontdict={"fontsize": 8})
        axs.flat[1+ijoint].imshow(img, vmin=0, vmax=1)
        axs.flat[1+ijoint].imshow(target_tf[ijoint], vmin=0, vmax=1, alpha=0.5)

    # fig.tight_layout()
    plt.show()
    if spec_index:
        break

#30844
#4105
#30764


# 5115
