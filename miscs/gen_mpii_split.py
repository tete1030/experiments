import numpy as np
from scipy.io import loadmat
import torch
import sys
import os

assert len(sys.argv) == 3

mpii_data = loadmat(os.path.join(sys.argv[1], "mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"))

Tompson_data = loadmat(os.path.join(sys.argv[2], "data/detections.mat"))
Tompson_img_index = Tompson_data["RELEASE_img_index"]
Tompson_person_index = Tompson_data["RELEASE_person_index"]

Tompson_i_p = np.r_[Tompson_img_index, Tompson_person_index]

train_list = list()
valid_list = list()
neither_list = list()
annolist = mpii_data["RELEASE"][0,0]["annolist"][0]
for i in range(annolist.shape[0]):
    annorect = annolist[i]["annorect"]
    if len(annorect.flat) < 1:
        continue
    for p in range(annorect[0].shape[0]):
        tomp = np.where((Tompson_i_p.T == (i+1, p+1)).all(axis=1))[0]
        person = annorect[0, p]
        try:
            person["annopoints"][0,0]["point"]
            isusable = True
        except (TypeError,ValueError,IndexError):
            isusable = False
        if tomp.shape[0] > 0 and isusable:
            valid_list.append((i,p))
        elif (i+1) not in Tompson_img_index[0] and isusable:
            train_list.append((i,p))
        else:
            neither_list.append((i,p))

print(len(train_list))
print(len(valid_list))
torch.save({"train": train_list, "valid": valid_list}, "mpii_split.pth")
