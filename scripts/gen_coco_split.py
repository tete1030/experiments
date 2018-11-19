from pycocotools.coco import COCO
import sys
import torch

if len(sys.argv) != 2:
    print("Argument wrong. command: gen_coco_split anno_file")
    sys.exit(1)

with open("pae_coco_valid_id", "r") as f:
    valid_id = list(map(lambda x:int(x.strip()), f.readlines()))

coco = COCO(sys.argv[1])
imgids = coco.getImgIds()
train = list()
valid = list()
for imgid in imgids:
    ann_ids = coco.getAnnIds(imgIds=imgid)
    anns = coco.loadAnns(ann_ids)
    ann_count = 0
    for ann in anns:
        if ann["num_keypoints"] > 0:
            ann_count += 1
            break
    if ann_count > 0 and imgid not in valid_id:
        train.append(imgid)
    elif imgid in valid_id:
        valid.append(imgid)
        
torch.save({"train": train, "valid": valid}, "coco_split.pth")