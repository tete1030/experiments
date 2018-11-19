# coding: utf-8
from scipy.io import loadmat
import json
import numpy as np

def safe_execute(exception, function):
    is_safe = False
    try:
        function()
        is_safe = True
    except exception:
        is_safe = False
    return is_safe

mpii_data = loadmat("/home/bcmilht/datasets/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat")

mpii_new = {"annolist": list(), "img_train": list(), "single_person": list(), "act": list(), "video_list": list()}
annolist = mpii_data["RELEASE"][0,0]["annolist"][0]
for i in range(len(annolist)):
    annolist_item_new = dict()
    mpii_new["annolist"].append(annolist_item_new)
    assert isinstance(annolist[i]["image"][0,0]["name"][0], unicode)
    annolist_item_new["image"] = unicode(annolist[i]["image"][0,0]["name"][0])
    annolist_item_new["annorect"] = list()
    assert "vidx" not in annolist[i].dtype.names
    assert (len(annolist[i]["vididx"].flat) > 0) == (len(annolist[i]["frame_sec"].flat) > 0)
    if len(annolist[i]["vididx"].flat) > 0:
        assert isinstance(annolist[i]["vididx"][0,0], np.unsignedinteger)
        annolist_item_new["vididx"] = int(annolist[i]["vididx"][0,0])
        assert isinstance(annolist[i]["frame_sec"][0,0], np.unsignedinteger)
        annolist_item_new["frame_sec"] = int(annolist[i]["frame_sec"][0,0])
    else:
        annolist_item_new["vididx"] = None
        annolist_item_new["frame_sec"] = None
    annorect_new = annolist_item_new["annorect"]
    if annolist[i]["annorect"].size > 0:
        annorect = annolist[i]["annorect"][0]
    else:
        annorect = []
    is_train = bool(mpii_data["RELEASE"][0,0]["img_train"][0,i])
    for j in range(len(annorect)):
        annorect_item_new = dict()
        annorect_new.append(annorect_item_new)
        if is_train:
            assert isinstance(annorect[j]["x1"][0,0], np.unsignedinteger) and isinstance(annorect[j]["y1"][0,0], np.unsignedinteger) and isinstance(annorect[j]["x2"][0,0], np.unsignedinteger) and isinstance(annorect[j]["y2"][0,0], np.unsignedinteger)
            annorect_item_new["head_rect"] = [[int(annorect[j]["x1"][0,0]), int(annorect[j]["y1"][0,0])], [int(annorect[j]["x2"][0,0]), int(annorect[j]["y2"][0,0])]]
        else:
            assert annorect[j] is None or len({"x1", "x2", "y1", "y2"} & set(annorect[j].dtype.names)) == 0

        if annorect[j] is not None and "scale" in annorect[j].dtype.names and annorect[j]["scale"].size > 0:
            assert isinstance(annorect[j]["scale"][0,0], (np.unsignedinteger,float))
            annorect_item_new["scale"] = float(annorect[j]["scale"][0,0])
            assert isinstance(annorect[j]["objpos"][0,0]["x"][0,0], np.integer) and isinstance(annorect[j]["objpos"][0,0]["y"][0,0], np.integer)
            annorect_item_new["objpos"] = [int(annorect[j]["objpos"][0,0]["x"][0,0]), int(annorect[j]["objpos"][0,0]["y"][0,0])]
            if is_train:
                annorect_item_new["annopoints"] = dict()
                point = annorect[j]["annopoints"][0,0]["point"][0]
                for k in range(len(point)):
                    if "is_visible" not in point[k].dtype.names:
                        is_visible = None
                    elif point[k]["is_visible"].size == 0:
                        is_visible = None
                    elif point[k]["is_visible"].dtype == '<U1':
                        is_visible = int(point[k]["is_visible"][0])
                    else:
                        is_visible = int(point[k]["is_visible"][0,0])
                    assert is_visible is None or (type(is_visible) is int and is_visible in [0,1])
                    assert isinstance(point[k]["x"][0,0], (np.integer,float)) and isinstance(point[k]["y"][0,0], (np.integer,float))
                    annorect_item_new["annopoints"][int(point[k]["id"][0,0])] = \
                            [float(point[k]["x"][0,0]), float(point[k]["y"][0,0]), is_visible]
        else:
            is_scale_safe = safe_execute((TypeError,IndexError,ValueError), lambda: annorect[j]["scale"][0,0])
            is_objpos_safe = safe_execute((TypeError,IndexError,ValueError), lambda: annorect[j]["objpos"][0,0]["x"])
            is_annopoints_safe = safe_execute((TypeError,IndexError,ValueError), lambda: annorect[j]["annopoints"][0,0]["point"])
            assert (not is_scale_safe) and (not is_objpos_safe) and (not is_annopoints_safe)
            annorect_item_new["scale"] = None
            annorect_item_new["objpos"] = None
            if is_train:
                annorect_item_new["annopoints"] = None

img_train = mpii_data["RELEASE"][0,0]["img_train"][0]
for i in range(len(img_train)):
    assert isinstance(img_train[i], np.unsignedinteger)
    mpii_new["img_train"].append(int(img_train[i]))

single_person = mpii_data["RELEASE"][0,0]["single_person"][:,0]
for i in range(len(single_person)):
    assert isinstance(single_person[i], np.unsignedinteger) or (type(single_person[i]) is np.ndarray)
    if isinstance(single_person[i], np.unsignedinteger):
        mpii_new["single_person"].append([int(single_person[i])])
    else:
        if single_person[i].size > 0:
            assert single_person[i].shape[1] == 1 and single_person[i].dtype == np.uint8
            mpii_new["single_person"].append(list(single_person[i][:,0].astype(np.int)))
        else:
            mpii_new["single_person"].append([])

act = mpii_data["RELEASE"][0,0]["act"][:, 0]
for i in range(len(single_person)):
    if act[i]["act_id"][0, 0] != -1:
        assert act[i]["act_name"].size == 1 and act[i]["cat_name"].size == 1
        assert isinstance((act[i]["act_name"][0]),unicode) and isinstance(act[i]["cat_name"][0],unicode) and isinstance(act[i]["act_id"][0,0], np.unsignedinteger)
        mpii_new["act"] = dict(act_name=unicode(act[i]["act_name"][0]), cat_name=unicode(act[i]["cat_name"][0]), \
            act_id=int(act[i]["act_id"][0,0]))
    else:
        assert act[i]["act_name"].size == 0 and act[i]["cat_name"].size == 0
        mpii_new["act"] = None

video_list = mpii_data["RELEASE"][0,0]["video_list"][0]
for i in range(len(video_list)):
    assert isinstance(video_list[i][0], unicode)
    mpii_new["video_list"] = unicode(video_list[i][0])

json.dump(mpii_new, open("mpii_human_pose.json", "w"))
