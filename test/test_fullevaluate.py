import sys
import os
import torch
import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from pose.utils.evaluation import filter_person_det, generate_person_det_ans, keypoints_nms

def generate_ans(image_ids, preds, scores, det_roi_scores=None, det_roi_use="no"):
    ans = []
    for sample_i in range(len(preds)):
        image_id = image_ids[sample_i]

        val = preds[sample_i]
        if det_roi_use == "no":
            score = scores[sample_i].mean()
        elif det_roi_use == "avg":
            score = (scores[sample_i].sum() + det_roi_scores[sample_i]) / (scores.shape[1] + 1)
        elif det_roi_use == "mul":
            score = scores[sample_i].mean() * det_roi_scores[sample_i]
        else:
            raise ValueError()
        tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(score)}
        # # p: average detected locations
        # p = val[val[:, 2] > 0][:, :2].mean(axis = 0)
        # for j in val:
        #     if j[2]>0.:
        #         tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
        #     else:
        #         # TRICK: for not detected points, place them at the average point
        #         tmp["keypoints"] += [float(p[0]), float(p[1]), 0]
        tmp["keypoints"] = val.ravel().tolist()
        ans.append(tmp)
    return ans

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-e", default="annotates_evaluate.pth")
    argparser.add_argument("-d", action="store_true")
    argparser.add_argument("--use-det-score", choices=["no", "avg", "mul"], default="no")
    argparser.add_argument("--person-det-min-score", type=float, default=1e-10)
    argparser.add_argument("--person-det-min-size", type=float, default=0)
    argparser.add_argument("--softnms", action="store_true")
    argparser.add_argument("--softnms-sigma", type=float, default=0.5)
    argparser.add_argument("--softnms-thres", type=float, default=0.001)
    argparser.add_argument("--oksnms", action="store_true")
    argparser.add_argument("--oksnms-thres", type=float, default=0.999)
    argparser.add_argument("--okssoftnms", action="store_true")
    argparser.add_argument("--okssoftnms-sigma", type=float, default=0.5)
    argparser.add_argument("--okssoftnms-thres", type=float, default=0.001)
    args = argparser.parse_args()

    coco = COCO("data/mscoco/person_keypoints_val2017.json")
    result = torch.load(os.path.join("checkpoint/offset/72/", args.e))
    image_ids = np.array(result["image_index"])
    preds = result["pred_affined"]
    pred_scores = result["pred_score"]
    rois = result["roi"]
    roi_scores = np.array(result["roi_score"])

    if args.d:
        import ipdb; ipdb.set_trace()

    print("Sorting...")
    indsort = np.argsort(image_ids, axis=0)
    image_ids = image_ids[indsort]
    preds = preds[indsort]
    pred_scores = pred_scores[indsort]
    rois = rois[indsort]
    roi_scores = roi_scores[indsort]

    if args.d:
        import ipdb; ipdb.set_trace()

    print("Generating det annotates...")
    det_anns = generate_person_det_ans(image_ids, rois, roi_scores)

    if args.d:
        import ipdb; ipdb.set_trace()

    if args.softnms:
        print("Soft-NMS...")
        indkeep, alldets = filter_person_det(det_anns, min_score=args.person_det_min_score, min_box_size=args.person_det_min_size, nms=True, return_ind=True, softnms_sigma=args.softnms_sigma, softnms_thres=args.softnms_thres)
        image_ids = image_ids[indkeep]
        preds = preds[indkeep]
        pred_scores = pred_scores[indkeep]
        rois = rois[indkeep]
        roi_scores = alldets[:, 4]

        if args.d:
            import ipdb; ipdb.set_trace()

    print("Generating person annotates...")
    anns = generate_ans(image_ids, preds, pred_scores, roi_scores, det_roi_use=args.use_det_score)

    if args.d:
        import ipdb; ipdb.set_trace()
    
    if args.oksnms:
        assert not args.okssoftnms
        print("Keypoint NMS...")
        anns = keypoints_nms(anns, rois, 17, method="nms", nms_thres=args.oksnms_thres)

        if args.d:
            import ipdb; ipdb.set_trace()
    elif args.okssoftnms:
        assert not args.oksnms
        print("Keypoint SoftNMS...")
        anns = keypoints_nms(anns, rois, 17, method="softnms", softnms_sigma=args.okssoftnms_sigma, softnms_thres=args.okssoftnms_thres)

        if args.d:
            import ipdb; ipdb.set_trace()

    print("Evaluating...")
    coco_dets = coco.loadRes(anns)
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()

    coco_eval.summarize()

    if args.d:
        import ipdb; ipdb.set_trace()
