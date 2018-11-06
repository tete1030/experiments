import torch
import pickle
import numpy as np
import re
import argparse
from collections import OrderedDict

def convert_name(key):
    out_modules = list()
    re_type = type(re.compile(r""))

    replace = {
        re.compile(r"^conv1_gn_"): "bn1",
        re.compile(r"^conv1_w"): "conv1.weight",
        re.compile(r"^pred_"): "fc",
        re.compile(r"^res(\d+)_"): lambda match: "layer" + str(int(match.group(1))-1),
        re.compile(r"^\d+_"): lambda match: match.group(0)[:-1],
        re.compile(r"^branch1_gn_"): "downsample.1",
        "branch1_w": "downsample.0.weight",
        re.compile(r"^branch2([abc])_gn_"): lambda match: "bn" + dict(a="1", b="2", c="3")[match.group(1)],
        re.compile(r"^branch2([abc])_w"): lambda match: "conv" + dict(a="1", b="2", c="3")[match.group(1)] + ".weight",
        "b": "bias",
        "s": "weight",
        "w": "weight"
    }

    ori_key = key
    while True:
        matched = False
        for pat, rep in replace.items():
            if isinstance(pat, re_type):
                m = pat.match(key)
            else:
                m = re.match(re.escape(pat), key)

            if m is not None:
                if callable(rep):
                    out_modules.append(rep(m))
                else:
                    out_modules.append(rep)
                key = key[len(m.group(0)):]
                matched = True
                break
        assert matched, key
        if len(key) == 0:
            break
    return ".".join(out_modules)

def convert(trained_model):
    with open(trained_model, "rb") as f:
        data = pickle.load(f, encoding="latin1")["blobs"]
    
    torch_state_dict = OrderedDict()
    for key in data:
        new_key = convert_name(key)
        print("{:30}{}".format(key, new_key))
        torch_state_dict[new_key] = torch.from_numpy(data[key])
    return torch_state_dict

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("ori", type=str)
    argparser.add_argument("new", type=str)
    args = argparser.parse_args()
    torch.save(convert(args.ori), args.new)


