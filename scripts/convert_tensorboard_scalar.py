import os
import sys
import numpy as np
import argparse
import re
import copy
import shutil
import time
import traceback
from tqdm import tqdm

from collections import defaultdict
import tensorboard.version
from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator, DEFAULT_SIZE_GUIDANCE, TENSORS

from tensorboardX.event_file_writer import EventFileWriter, EventsWriter
from tensorboardX.summary import scalar, custom_scalars
from tensorboardX.proto import event_pb2
from tensorboardX.proto import summary_pb2

assert tensorboard.version.VERSION == "1.12.0"
size_guidance = copy.deepcopy(DEFAULT_SIZE_GUIDANCE)
size_guidance[TENSORS] = 0

class MyEventAccumulator(EventAccumulator):
    def Reload(self):
        with self._generator_mutex:
            for event in tqdm(self._generator.Load()):
                self._ProcessEvent(event)
        return self

class MissingFile(RuntimeError):
    pass

class MultipleFile(RuntimeError):
    pass

class ExtraFile(RuntimeError):
    pass

class EventFileNode(object):
    def __init__(self, tag_conversion, missing="raise"):
        super().__init__()
        self.tag_conversion = tag_conversion
        self.missing = missing

class TagAttribute(object):
    def __init__(self, tag_name, missing="raise"):
        super().__init__()
        self.tag_name = tag_name
        self.missing = missing

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dry-run", action="store_true")
    argparser.add_argument("--continue-on-error", action="store_true")
    argparser.add_argument("--data-source")
    argparser.add_argument("run_dir", nargs="+")
    return argparser.parse_args()

def get_dir_structure(event_dir):
    event_paths_dict = dict()
    extra_paths = list()
    for root, dirs, files in os.walk(event_dir):
        event_files = []
        extra_files = []
        for f in files:
            if re.match(r"^events\..*.(?!\.converted)$", f):
                event_files.append(f)
            else:
                extra_files.append(f)
        if len(event_files) > 0:
            event_paths_dict[os.path.relpath(os.path.normpath(root), event_dir)] = event_files
        for f in extra_files:
            extra_paths.append(os.path.relpath(os.path.normpath(os.path.join(root, f)), event_dir))

    return event_paths_dict, extra_paths

def get_convert_config(data_source):
    if data_source == "coco":
        data_config = {
            "AP": {
                "avg": EventFileNode({"offset/AP": "coco/AP_avg"}),
                "i50": EventFileNode({"offset/AP": "coco/AP_i50"}),
                "i75": EventFileNode({"offset/AP": "coco/AP_i75"}),
                "lar": EventFileNode({"offset/AP": "coco/AP_lar"}),
                "med": EventFileNode({"offset/AP": "coco/AP_med"})},
            "AR": {
                "avg": EventFileNode({"offset/AR": "coco/AR_avg"}),
                "i50": EventFileNode({"offset/AR": "coco/AR_i50"}),
                "i75": EventFileNode({"offset/AR": "coco/AR_i75"}),
                "lar": EventFileNode({"offset/AR": "coco/AR_lar"}),
                "med": EventFileNode({"offset/AR": "coco/AR_med"})}
        }
    elif data_source == "mpii":
        data_config = {
            "PCKh": {
                "avg":   EventFileNode({"offset/PCKh": "mpii/PCKh_avg"  }),
                "ank_l": EventFileNode({"offset/PCKh": "mpii/PCKh_ank_l"}),
                "ank_r": EventFileNode({"offset/PCKh": "mpii/PCKh_ank_r"}),
                "elb_l": EventFileNode({"offset/PCKh": "mpii/PCKh_elb_l"}),
                "elb_r": EventFileNode({"offset/PCKh": "mpii/PCKh_elb_r"}),
                "hip_l": EventFileNode({"offset/PCKh": "mpii/PCKh_hip_l"}),
                "hip_r": EventFileNode({"offset/PCKh": "mpii/PCKh_hip_r"}),
                "htop":  EventFileNode({"offset/PCKh": "mpii/PCKh_htop" }),
                "kne_l": EventFileNode({"offset/PCKh": "mpii/PCKh_kne_l"}),
                "kne_r": EventFileNode({"offset/PCKh": "mpii/PCKh_kne_r"}),
                "pelv":  EventFileNode({"offset/PCKh": "mpii/PCKh_pelv" }),
                "sho_l": EventFileNode({"offset/PCKh": "mpii/PCKh_sho_l"}),
                "sho_r": EventFileNode({"offset/PCKh": "mpii/PCKh_sho_r"}),
                "thor":  EventFileNode({"offset/PCKh": "mpii/PCKh_thor" }),
                "upnk":  EventFileNode({"offset/PCKh": "mpii/PCKh_upnk" }),
                "wri_l": EventFileNode({"offset/PCKh": "mpii/PCKh_wri_l"}),
                "wri_r": EventFileNode({"offset/PCKh": "mpii/PCKh_wri_r"})}
        }

    config = {
        ".": EventFileNode({
            "offset/move_dis_right_ratio": TagAttribute("move_dis/right_ratio", "ignore"),
            "offset/sigma_change_right_ratio": TagAttribute("sigma_change/right_ratio", "ignore")}),
        "offset": {
            "loss": {
                "train": EventFileNode({"offset/loss": "loss/all_train"}),
                "valid": EventFileNode({"offset/loss": "loss/all_valid"})},
            "feature_loss": {
                "train": EventFileNode({"offset/feature_loss": "loss/feature"}, missing="ignore")},
            "move_dis": {
                "mod": EventFileNode({"offset/move_dis": "move_dis/avg"}),
                "mod_cur": EventFileNode({"offset/move_dis": "move_dis/cur"})},
            "sigma_change": {
                "avg": EventFileNode({"offset/sigma_change": "sigma_change/avg"}, missing="ignore"),
                "cur": EventFileNode({"offset/sigma_change": "sigma_change/cur"}, missing="ignore")},
            **data_config}}

    cus_scalars = {
        "loss": {
            "all": ["Multiline", [r"loss/all_.*"]],
            "train": ["Multiline", [r"loss/all_train", r"loss/(?!all_).*"]]
        },
        "offset": {
            "move_dis": ["Multiline", [r"move_dis/(?!right_ratio)"]],
            "sigma_change": ["Multiline", [r"sigma_change/(?!right_ratio)"]]
        }
    }
    if data_source == "coco":
        cus_scalars.update({
            "coco": {
                "AP": ["Multiline", [r"coco/AP_.*"]],
                "AR": ["Multiline", [r"coco/AR_.*"]]
            }
        })
    elif data_source == "mpii":
        cus_scalars.update({
            "mpii": {
                "PCKh": ["Multiline", [r"mpii/.*"]]
            }
        })
    else:
        raise ValueError("Unknown data_source")

    return config, cus_scalars

def flatten_config(conf_path, conf, flat_conf=None):
    if flat_conf is None:
        flat_conf = dict()
    for path in conf:
        full_path = os.path.normpath(os.path.join(conf_path, path))
        if isinstance(conf[path], EventFileNode):
            flat_conf[full_path] = conf[path]
        else:
            flatten_config(full_path, conf[path], flat_conf=flat_conf)

    return flat_conf

def load_dir(event_dir, conf, load=True):
    event_path_file_dict, extra_files = get_dir_structure(event_dir)
    flat_conf = flatten_config(".", conf)
    events_found = set(event_path_file_dict.keys())
    events_conf = set(flat_conf.keys())

    extra_events = events_found - events_conf
    if extra_events:
        raise ExtraFile("Extra events {}".format(extra_events))
    
    missing_events = events_conf - events_found
    raise_events = list()
    for missing_ev in missing_events:
        if flat_conf[missing_ev].missing == "raise":
            raise_events.append(missing_ev)
        elif flat_conf[missing_ev].missing == "ask":
            if input("{} missing in {}, continue (y/n)?".format(missing_ev, event_dir)) != "y":
                raise_events.append(missing_ev)
    if raise_events:
        raise MissingFile("Missing events {}".format(raise_events))

    events = list()
    for iev, event_path in enumerate(event_path_file_dict.keys()):
        event_filepath = os.path.join(event_dir, event_path)
        print("==> ({}/{}) Loading events {}".format(iev+1, len(event_path_file_dict), event_path))
        event_accumulator = MyEventAccumulator(event_filepath, size_guidance=size_guidance)
        if load:
            event_accumulator.Reload()
        events.append((flat_conf[event_path], event_accumulator))

    return events

def main(args):
    convert_conf, cusscalar_layout = get_convert_config(args.data_source)

    for cur_run_dir in args.run_dir:
        print("====> Loading " + cur_run_dir)
        inside_filedirs = os.listdir(cur_run_dir)
        if len(inside_filedirs) == 1 and re.match(r"^.*\.converted$", inside_filedirs[0]):
            print("Already converted, skipping " + cur_run_dir)
            continue

        try:
            events = load_dir(cur_run_dir, convert_conf, load=not args.dry_run)
        except RuntimeError as rerr:
            if args.dry_run or args.continue_on_error:
                traceback.print_exception(type(rerr), rerr, None, file=sys.stdout)
                continue
            else:
                raise rerr

        if args.dry_run:
            continue
        
        writer = EventFileWriter(cur_run_dir, filename_suffix=".converted")
        try:
            cusscalar_event = event_pb2.Event(summary=custom_scalars(cusscalar_layout))
            cusscalar_event.wall_time = time.time()
            writer.add_event(cusscalar_event)

            for event_conf, event_ac in events:
                tags = event_ac.Tags()
                for cat in tags:
                    if cat == TENSORS:
                        continue
                    if isinstance(tags[cat], list):
                        assert len(tags[cat]) == 0, "Extra tag in '{}': {}".format(cat, tags[cat])
                    elif isinstance(tags[cat], bool):
                        assert tags[cat] is False, "Category '{}' should be False".format(cat)
                    else:
                        raise RuntimeError("Unknown type for category: {}({})".format(cat, type(tags[cat])))

                for tag_ori, tag_new_attr in event_conf.tag_conversion.items():
                    tag_new_name = tag_new_attr.tag_name if isinstance(tag_new_attr, TagAttribute) else tag_new_attr
                    print("Converting from {} to {}".format(tag_ori, tag_new_name))
                    if tag_ori not in tags[TENSORS]:
                        msg = "No tag {} found".format(tag_ori)
                        if isinstance(tag_new_attr, str) or tag_new_attr.missing == "raise":
                            raise RuntimeError(msg)
                        else:
                            print(msg)
                            continue
                    for event in tqdm(event_ac.Tensors(tag_ori)):
                        event_new = event_pb2.Event(summary=scalar(tag_new_name, event.tensor_proto.float_val[0]))
                        event_new.wall_time = event.wall_time
                        event_new.step = event.step
                        writer.add_event(event_new)
        except (Exception, KeyboardInterrupt) as e:
            try:
                writer.close()
            finally:
                writer_filename = writer._ev_writer._file_name
                print("Exception! Deleting " + writer_filename)
                os.remove(writer_filename)
                raise e
        else:
            writer.close()

        delete_filedirs = set(os.listdir(cur_run_dir)) - set([os.path.relpath(writer._ev_writer._file_name, cur_run_dir)])
        temp_path = os.path.join("/tmp", "mypose_runs", cur_run_dir)
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)
        for delete_filedir in delete_filedirs:
            full_path = os.path.join(cur_run_dir, delete_filedir)
            print("Moving {} to {}".format(full_path, temp_path))
            shutil.move(full_path, temp_path)

if __name__ == "__main__":
    main(get_args())