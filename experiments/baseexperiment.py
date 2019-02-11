import os
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from lib.utils.evaluation import AverageMeter, CycleAverageMeter
from utils.globals import globalvars, hparams, config
from utils.checkpoint import load_pretrained_loose, save_checkpoint, RejectLoadError

def combine_result(prev, cur):
    assert type(prev) == type(cur)
    if isinstance(prev, dict):
        assert set(prev.keys()) == set(cur.keys())
        for key in prev:
            prev[key] = combine_result(prev[key], cur[key])
        return prev
    elif isinstance(prev, list):
        return prev + cur
    elif isinstance(prev, tuple):
        assert len(prev) == len(cur)
        return (combine_result(prev[i], cur[i]) for i in range(len(prev)))
    elif isinstance(prev, torch.Tensor):
        assert prev.size()[1:] == cur.size()[1:]
        return torch.cat([prev, cur])
    elif isinstance(prev, np.ndarray):
        assert prev.shape[1:] == cur.shape[1:]
        return np.concatenate([prev, cur])
    raise TypeError("Not supported type")

class EpochContext(object):
    def __init__(self):
        self.scalar = dict()
        self.format = dict()
        self.stat_avg = dict()
        self.stored = dict()
        self.iter_data = dict()

    def add_scalar(self, sname, sval, val_format=None, stat_avg=True, cycle_avg=0):
        if sname not in self.scalar:
            self.scalar[sname] = AverageMeter() if cycle_avg == 0 else CycleAverageMeter(cycle_avg)
        if val_format is not None:
            self.format[sname] = val_format
        self.scalar[sname].update(sval)
        self.stat_avg[sname] = stat_avg

    def set_iter_data(self, sname, sval):
        self.iter_data[sname] = sval

    def clear_iter_data(self):
        for sname in list(self.iter_data.keys()):
            del self.iter_data[sname]

    def add_store(self, sname, sval):
        if sname not in self.stored:
            self.stored[sname] = sval
        else:
            self.stored[sname] = combine_result(self.stored[sname], sval)

class BaseExperiment(object):
    def __init__(self):
        self.train_collate_fn = default_collate
        self.valid_collate_fn = default_collate
        self.worker_init_fn = None
        self.print_iter_start = " | " # "\n\t"
        self.print_iter_sep = " | "
        self.init()

    def init(self):
        self.init_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_dataloader()

    def init_model(self):
        self.model = None
        self.criterion = None

    def init_optimizer(self):
        self.optimizer = None

    def init_dataset(self):
        self.train_dataset = None
        self.val_dataset = None

    def init_dataloader(self):
        # Data loading code
        if self.train_dataset is not None:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                collate_fn=self.train_collate_fn,
                batch_size=hparams.TRAIN.TRAIN_BATCH,
                num_workers=config.workers,
                shuffle=True,
                pin_memory=True,
                drop_last=config.train_drop_last,
                worker_init_fn=self.worker_init_fn)
        else:
            self.train_loader = None

        if self.val_dataset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                collate_fn=self.valid_collate_fn,
                batch_size=hparams.TRAIN.TEST_BATCH,
                num_workers=config.workers,
                shuffle=False,
                pin_memory=True,
                worker_init_fn=self.worker_init_fn)
        else:
            self.val_loader = None

    def evaluate(self, epoch_ctx:EpochContext, epoch:int, step:int):
        pass

    def epoch_start(self, epoch:int, step:int, evaluate_only:bool):
        pass

    def epoch_end(self, epoch:int, step:int, evaluate_only:bool):
        pass

    def train_once(self, epoch_ctx:EpochContext, batch:dict, progress:dict):
        loss = self.iter_process(epoch_ctx, batch, progress, True)
        self.iter_step(epoch_ctx, loss, progress)

    def test_once(self, epoch_ctx:EpochContext, batch:dict, progress:dict):
        self.iter_process(epoch_ctx, batch, progress, False)

    def iter_process(self, epoch_ctx:EpochContext, batch:dict, progress:dict, train:bool) -> torch.Tensor:
        pass

    def iter_step(self, epoch_ctx:EpochContext, loss:torch.Tensor, progress:dict):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def process_stored(self, epoch_ctx:EpochContext, epoch:int, step:int):
        pass

    def load_checkpoint(self, checkpoint_full,
                        no_strict_model_load=False,
                        no_criterion_load=False,
                        no_optimizer_load=False):

        # Load checkpoint data
        checkpoint = torch.load(checkpoint_full)
        if no_strict_model_load:
            model_state_dict = self.model.state_dict()
            try:
                model_state_dict = load_pretrained_loose(model_state_dict, checkpoint["state_dict"])
            except RejectLoadError:
                return None
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
        if self.criterion is not None and not no_criterion_load:
            self.criterion.load_state_dict(checkpoint["criterion"])
        if not no_optimizer_load:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_full, epoch):
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        if self.criterion is not None:
            checkpoint_dict["criterion"] = self.criterion.state_dict()
        save_checkpoint(checkpoint_dict, checkpoint_full=checkpoint_full, force_replace=True)

    def summarize_iter(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        pass

    def summarize_epoch(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        pass
            
    def print_iter(self, epoch_ctx:EpochContext, epoch:int, step:int):
        val_list = list()
        avg_list = list()
        for scalar_name, scalar_value in epoch_ctx.scalar.items():
            if scalar_name in epoch_ctx.format:
                sformat = epoch_ctx.format[scalar_name]
            else:
                sformat = "7.4f"
            val_list.append("{sname}: {sval:{sformat}}".format(sname=scalar_name, sval=scalar_value.val, sformat=sformat))
            if epoch_ctx.stat_avg[scalar_name]:
                avg_list.append("{sname}: {sval:{sformat}}".format(sname=scalar_name[:-1] + "_", sval=scalar_value.avg, sformat=sformat))
        print(self.print_iter_start + self.print_iter_sep.join(val_list), end="")
        print(self.print_iter_start + self.print_iter_sep.join(avg_list), end="")
        print("")
