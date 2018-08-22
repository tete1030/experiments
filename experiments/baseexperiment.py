import os
from pose.utils.evaluation import AverageMeter
from utils.globals import globalvars
import torch
from torch.utils.data.dataloader import default_collate
from utils.checkpoint import load_pretrained_loose, save_checkpoint, RejectLoadError

class EpochContext(object):
    def __init__(self):
        self.scalar = dict()
        self.format = dict()
        self.stat_avg = dict()

    def add_scalar(self, sname, sval, n, val_format=None, stat_avg=True):
        if sname not in self.scalar:
            self.scalar[sname] = AverageMeter()
        if val_format is not None:
            self.format[sname] = val_format
        self.scalar[sname].update(sval, n)
        self.stat_avg[sname] = stat_avg

class BaseExperiment(object):
    def __init__(self):
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_collate_fn = default_collate
        self.valid_collate_fn = default_collate
        self.train_drop_last = False
        self.worker_init_fn = None
        self.print_iter_start = "\n\t"
        self.print_iter_sep = " | "
        self.cur_lr = None
        self.init()

    def init(self):
        pass

    def evaluate(self, preds, step):
        pass

    def epoch_start(self, epoch):
        pass

    def iter_process(self, epoch_ctx: EpochContext, batch: dict, is_train: bool, progress: dict) -> dict:
        pass

    def iter_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epoch_end(self, epoch):
        pass

    def load_checkpoint(self, checkpoint_folder, checkpoint_file,
                        no_strict_model_load=False,
                        no_criterion_load=False,
                        no_optimizer_load=False):

        # Load checkpoint data
        checkpoint_full = os.path.join(checkpoint_folder, checkpoint_file)
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
        if not no_criterion_load:
            self.criterion.load_state_dict(checkpoint["criterion"])
        if not no_optimizer_load:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_folder, checkpoint_file, epoch):
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict()
        }
        save_checkpoint(checkpoint_dict, checkpoint_folder=checkpoint_folder, checkpoint_file=checkpoint_file, force_replace=True)

    def summary_scalar_avg(self, epoch_ctx, step, phase=None):
        for scalar_name, scalar_value in epoch_ctx.scalar.items():
            if not epoch_ctx.stat_avg[scalar_name]:
                continue
            if phase is not None:
                globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, scalar_name), {phase: scalar_value.avg}, step)
            else:
                globalvars.tb_writer.add_scalar("{}/{}".format(globalvars.exp_name, scalar_name), scalar_value.avg, step)

    def summary_scalar(self, epoch_ctx, step, phase=None):
        for scalar_name, scalar_value in epoch_ctx.scalar.items():
            if phase is not None:
                globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, scalar_name), {phase: scalar_value.val}, step)
            else:
                globalvars.tb_writer.add_scalar("{}/{}".format(globalvars.exp_name, scalar_name), scalar_value.val, step)
            
    def print_iter(self, epoch_ctx):
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
