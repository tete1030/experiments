from pose.utils.evaluation import AverageMeter
import pose.utils.config as config
from torch.utils.data.dataloader import default_collate

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
    def __init__(self, hparams):
        self.hparams = hparams
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

    def evaluate(self, preds):
        pass

    def epoch_start(self, epoch):
        pass

    def iter_process(self, epoch_ctx, batch, is_train, detail=None):
        pass

    def iter_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epoch_end(self, epoch):
        pass

    def summary_scalar_avg(self, epoch_ctx, step, phase=None):
        for scalar_name, scalar_value in epoch_ctx.scalar.items():
            if not epoch_ctx.stat_avg[scalar_name]:
                continue
            if phase is not None:
                config.tb_writer.add_scalars("{}/{}".format(config.exp_name, scalar_name), {phase: scalar_value.avg}, step)
            else:
                config.tb_writer.add_scalar("{}/{}".format(config.exp_name, scalar_name), scalar_value.avg, step)

    def summary_scalar(self, epoch_ctx, step, phase=None):
        for scalar_name, scalar_value in epoch_ctx.scalar.items():
            if phase is not None:
                config.tb_writer.add_scalars("{}/{}".format(config.exp_name, scalar_name), {phase: scalar_value.val}, step)
            else:
                config.tb_writer.add_scalar("{}/{}".format(config.exp_name, scalar_name), scalar_value.val, step)
            
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
