def adjust_learning_rate(optimizer, epoch, init_lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    lr_deg = 0
    for ep in schedule:
        if epoch >= ep:
            lr_deg += 1
    lr = init_lr * gamma ** lr_deg
    if lr_deg > 0:
        for param_group in optimizer.param_groups:
            if "init_lr" in param_group:
                param_group["lr"] = param_group["init_lr"] * gamma ** lr_deg
            else:
                param_group["lr"] = lr

    return lr
