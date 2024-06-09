import torch
from torch.nn import MSELoss
from monai.losses import BendingEnergyLoss, MultiScaleLoss, DiceLoss
from monai.optimizers.lr_scheduler import LinearLR, WarmupCosineSchedule
from monai.metrics import DiceMetric


def set_loss():
    image_loss = MSELoss()
    label_loss = DiceLoss()
    label_loss = MultiScaleLoss(label_loss, scales=[0, 1, 2, 4, 8, 16])
    regularization = BendingEnergyLoss()
    return image_loss, label_loss, regularization


def set_lr_scheduler(scheduler_name, optimizer, epochs, gamma, end_lr=1e-6, cycles=0.5):
    if scheduler_name == 'LambdaLR':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: gamma ** epoch)
    elif scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 4, gamma=gamma)
    elif scheduler_name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.8),
                                                                           int(epochs * 0.9)], gamma=gamma)
    elif scheduler_name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == 'LinearLR':
        return LinearLR(optimizer, end_lr=end_lr, num_iter=epochs)
    elif scheduler_name == 'WarmupCosineSchedule':
        return WarmupCosineSchedule(optimizer, warmup_steps=5, t_total=epochs, cycles=cycles, warmup_multiplier=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise Exception('scheduler not found')


def set_optimizer(optimizer_name, model, lr):
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-6)
    elif optimizer_name == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise Exception(f'no optimizer named {optimizer_name}')
    return optimizer


def set_metric():
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    return dice_metric
