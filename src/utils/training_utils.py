import torch.optim as optim
import torch
import os

from transformers.optimization import (get_constant_schedule_with_warmup,
                                       get_linear_schedule_with_warmup,
                                       get_cosine_schedule_with_warmup)


def get_lr_scheduler(epochs: int,
                     optimizer: optim.Optimizer,
                     dataloader_length: int,
                     scheduler_name: str = 'linear',
                     warmup_epochs: int = 1) -> optim.lr_scheduler.LRScheduler:
    accum_steps = 1
    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * (dataloader_length // accum_steps)
    else:
        warmup_steps = 0

    if scheduler_name == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                         num_warmup_steps=warmup_steps)
    elif scheduler_name == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=warmup_steps,
                                                       num_training_steps=(
                                                           dataloader_length // accum_steps * epochs),
                                                       )
    elif scheduler_name == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(dataloader_length // accum_steps) * epochs,
        )
    else:
        raise NotImplementedError
    return lr_scheduler


def get_optim(opt: str, model: torch.nn.Module, lr: float = 1e-3) -> optim.Optimizer:
    parameters = model.parameters()
    if opt == "adamW":
        optimizer = optim.AdamW(parameters, lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
    elif opt == 'RAdam':
        optimizer = optim.RAdam(parameters, lr=lr)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net: torch.nn.Module) -> None:
    num_params = 0
    num_params_train = 0

    print(str(net))

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print(f'Total number of parameters: {num_params}')
    print(f'Total number of trainable parameters: {num_params_train}')


def save_checkpoint(epoch: int, model: torch.nn.Module, score: float, save_dir: str) -> None:
    save_state = {
        'model': model.state_dict(),
        'score': score,
        'epoch': epoch
    }
    save_path = os.path.join(save_dir, f'ckpt_epoch_{epoch}.pth')

    torch.save(save_state, save_path)
