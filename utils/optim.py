from pytorch_lightning import Trainer
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import transformers

def epoch_warmup_optimizers(model: nn.Module, config):
    lr = config.lr
    weight_decay = config.weight_decay
    warm_up_epochs = config.warmup_epochs

    def lr_foo(epoch):
        if epoch < warm_up_epochs:
            lr_scale = 0.1 ** (warm_up_epochs - epoch)
        else:
            lr_scale = 0.95 ** epoch
        return lr_scale

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_foo)
    return [optimizer], [scheduler]


def step_warmup_optimizers(model: nn.Module, config):
    lr = config.lr
    weight_decay = config.weight_decay
    warm_up_steps = config.warnup_steps

    def lr_foo(step):
        if step < warm_up_steps:
            lr_scale = step / warm_up_steps
        else:
            lr_scale = 1
        return lr_scale

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda=lr_foo),
                 'interval': 'step',
                 'frequency': 1}
    return [optimizer], [scheduler]


def get_linear_schedule_with_warmup(model, config, trainer: Trainer):
    lr = config.lr
    weight_decay = config.weight_decay
    warmup_steps = config.warnup_steps
    num_training_steps = trainer.estimated_stepping_batches
    # print('{num_training_steps=}')

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda),
                 'interval': 'step',
                 'frequency': 1}
    return [optimizer], [scheduler]
