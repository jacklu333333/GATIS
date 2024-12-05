import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch, **kwargs):
        # make sure the epoch is int
        epoch = float(epoch)
        # print(f'epoch: {epoch}')
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class CycleWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch, **kwargs):
        # make sure the epoch is int
        epoch = float(epoch)
        # print(f'epoch: {epoch}')
        lr_factor = (
            0.5
            * (1 + np.cos(np.pi * epoch / self.max_num_iters))
            * (0.99 ** (epoch / self.max_num_iters))
        )
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class ReduceLROnPlateauWarmup(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        warmup_factor=0.1,
        warmup_iters=10,
    ):
        super().__init__(
            optimizer,
            mode,
            factor,
            patience,
            verbose,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
        )

        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.finished_warmup = False

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.warmup_iters:
            warmup_factor = 1 - (1 - self.warmup_factor) * (
                1 - epoch / self.warmup_iters
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * warmup_factor
        else:
            self.finished_warmup = True
            super().step(metrics, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


# use CosineWarmupScheduler until the loss improvement is high enough than threshold, then use  torch.optim.lr_scheduler.ReduceLROnPlateau
class dynamicScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup,
        max_iters,
        threshold,
        patience,
        mode="min",
        factor=0.1,
        verbose=False,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):
        self.cosine_scheduler = CosineWarmupScheduler(optimizer, warmup, max_iters)
        self.reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        self.use_cosine_scheduler = True
        self.pre_loss = None

    def get_lr(self):
        if self.use_cosine_scheduler:
            return self.cosine_scheduler.get_lr()
        else:
            return self.reduce_lr_scheduler.get_lr()

    def step(self, metrics, step, epoch=None):
        if self.use_cosine_scheduler:
            self.cosine_scheduler.step(epoch)
            if self.pre_loss is None:
                self.pre_loss = metrics
            elif (self.pre_loss - metrics) < self.threshold:
                self.use_cosine_scheduler = False
                self.pre_loss = None
            else:
                self.pre_loss = min(self.pre_loss, metrics)

        else:
            self.reduce_lr_scheduler.step(metrics, epoch)
