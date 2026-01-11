import math


class WarmupCosineWithPlateau:
    def __init__(
        self,
        optimizer,
        total_epochs,
        min_lr,
        warmup_steps=0,
        warmup_ratio=0.0,
        plateau_patience=5,
        plateau_factor=0.5,
        plateau_min_delta=1e-4,
        plateau_min_lr=1e-6,
    ):
        self.optimizer = optimizer
        self.total_epochs = max(1, total_epochs)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        computed_warmup = int(self.total_epochs * warmup_ratio) if warmup_ratio > 0 else 0
        self.warmup_steps = max(warmup_steps, computed_warmup)
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.plateau_min_delta = plateau_min_delta
        self.plateau_min_lr = plateau_min_lr

        self.best = None
        self.bad_epochs = 0
        self.lr_scale = 1.0
        self.last_lrs = list(self.base_lrs)

    def _base_lr(self, epoch):
        if self.warmup_steps > 0 and epoch < self.warmup_steps:
            warmup_frac = float(epoch + 1) / float(self.warmup_steps)
            return [lr * warmup_frac for lr in self.base_lrs]

        progress = float(epoch - self.warmup_steps) / float(max(1, self.total_epochs - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lrs = []
        for lr in self.base_lrs:
            lrs.append(self.min_lr + (lr - self.min_lr) * cosine)
        return lrs

    def step(self, epoch, metric):
        if metric is not None:
            if self.best is None or metric < (self.best - self.plateau_min_delta):
                self.best = metric
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.plateau_patience:
                    self.lr_scale *= self.plateau_factor
                    self.bad_epochs = 0

        base_lrs = self._base_lr(epoch)
        scaled_lrs = []
        for lr in base_lrs:
            scaled = max(self.plateau_min_lr, lr * self.lr_scale)
            scaled_lrs.append(scaled)

        for group, lr in zip(self.optimizer.param_groups, scaled_lrs):
            group["lr"] = lr

        self.last_lrs = scaled_lrs

    def get_last_lr(self):
        return list(self.last_lrs)
