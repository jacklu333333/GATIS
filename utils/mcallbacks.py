import random
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

"""
progress bar with time
"""


class TimeProgressBar(TQDMProgressBar):
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        # super().on_train_batch_end(trainer, pl_module, batch, batch_idx, dataloader_idx)
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            info = self.get_metrics(trainer, pl_module)
            # only keep f1 loss acc
            info = {
                k: v for k, v in info.items() if "f1" in k or "loss" in k or "acc" in k
            }

            learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
            info["lr"] = learning_rate

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info["time"] = current_time

            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(info)
        # self.train_progress_bar.set_postfix(time=current_time, refresh=False)
        # make time and all the metrics available in progress bar
        # self.main_progress_bar.


"""
Early stopping callback
# instead of monitor the patient by epoch, monitor by the train step
"""


class EarlyStopping(EarlyStopping):
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float = 0.0,
        divergence_threshold: float = 0.0,
        check_on_train_epoch_end: bool = False,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_last: bool | None = None,
        save_top_k: int | None = None,
        save_weights_only: bool = False,
        auto_insert_metric_name: bool = True,
        verbose_metrics: bool = False,
        path: str | None = None,
        _open_mode: str = "w",
        _use_ckpt: bool = False,
        _stopping: bool = False,
    ):
        super().__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            strict,
            check_finite,
            stopping_threshold,
            divergence_threshold,
            check_on_train_epoch_end,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_last,
            save_top_k,
            save_weights_only,
            auto_insert_metric_name,
            verbose_metrics,
            path,
            _open_mode,
            _use_ckpt,
            _stopping,
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        if self._should_skip_check(trainer):
            return
        if self._should_stop:
            trainer.should_stop = True
            return
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Checks whether the early stopping condition is met
        """
        # monitor key and current value
        monitor_val = self._monitor_candidates(trainer)
        # monitor_val = monitor_val["val_loss"]
        #


"""
Don't use it yet
this will invoke the val every time perform the  save
Need to be test further
"""


class valCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        dirpath: _PATH | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | None = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        dm: pl.LightningDataModule = None,
    ):
        self._dm = dm
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
        )

    def _save_topk_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        trainer.validate(self._dm)
        super()._save_checkpoint(trainer, monitor_candidates)
