import copy
import datetime
import glob
import io
import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Union

import ahrs
import colored as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchvision
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from timm.models.layers import DropPath
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchaudio.models import Conformer

# import auc F1 score  from torch
from torchmetrics.classification import (
    AUROC,
    ROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MulticlassConfusionMatrix,
    Precision,
    Recall,
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix,
)
from tqdm import tqdm

from .augmentation import Augmentation
from .ConvNeXt import ConvNeXtV2, convnextv2_base
from .correction import rotateToWorldFrame
from .geoPreprocessor import GravityRemoval, magRescale
from .mLoss import CISSL, DISSL, ContrastiveLoss, StudentNet, reweightMSELoss
from .mnetwork import (
    channelShffuleNet,
    depthwiseSeparableConvolution2d,
    mResnet1d,
    mResnet2d,
    reshapeNet,
    swapaxesNet,
)
from .mplot import plot_to_image, plotConfusionMatrix
from .mscheduler import CosineWarmupScheduler, ReduceLROnPlateauWarmup, dynamicScheduler
from .stepDetection import stepDetection

WINDOW_SIZE = 100
BATCH_SIZE = 32 * 32


def positionEncoding(d_model, length):
    """
    Positional encoding for 1D signals
    """
    encoding = np.zeros((length, d_model))
    for i in range(length):
        for j in range(d_model):
            if j % 2 == 0:
                encoding[i, j] = np.sin(i / 10000 ** (j / d_model))
            else:
                encoding[i, j] = np.cos(i / 10000 ** ((j - 1) / d_model))
    return torch.tensor(encoding).float().unsqueeze(0)


class sensorInterpreter(nn.Module):
    def __init__(self):
        super(sensorInterpreter, self).__init__()
        self.pe_time = positionEncoding((WINDOW_SIZE // 2 + 1) * 4, WINDOW_SIZE)
        self.pe_freq = positionEncoding((WINDOW_SIZE // 2 + 1) * 4, 2)
        self.encoder = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=(WINDOW_SIZE // 2 + 1) * 4,
                nhead=12,
                dim_feedforward=512,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            swapaxesNet(1, 2),
            # nn.BatchNorm1d((WINDOW_SIZE // 2 + 1) * 4),
            nn.InstanceNorm1d((WINDOW_SIZE // 2 + 1) * 4),
            swapaxesNet(1, 2),
            # nn.BatchNorm1d(2),
        )
        self.decoder = nn.TransformerDecoderLayer(
            d_model=(WINDOW_SIZE // 2 + 1) * 4,
            nhead=12,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.Sequential(
            reshapeNet((-1, WINDOW_SIZE, 4, WINDOW_SIZE // 2 + 1)),
            swapaxesNet(1, 2),
            nn.InstanceNorm2d(4),
            # nn.BatchNorm2d(4),
            swapaxesNet(1, 2),
            reshapeNet((-1, WINDOW_SIZE, 4 * (WINDOW_SIZE // 2 + 1))),
        )

    def forward(self, time_domain, freq_domain):
        time_domain = time_domain + self.pe_time.to(time_domain.device)
        freq_domain = freq_domain + self.pe_freq.to(freq_domain.device)
        freq = self.encoder(freq_domain)
        sensor = self.decoder(time_domain, freq)
        sensor = self.norm(sensor)

        return sensor


class interpretationNet(nn.Module):
    def __init__(self, activation=F.gelu):
        super(interpretationNet, self).__init__()
        self.encoder = nn.TransformerDecoderLayer(
            d_model=(WINDOW_SIZE // 2 + 1) * 4,
            nhead=12,
            dim_feedforward=512,
            dropout=0.1,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.Sequential(
            reshapeNet((-1, WINDOW_SIZE, 4, WINDOW_SIZE // 2 + 1)),
            swapaxesNet(1, 2),
            nn.InstanceNorm2d(4),
            # nn.BatchNorm2d(4),
            swapaxesNet(1, 2),
            reshapeNet((-1, WINDOW_SIZE, 4 * (WINDOW_SIZE // 2 + 1))),
        )

    def forward(self, sensorA, sensorB):
        meanings = self.encoder(sensorA, sensorB)
        meanings = self.norm(meanings)

        return meanings


class residualMlpBlock(nn.Module):
    def __init__(self, hidden):
        super(residualMlpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            # DropPath(0.1),
            nn.Linear(hidden // 2, hidden),
            nn.BatchNorm1d(hidden),
            # nn.GELU(),
            nn.PReLU(),
            #
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class residualMlp(nn.Module):
    def __init__(
        self,
        in_size=1024,
        hidden=128,
        num_layers=3,
        out_size=2,
        activation: str = "softmax",
    ):
        super(residualMlp, self).__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(in_size),
            # DropPath(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, hidden),
            nn.LayerNorm(hidden),
        )

        self.blocks = nn.Sequential(
            *[residualMlpBlock(hidden) for _ in range(num_layers)]
        )

        self.mlp = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, out_size),
        )
        if activation == "softmax":
            self.activate = nn.Softmax(dim=-1)
        elif activation == "sigmoid":
            self.activate = nn.Sigmoid()
        else:
            raise ValueError("activation must be either softmax or sigmoid")

    def forward(self, x):
        x1 = self.projection(x)
        x2 = self.blocks(x1)
        x3 = self.mlp(x2)
        out = self.activate(x3)

        return out


class motionModel(nn.Module):
    def __init__(self, config):
        super(motionModel, self).__init__()
        self.config = config
        self.acc = sensorInterpreter()
        self.gyr = sensorInterpreter()
        self.motionInterpreter = interpretationNet()

    def forward(self, batch_size, accT, accF, gyrT, gyrF):
        accSensor = self.acc(accT, accF)
        gyrSensor = self.gyr(gyrT, gyrF)
        motion = self.motionInterpreter(accSensor, gyrSensor)

        return motion


class magneticModel(nn.Module):
    def __init__(self, config):
        super(magneticModel, self).__init__()
        self.config = config
        self.mag = sensorInterpreter()
        self.magneticInterpreter = interpretationNet(activation=F.sigmoid)

    def forward(self, batch_size, magT, magF, motion):
        magSensor = self.mag(magT, magF)
        mag = self.magneticInterpreter(magSensor, motion)

        return mag


class baseModule(pl.LightningModule):
    def __init__(self, config):
        super(baseModule, self).__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.lr = self.config["lr"]
        self.augmentation = Augmentation(
            upstreamtask=config["upstreamtask"],
            mode="regression" if config["upstreamtask"] else "classification",
        )
        if not config["upstreamtask"]:
            self.train_metrics = self.gen_metrics("train")
            self.val_metrics = self.gen_metrics("val")
            self.test_metrics = self.gen_metrics("test")
        else:
            self.train_metrics = dict()
            self.val_metrics = dict()
            self.test_metrics = dict()

    def gen_metrics(self, suffix):
        metrics = {
            f"accuracy/{suffix}": Accuracy(
                task="multiclass",
                num_classes=self.config["num_classes"],
                average="macro",
            ),
            f"auc/{suffix}": AUROC(
                task="multiclass",
                num_classes=self.config["num_classes"],
                average="macro",
            ),
            f"f1/{suffix}": F1Score(
                task="multiclass",
                num_classes=self.config["num_classes"],
                average="macro",
            ),
            f"precision/{suffix}": Precision(
                task="multiclass",
                num_classes=self.config["num_classes"],
                average="macro",
            ),
            f"recall/{suffix}": Recall(
                task="multiclass",
                num_classes=self.config["num_classes"],
                average="macro",
            ),
            f"confusion_matrix/{suffix}": MulticlassConfusionMatrix(
                num_classes=self.config["num_classes"],
                normalize="true",
            ),
        }
        for k, v in metrics.items():
            self.add_module(f"{k}", v)
        return metrics

    def compute_metrics(self, y_hat, y, self_metrics):
        metrics = dict()

        for key, metric in self_metrics.items():
            metric.update(y_hat, y)
            if "confusion_matrix" in key:
                continue
            metrics[key] = metric.compute()

        return metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.train_loss(y_hat, y)

        params = self._get_regularization_params()
        l1 = torch.norm(params, 1) * self.config["l1"]
        l2 = torch.norm(params, 2) * self.config["l2"]
        metrics = self.compute_metrics(y_hat, y, self.train_metrics)

        self.log_dict(
            {
                "loss/train": loss,
                "l1/train": l1,
                "l2/train": l2,
                "total_loss/train": loss + l1 + l2,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss + l1 + l2

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.val_loss(y_hat, y)

        l1 = torch.norm(self._get_regularization_params(), 1) * self.config["l1"]
        l2 = torch.norm(self._get_regularization_params(), 2) * self.config["l2"]
        metrics = self.compute_metrics(y_hat, y, self.val_metrics)
        self.log_dict(
            {
                "loss/val": loss,
                "l1/val": l1,
                "l2/val": l2,
                "total_loss/val": loss + l1 + l2,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.test_loss(y_hat, y)

        metrics = self.compute_metrics(y_hat, y, self.test_metrics)
        self.log_dict(
            {
                "loss/test": loss,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        for metric in self.train_metrics.values():
            metric.reset()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        for metric in self.val_metrics.values():
            metric.reset()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        for metric in self.test_metrics.values():
            metric.reset()

    def on_train_epoch_end(self):
        if "confusion_matrix/train" in list(self.train_metrics.keys()):
            confusion_matrix = self.train_metrics["confusion_matrix/train"].compute()
            self.log_cmimage(confusion_matrix, "train")
        super().on_train_epoch_end()
        for metric in self.train_metrics.values():
            metric.reset()

    def on_validation_epoch_end(self):
        # check confusion matrix key exist in the metrics or not
        if "confusion_matrix/val" in list(self.val_metrics.keys()):
            confusion_matrix = self.val_metrics["confusion_matrix/val"].compute()
            self.log_cmimage(confusion_matrix, "val")
        super().on_validation_epoch_end()
        for metric in self.val_metrics.values():
            metric.reset()

    def on_test_epoch_end(self):
        if "confusion_matrix/test" in list(self.test_metrics.keys()):
            confusion_matrix = self.test_metrics["confusion_matrix/test"].compute()
            self.log_cmimage(confusion_matrix, "test")
        super().on_test_epoch_end()
        for metric in self.test_metrics.values():
            metric.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = dynamicScheduler(optimizer, self.config)
        # scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=10, max_epochs=100)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _get_regularization_params(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param.view(-1))
        params = torch.cat(params)
        return params

    def log_cmimage(self, cm, mode=None):
        fig = plotConfusionMatrix(cm, labels=self.config["labels"])
        figure = plot_to_image(fig)
        # store the confusion matrix to tensorboard
        self.logger.experiment.add_image(
            f"confusion_matrix/{mode}",
            figure,
            self.current_epoch,
            dataformats="CHW",
        )


class HARModule(baseModule):
    def __init__(self, config):
        super(HARModule, self).__init__(config=config)
        self.motionModel = motionModel(self.config)
        self.reshaper = nn.Sequential(
            swapaxesNet(1, 2),
            reshapeNet((-1, 4, WINDOW_SIZE, WINDOW_SIZE // 2 + 1)),
        )
        self.downSampler = ConvNeXtV2(
            in_chans=4,
            num_classes=1024,
            drop_path_rate=0.1,
            depths=[2, 2, 6, 2],
            dims=[40, 80, 160, 320],
            head_init_scale=1.0,
        )

        if not self.config["upstreamtask"]:
            self.train_loss = nn.CrossEntropyLoss(weight=config["train_label_weights"])
            self.val_loss = nn.CrossEntropyLoss(weight=config["val_label_weights"])
            self.test_loss = nn.CrossEntropyLoss(weight=config["test_label_weights"])
            self.motionClassifier = residualMlp(
                in_size=1024,
                hidden=self.config["hidden"],
                out_size=self.config["num_classes"],
                num_layers=self.config["num_layers"],
            )

    def forward(self, x):
        batch_size = x["time"].shape[0]

        accT = x["time"][:, :4, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)
        accF = x["freq"][:, :4, :, :].permute(0, 3, 2, 1).reshape(batch_size, 2, -1)

        gyrT = x["time"][:, 4:8, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)
        gyrF = x["freq"][:, 4:8, :, :].permute(0, 3, 2, 1).reshape(batch_size, 2, -1)

        motion = self.motionModel(batch_size, accT, accF, gyrT, gyrF)
        reshaper = self.reshaper(motion)
        down = self.downSampler(reshaper)
        if not self.config["upstreamtask"]:
            out = self.motionClassifier(down)
        else:
            out = down

        return out

    def load_from_upstream_weights(self, path):
        upstream = torch.load(path)["state_dict"]
        # remove any keys with of "loss"
        upstream = {k: v for k, v in upstream.items() if "loss" not in k}
        self.load_state_dict(upstream, strict=False)

        # # freeze
        # for param in self.parameters():
        #     param.requires_grad = False
        # for param in self.downSampler.parameters():
        #     param.requires_grad = True
        # for param in self.motionClassifier.parameters():
        #     param.requires_grad = True


class IODModule(baseModule):
    def __init__(self, config):
        super(IODModule, self).__init__(config)
        self.motionModel = motionModel(self.config)
        self.magModel = magneticModel(self.config)
        self.reshaper = nn.Sequential(
            swapaxesNet(1, 2),
            reshapeNet((-1, 4, WINDOW_SIZE, WINDOW_SIZE // 2 + 1)),
        )
        self.downSampler = ConvNeXtV2(
            in_chans=4,
            num_classes=1024,
            drop_path_rate=0.1,
            depths=[2, 2, 6, 2],
            dims=[40, 80, 160, 320],
            head_init_scale=1.0,
        )
        if not self.config["upstreamtask"]:
            self.train_loss = nn.CrossEntropyLoss(weight=config["train_label_weights"])
            self.val_loss = nn.CrossEntropyLoss(weight=config["val_label_weights"])
            self.test_loss = nn.CrossEntropyLoss()

            self.magClassifier = residualMlp(
                in_size=1024,
                hidden=self.config["hidden"],
                out_size=self.config["num_classes"],
                num_layers=self.config["num_layers"],
            )

    def forward(self, x):
        batch_size = x["time"].shape[0]

        accT = x["time"][:, :4, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)
        accF = x["freq"][:, :4, :, :].permute(0, 3, 2, 1).reshape(batch_size, 2, -1)

        gyrT = x["time"][:, 4:8, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)
        gyrF = x["freq"][:, 4:8, :, :].permute(0, 3, 2, 1).reshape(batch_size, 2, -1)

        magT = x["time"][:, 8:, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)
        magF = x["freq"][:, 8:, :, :].permute(0, 3, 2, 1).reshape(batch_size, 2, -1)

        motion = self.motionModel(batch_size, accT, accF, gyrT, gyrF)
        mag = self.magModel(batch_size, magT, magF, motion)
        reshaper = self.reshaper(mag)
        down = self.downSampler(reshaper)
        if not self.config["upstreamtask"]:
            out = self.magClassifier(down)
        else:
            out = down

        return out

    def load_from_upstream_weights(self, path):
        upstream = torch.load(path)["state_dict"]
        # remove any keys with of "loss"
        upstream = {k: v for k, v in upstream.items() if "loss" not in k}
        self.load_state_dict(upstream, strict=False)
        self.freeze()

    def freeze(self):
        # freeze
        for param in self.parameters():
            param.requires_grad = False
        for param in self.magModel.parameters():
            param.requires_grad = True
        for param in self.downSampler.parameters():
            param.requires_grad = True
        for param in self.magClassifier.parameters():
            param.requires_grad = True
        for param in self.magClassifier.parameters():
            param.requires_grad = True

    def _get_regularization_params(self):
        params = []
        for param in [
            self.motionModel.acc.norm.parameters(),
            self.motionModel.gyr.norm.parameters(),
            self.magModel.mag.norm.parameters(),
            self.motionModel.motionInterpreter.norm.parameters(),
            self.magModel.magneticInterpreter.norm.parameters(),
            self.downSampler.parameters(),
        ]:
            # if param.requires_grad:
            params.extend([x.view(-1) for x in param])
        params = torch.cat(params)
        return params

    # def gen_metrics(self, suffix):
    #     metrics = {
    #         f"accuracy/{suffix}": BinaryAccuracy(
    #         ),
    #         f"auc/{suffix}": BinaryAUROC(
    #         ),
    #         f"f1/{suffix}": BinaryF1Score(
    #         ),
    #         f"precision/{suffix}": BinaryPrecision(
    #         ),
    #         f"recall/{suffix}": BinaryRecall(
    #         ),
    #         f"confusion_matrix/{suffix}": BinaryConfusionMatrix(
    #             normalize="true",
    #         ),
    #     }
    #     for k, v in metrics.items():
    #         self.add_module(f"{k}", v)
    #     return metrics

    # def compute_metrics(self, y_hat, y):
    #     metrics = dict()
    #     y_hat = torch.argmax(y_hat, dim=1)
    #     y = y.float()

    #     for key, metric in self.train_metrics.items():
    #         metric.update(y_hat, y)
    #         if "confusion_matrix" in key:
    #             continue
    #         metrics[key] = metric.compute()

    #     return metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = dynamicScheduler(optimizer, self.config)
        scheduler = CosineWarmupScheduler(optimizer, warmup=2, max_iters=17 * 3)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.5,
        #     patience=10,
        #     verbose=True,
        #     threshold=0.0001,
        #     threshold_mode="rel",
        #     cooldown=0,
        #     min_lr=0,
        #     eps=1e-08,
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class HARUpstreamModule(HARModule):
    def __init__(self, config):
        super(HARUpstreamModule, self).__init__(config)
        self.loss = CISSL(
            hidden_dim=self.config["hidden"],
            proj_dim=self.config["proj_dim"],
            z_dim=self.config["z_dim"],
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_a, x_b = self.augmentation(x, pair=True)
        predict_a = self.forward(x_a)
        predict_b = self.forward(x_b)

        loss = self.loss(predict_a, predict_b)

        params = self._get_regularization_params()
        l1 = torch.norm(params, 1) * self.config["l1"]
        l2 = torch.norm(params, 2) * self.config["l2"]

        self.log_dict(
            {
                "loss/train": loss,
                "l1/train": l1,
                "l2/train": l2,
                "total_loss/train": loss + l1 + l2,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss + l1 + l2

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_a, x_b = self.augmentation(x, pair=True)
        predict_a = self.forward(x_a)
        predict_b = self.forward(x_b)

        loss = self.loss(predict_a, predict_b)

        l1 = torch.norm(self._get_regularization_params(), 1) * self.config["l1"]
        l2 = torch.norm(self._get_regularization_params(), 2) * self.config["l2"]

        self.log_dict(
            {
                "loss/val": loss,
                "l1/val": l1,
                "l2/val": l2,
                "total_loss/val": loss + l1 + l2,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_a, x_b = self.augmentation(x, pair=True)
        predict_a = self.forward(x_a)
        predict_b = self.forward(x_b)

        loss = self.loss(predict_a, predict_b)

        self.log_dict(
            {
                "loss/test": loss,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
        # scheduler = dynamicScheduler(optimizer, self.config)
        scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val",
                "interval": "step",
                "frequency": 1,
            },
        }


class IODUpstreamModule(IODModule):
    def __init__(self, config):
        super(IODUpstreamModule, self).__init__(config)
        self.train_loss = self.val_loss = self.test_loss = CISSL(
            hidden_dim=self.config["hidden"],
            proj_dim=self.config["proj_dim"],
            z_dim=self.config["z_dim"],
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_a, x_b = self.augmentation(x, pair=True)
        predict_a = self.forward(x_a)
        predict_b = self.forward(x_b)

        loss = self.train_loss(predict_a, predict_b)

        params = self._get_regularization_params()
        l1 = torch.norm(params, 1) * self.config["l1"]
        l2 = torch.norm(params, 2) * self.config["l2"]

        self.log_dict(
            {
                "loss/train": loss,
                "l1/train": l1,
                "l2/train": l2,
                "total_loss/train": loss + l1 + l2,
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss + l1 + l2

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_a, x_b = self.augmentation(x, pair=True)
        predict_a = self.forward(x_a)
        predict_b = self.forward(x_b)

        loss = self.val_loss(predict_a, predict_b)

        l1 = torch.norm(self._get_regularization_params(), 1) * self.config["l1"]
        l2 = torch.norm(self._get_regularization_params(), 2) * self.config["l2"]

        self.log_dict(
            {
                "loss/val": loss,
                "l1/val": l1,
                "l2/val": l2,
                "total_loss/val": loss + l1 + l2,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_a, x_b = self.augmentation(x, pair=True)
        predict_a = self.forward(x_a)
        predict_b = self.forward(x_b)

        loss = self.test_loss(predict_a, predict_b)

        self.log_dict(
            {
                "loss/test": loss,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])
        # scheduler = dynamicScheduler(optimizer, self.config)
        scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val",
                "interval": "step",
                "frequency": 1,
            },
        }

    def load_from_har_weights(self, path):
        har = torch.load(path)["state_dict"]
        # remove any keys with of "loss"
        har = {k: v for k, v in har.items() if "loss" not in k}
        # remove any keys with of "downSampler"
        har = {k: v for k, v in har.items() if "downSampler" not in k}

        self.load_state_dict(har, strict=False)

        # freeze
        for param in self.parameters():
            param.requires_grad = False
        for param in self.downSampler.parameters():
            param.requires_grad = True

        for param in self.train_loss.parameters():
            param.requires_grad = True
        for param in self.val_loss.parameters():
            param.requires_grad = True
        for param in self.test_loss.parameters():
            param.requires_grad = True


# class magneticModule(pl):


# class BaseModel(nn.Module):
#     def __init__(self):
#         super(BaseModel, self).__init__()
#         # sensor interpreter
#         self.acc = sensorInterpreter()
#         self.gyr = sensorInterpreter()
#         self.mag = sensorInterpreter()

#         # motion interpreter
#         self.motionInterpreter = interpretationNet()
#         self.motionDownsmapler = ConvNeXtV2(
#             in_chans=4,
#             num_classes=1024,
#             drop_path_rate=0.1,
#             depths=[2, 2, 6, 2],
#             dims=[40, 80, 160, 320],
#             head_init_scale=1.0,
#         )
#         self.motionClassifier = residualMlp(
#             in_size=1024,
#             hidden=128,
#             out_size=self.config.num_classes,
#             num_layers=self.config.num_layers,
#         )

#         # magnetic interpreter
#         self.magneticInterpreter = interpretationNet()
#         self.magDownsmapler = ConvNeXtV2(
#             in_chans=4,
#             num_classes=1024,
#             drop_path_rate=0.1,
#             depths=[2, 2, 6, 2],
#             dims=[40, 80, 160, 320],
#             head_init_scale=1.0,
#         )
#         self.magClassifier = residualMlp(
#             in_size=1024,
#             hidden=128,
#             out_size=self.config.num_classes,
#             num_layers=self.config.num_layers,
#         )

#     def forward(self, x):
#         xTime = x["time"]
#         xFreq = x["freq"]
#         batch = xTime.shape[0]

#         # shape : (batch, WINDOW_SIZE, (WINDOW_SIZE // 2 + 1)*4)
#         accT = xTime[:, :4, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)
#         gyrT = xTime[:, 4:8, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)
#         magT = xTime[:, 8:, :].swapaxes(1, 2).repeat(1, 1, WINDOW_SIZE // 2 + 1)

#         # shape : (batch, 2, (WINDOW_SIZE // 2 + 1)*4)
#         accF = xFreq[:, :4, :, :].permute(0, 3, 2, 1).reshape(batch, 2, -1)
#         gyrF = xFreq[:, 4:8, :, :].permute(0, 3, 2, 1).reshape(batch, 2, -1)
#         magF = xFreq[:, 8:, :, :].permute(0, 3, 2, 1).reshape(batch, 2, -1)

#         accSensor = self.acc(accT, accF)
#         gyrSensor = self.gyr(gyrT, gyrF)
#         magSensor = self.mag(magT, magF)

#         motion = self.motionInterpreter(accSensor, gyrSensor)
#         motion = self.motionDownsmapler(motion)
#         motion = self.motionClassifier(motion)

#         mag = self.magneticInterpreter(magSensor, magSensor)
#         mag = self.magDownsmapler(mag)
#         mag = self.magClassifier(mag)

#         return motion, mag
