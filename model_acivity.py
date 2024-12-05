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
from einops import rearrange
from einops.layers.torch import Rearrange
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from timm.models.layers import DropPath
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchaudio.models import Conformer

# import auc F1 score  from torch
from torchmetrics.classification import (
    AUROC,
    ROC,
    Accuracy,  # F1Score,
    ConfusionMatrix,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    Precision,
    Recall,
)
from tqdm import tqdm

from datasets.MotionSense.processor import ACTIONS
from myutils import *
from utils.augmentation import Augmentation
from utils.ConvNeXt import ConvNeXtV2, convnextv2_base
from utils.correction import rotateToWorldFrame
from utils.geoPreprocessor import GravityRemoval, magRescale
from utils.mLoss import (
    CISSL,
    DISSL,
    AngleLoss,
    ContrastiveLoss,
    DistanceLoss,
    PairLoss,
    StudentNet,
    reweightMSELoss,
    simpleDistanceLoss,
    timeWiseLoss,
)
from utils.mnetwork import (
    TrigonometricActivation,
    angleActivation,
    channelShffuleNet,
    depthwiseSeparableConvolution2d,
    mResnet1d,
    mResnet2d,
    mThresholdActivation,
    vectorAngleActivation,
)
from utils.mplot import plot_to_image, plotConfusionMatrix
from utils.mscheduler import (
    CosineWarmupScheduler,
    ReduceLROnPlateauWarmup,
    dynamicScheduler,
)
from utils.stepDetection import stepDetection

torch.set_float32_matmul_precision("high")

if os.environ.get("WORLD_SIZE") is None:
    # get execution environment file name
    filename = sys.argv[0]
    if "train_downstream" in filename:
        # BATCH_SIZE = 1024 * 4
        BATCH_SIZE = 1024
    else:
        BATCH_SIZE = 800
else:
    filename = sys.argv[0]
    if "train_downstream" in filename:
        BATCH_SIZE = 1024 * 8
    elif "train_upstream" in filename:
        # BATCH_SIZE = int(1024 // 4)
        BATCH_SIZE = 512

# BATCH_SIZE = 2000 * 3
# BATCH_SIZE = 200 * 9 * 2
BATCH_SIZE = 1024
print(f"{cl.Fore.YELLOW}Batch size as {BATCH_SIZE}{cl.Style.reset}")

WINDOW_SIZE = 100
DROPFRONT = 3000
ACCUMULATION_STEPS = 32


class activityModel(pl.LightningModule):
    def __init__(
        self,
        root: str = "",
        lr: float = 1e-2,
        num_classes: int = 2,
        train_weight: Union[list, np.array, torch.tensor] = np.ones(2),
        val_weight: Union[list, np.array, torch.tensor] = np.ones(2),
        num_batches: int = 25,
        upsteamtask: bool = False,
        gradient_accumulation: bool = False,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        include_top: bool = False,
    ):
        super().__init__()
        self.datasetPath = root
        self.automatic_optimization = False
        self.hparams.batch_size = BATCH_SIZE
        self.hparams.lr = lr
        self.hparams.num_batches = num_batches
        self.hparams.patience = 20
        self.hparams.cycle = num_batches
        self.hparams.warmup = 10
        self.hparams.num_classes = num_classes
        self.hparams.num_layer = 6
        self.hparams.upsteamtask = upsteamtask
        self.hparams.gradient_accumulation = gradient_accumulation
        self.hparams.l1_lambda = l1_lambda
        self.hparams.l2_lambda = l2_lambda
        self.hparams.include_top = include_top

        if not self.hparams.upsteamtask:
            print(f"{cl.Fore.YELLOW}Using downstream task{cl.Style.reset}")
            print(f"{cl.Fore.YELLOW}Using {train_weight} classes{cl.Style.reset}")
            print(f"{cl.Fore.YELLOW}Using {val_weight} classes{cl.Style.reset}")
            assert len(train_weight) == self.hparams.num_classes == len(val_weight)
            self.train_loss = nn.CrossEntropyLoss(weight=train_weight)
            self.val_loss = nn.CrossEntropyLoss(weight=val_weight)
            self.test_loss = nn.CrossEntropyLoss()
        else:
            self.train_loss = self.val_loss = self.test_loss = CISSL(
                z_dim=1024,
                proj_dim=int(BATCH_SIZE * int(os.getenv("WORLD_SIZE", 1))),
                hidden_dim=int(BATCH_SIZE * int(os.getenv("WORLD_SIZE", 1))),
                # n_equiv=1024,
            )
        task = "multiclass" if self.hparams.num_classes > 2 else "binary"
        self.train_metrics = self._get_metrics("train")
        self.val_metrics = self._get_metrics("val")
        self.test_metrics = self._get_metrics("test")

        self.augmentation = Augmentation(
            upstreamtask=self.hparams.upsteamtask,
            mode="regression",
        )

        self.accF_encoder = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=(WINDOW_SIZE // 2 + 1) * 4,
                nhead=12,
                dim_feedforward=512,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            Rearrange("b t (c f) -> b (c f) t", t=2, f=WINDOW_SIZE // 2 + 1, c=4),
            nn.BatchNorm1d((WINDOW_SIZE // 2 + 1) * 4),
            Rearrange("b (c f) t -> b t (c f)", t=2, f=WINDOW_SIZE // 2 + 1, c=4),
        )
        self.gyrF_encoder = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=(WINDOW_SIZE // 2 + 1) * 4,
                nhead=12,
                dim_feedforward=512,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            Rearrange("b t (c f) -> b (c f) t", t=2, f=WINDOW_SIZE // 2 + 1, c=4),
            nn.BatchNorm1d((WINDOW_SIZE // 2 + 1) * 4),
            Rearrange("b (c f) t -> b t (c f)", t=2, f=WINDOW_SIZE // 2 + 1, c=4),
        )

        self.accT_decoder = nn.TransformerDecoderLayer(
            d_model=(WINDOW_SIZE // 2 + 1) * 4,
            nhead=12,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.accT_decoder_norm = nn.Sequential(
            Rearrange(
                "b t (c f) -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
            Rearrange("b t c f -> b c t f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            nn.BatchNorm2d(4),
            Rearrange("b c t f -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            Rearrange(
                "b t c f -> b t (c f)", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
        )

        self.gyrT_decoder = nn.TransformerDecoderLayer(
            d_model=(WINDOW_SIZE // 2 + 1) * 4,
            nhead=12,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.gyrT_decoder_norm = nn.Sequential(
            Rearrange(
                "b t (c f) -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
            Rearrange("b t c f -> b c t f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            nn.BatchNorm2d(4),
            Rearrange("b c t f -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            Rearrange(
                "b t c f -> b t (c f)", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
        )

        self.accgyrDecoder = nn.TransformerDecoderLayer(
            d_model=(WINDOW_SIZE // 2 + 1) * 4,
            nhead=12,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.accgyrDecoder_norm = nn.Sequential(
            Rearrange(
                "b t (c f) -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
            Rearrange("b t c f -> b c t f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            # nn.InstanceNorm2d(4),
            nn.BatchNorm2d(4),
        )

        self.featureExtraction = ConvNeXtV2(
            in_chans=4,
            num_classes=1024,
            drop_path_rate=0.1,
            depths=[2, 2, 6, 2],
            dims=[40, 80, 160, 320],
            # depths=[3, 3, 27, 3],
            # dims=[128, 256, 512, 1024],
            head_init_scale=1.0,
        )

        if self.hparams.upsteamtask:
            pass
        else:
            # if self.hparams.include_top:
            #     self.top = StudentNet(in_dim=4802, out_dim=1024, hidden_dim=2048)
            #     for param in self.top.parameters():
            #         param.requires_grad = False
            # # freeze
            # for param in self.accF_encoder.parameters():
            #     param.requires_grad = False
            # for param in self.gyrF_encoder.parameters():
            #     param.requires_grad = False
            # # for param in self.magF_encoder.parameters():
            # #     param.requires_grad = False

            # for param in self.accT_decoder.parameters():
            #     param.requires_grad = False
            # for param in self.gyrT_decoder.parameters():
            #     param.requires_grad = False
            # # for param in self.magT_decoder.parameters():
            # #     param.requires_grad = False

            # for param in self.accT_decoder_norm.parameters():
            #     param.requires_grad = False
            # for param in self.gyrT_decoder_norm.parameters():
            #     param.requires_grad = False
            # # for param in self.magT_decoder_norm.parameters():
            # #     param.requires_grad = False

            # for param in self.accgyrDecoder.parameters():
            #     param.requires_grad = False
            # # for param in self.magMotionDecoder.parameters():
            # #     param.requires_grad = False

            # for param in self.accgyrDecoder_norm.parameters():
            #     param.requires_grad = False
            # # for param in self.magMotionDecoder_norm.parameters():
            # #     param.requires_grad = False

            # for param in self.featureExtraction.parameters():
            #     param.requires_grad = False

            # # for param in self.featureExtraction.head.parameters():
            # #     param.requires_grad = True
            # # for param in self.featureExtraction.norm.parameters():
            # #     param.requires_grad = True

            hidden = 128
            """
            angle
            """
            self.projection = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.1),
                nn.Linear(1024, hidden),
                nn.BatchNorm1d(hidden),
            )
            self.mlp_1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                # nn.BatchNorm1d(hidden),
                # nn.LogSoftmax(dim=-1),
                nn.ReLU(),
                #
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                # nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )
            self.mlp_2 = nn.Sequential(
                nn.Linear(hidden, hidden),
                # nn.BatchNorm1d(hidden),
                nn.Dropout(0.1),
                # nn.LogSoftmax(dim=-1),
                nn.ReLU(),
                #
                # nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                # nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )
            self.mlp_3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden, self.hparams.num_classes),
                nn.Softmax(dim=-1),
            )

    def _get_params(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.extend([x.view(-1) for x in param])
        return torch.cat(params)

    def _get_metrics(self, postfix: str = ""):
        task = "multiclass" if self.hparams.num_classes > 2 else "binary"
        if postfix != "":
            postfix = "/" + postfix
        return {
            "auc"
            + postfix: AUROC(task=task, num_classes=self.hparams.num_classes).cuda(),
            "f1"
            + postfix: F1Score(task=task, num_classes=self.hparams.num_classes).cuda(),
            "accuracy"
            + postfix: Accuracy(task=task, num_classes=self.hparams.num_classes).cuda(),
            "precision"
            + postfix: Precision(
                task=task, num_classes=self.hparams.num_classes
            ).cuda(),
            "recall"
            + postfix: Recall(task=task, num_classes=self.hparams.num_classes).cuda(),
            "confusion_matrix"
            + postfix: MulticlassConfusionMatrix(
                num_classes=self.hparams.num_classes, normalize="true"
            ).cuda(),
        }

    def load_upstream(self, path):
        self.hparams.weightpath = path
        # save the weight path in the hyperparameters
        # self.save_hyperparameters()
        # self.logger.log_hyperparams(self.hparams)
        checkpoint = torch.load(path)
        weight = checkpoint["state_dict"]

        # remove all the prefix with mlp from the weight dict
        # weight = {k: v for k, v in weight.items() if "featureLayer." not in k}
        # weight = {k: v for k, v in weight.items() if "featureLayer2." not in k}
        # # replace the prefix "train_loss.student_proj" to mlp
        # weight = {
        #     k.replace("train_loss.student_proj.net", "mlp"): v
        #     for k, v in weight.items()
        # }
        if self.hparams.include_top:
            weight = {
                k.replace("train_loss.student_proj.net", "top"): v
                for k, v in weight.items()
            }

        # weight = {
        #     k.replace("train_loss.student_proj.net.0", "mlp.0"): v
        #     for k, v in weight.items()
        # }

        # # For DeBUG
        # for k, v in weight.items():
        #     print(f"{cl.Fore.yellow}    {k}{cl.Style.reset}")
        # c = input("Do you want to load the weight of the mlp? [y/n]")
        # if c == "y":
        #     pass
        # elif c == "n":
        #     exit()
        # else:
        #     raise ValueError("Invalid input")

        # weight = {k: v for k, v in weight.items() if "mlp." not in k}
        weight = {k: v for k, v in weight.items() if "train_loss" not in k}
        weight = {k: v for k, v in weight.items() if "val_loss" not in k}
        weight = {k: v for k, v in weight.items() if "test_loss" not in k}
        weight = {
            k.replace("featureExtracton", "featureExtraction"): v
            for k, v in weight.items()
        }
        # weight = {k: v for k, v in weight.items() if "mlp." not in k or int(k[4]) < 4}

        self.load_state_dict(weight, strict=False)
        print(f"Loaded upstream model from {path} with keys :")
        # print all the keys in the weight
        for k, v in weight.items():
            print(f"{cl.Fore.yellow}    {k}{cl.Style.reset}")

        # # reinit all the wieht in mlp
        # for m in self.mlp.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)

    def base_forward(self, x):
        # xF = x['freq']
        xT = x["time"]  # (batch, 12, WINDOW_SIZE)
        xF = x["freq"]  # (batch, 12, WINDOW_SIZE // 2 + 1, 2)
        batch = xT.shape[0]

        # shape : (batch, WINDOW_SIZE, (WINDOW_SIZE // 2 + 1)*4)
        accT = rearrange(xT[:, :4, :], "b c t -> b t c").repeat(
            1, 1, WINDOW_SIZE // 2 + 1
        )
        gyrT = rearrange(xT[:, 4:8, :], "b c t -> b t c").repeat(
            1, 1, WINDOW_SIZE // 2 + 1
        )

        # shape : (batch, 2, (WINDOW_SIZE // 2 + 1)*4)
        accF = rearrange(
            xF[:, :4, :, :], "b c f t -> b t (f c)", t=2, f=WINDOW_SIZE // 2 + 1, c=4
        )
        gyrF = rearrange(
            xF[:, 4:8, :, :], "b c f t -> b t (f c)", t=2, f=WINDOW_SIZE // 2 + 1, c=4
        )

        ###############################################################
        accF = self.accF_encoder(accF)
        gyrF = self.gyrF_encoder(gyrF)
        # magF = self.magF_encoder(magF)

        accT = self.accT_decoder_norm(self.accT_decoder(accT, accF))
        gyrT = self.gyrT_decoder_norm(self.gyrT_decoder(gyrT, gyrF))
        # magT = self.magT_decoder_norm(self.magT_decoder(magT, magF))

        ###############################################################
        motion = self.accgyrDecoder(accT, gyrT)
        motion = self.accgyrDecoder_norm(motion)


        features = self.featureExtraction(motion)

        return features

    def forward(self, x):
        features = self.base_forward(x)
        if not self.hparams.upsteamtask:
            if self.hparams.include_top:
                features = self.top(features)
            final = self.projection(features)
            final = self.mlp_1(final) + final
            final = self.mlp_2(final) + final
            final = self.mlp_3(final)
        else:
            final = features

        return final  

    def configure_optimizers(self):
        if self.hparams.upsteamtask:
            # opt = torch.optim.SGD(
            #     self.parameters(),
            #     lr=self.hparams.lr,
            #     # momentum=0.9,
            # )
            # print(f"Using SGD optimizer with lr={self.hparams.lr}")
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            print(f"Using Adam optimizer with lr={self.hparams.lr}")
            lr_scheduler = {
                "scheduler": CosineWarmupScheduler(
                    opt,
                    # warmup=self.hparams.num_batches,
                    warmup=self.hparams.warmup,
                    # warmup=10,
                    # max_iters=self.hparams.cycle // 2,
                    max_iters=self.hparams.cycle,
                    # opt, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
                ),
                "monitor": "loss/train_step",
                "interval": "epoch",
            }
            return [opt], [lr_scheduler]
        else:
            # opt = torch.optim.SGD(
            #     self.parameters(),
            #     lr=self.hparams.lr,
            #     # momentum=0.9,
            # )
            # print(f"Using SGD optimizer with lr={self.hparams.lr}")
            # params = (
            #     list(self.projection.parameters())
            #     + list(self.mlp_1.parameters())
            #     + list(self.mlp_2.parameters())
            #     + list(self.mlp_3.parameters())
            # )
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            print(f"Using Adam optimizer with lr={self.hparams.lr}")

            lr_scheduler = {
                "scheduler": CosineWarmupScheduler(
                    opt,
                    # warmup=self.hparams.num_batches,
                    warmup=self.hparams.warmup,
                    # warmup=10,
                    # max_iters=self.hparams.cycle // 2,
                    max_iters=self.hparams.cycle,
                    # opt, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
                ),
                "monitor": "loss/train_step",
                "interval": "epoch",
            }

            return [opt], [lr_scheduler]

    def on_train_start(self) -> None:
        # params = {
        #     "batch_size": self.hparams.batch_size,
        #     "lr": self.hparams.lr,
        #     "l1_lambda": self.hparams.l1_lambda,
        #     "l2_lambda": self.hparams.l2_lambda,
        #     "num_classes": self.hparams.num_classes,
        #     "upsteamtask": self.hparams.upsteamtask,
        #     "include_top": self.hparams.include_top,
        #     "gradient_accumulation": self.hparams.gradient_accumulation,
        #     "cycle": self.hparams.cycle,
        #     "warmup": self.hparams.warmup,
        #     "num_batches": self.hparams.num_batches,
        #     "weightpath": (
        #         self.hparams.weightpath if hasattr(self.hparams, "weightpath") else ""
        #     ),
        # }

        augmentation_configuration = {
            "y_reverse": self.augmentation._ymag_reverse_percentage,
            "channel permute": self.augmentation._channel_permute_percentage,
            "time flip": self.augmentation._flip_percentage,
            "scale": self.augmentation._scale_percentage,
            "shift": self.augmentation._shift_percentage,
            "gaussian": self.augmentation._gaussian_noise_percentage,
            "rotate": self.augmentation._rotate_percentage,
            "base": self.augmentation._time_masking_percentage,
            "time masking": self.augmentation._frequency_masking_percentage,
            "freq masking": self.augmentation._frequency_masking_percentage,
            "channel masking": self.augmentation._channel_masking_percentage,
            "masking": self.augmentation._mask,
            "num_channel": self.augmentation._num_channel,
        }
        params = self.hparams
        params.update(augmentation_configuration)

        self.logger.log_hyperparams(params=params, metrics={"task": 0})

        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        for key, value in self.train_metrics.items():
            value.reset()

        self.train_distribution = []

        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        for key, value in self.val_metrics.items():
            value.reset()
        self.val_distribution = []

        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        for key, value in self.test_metrics.items():
            value.reset()
        self.test_distribution = []

        return super().on_test_epoch_start()

    def training_step(self, batch, batch_idx):

        if not self.hparams.upsteamtask:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            if not self.hparams.gradient_accumulation:
                opt.zero_grad()

            x, y = batch
            # y = torch.argmax(y, dim=-1)
            # x_, y_ = self.augmentation(data=x, label=y)
            x_, y_ = x, y
            predict = self.forward(x_)
            # predict, features = self.forward(x)
            loss = self.train_loss(predict, y_)
            # otherLoss = {
            #     key + "/train": value for key, value in loss.items() if key != "loss"
            # }
            # self.train_distribution.append(otherLoss.pop("error_distribution/train"))
            # loss = loss["loss"]

            params = self._get_params()
            l2_reg = self.hparams.l2_lambda * torch.norm(params, 2)

            loss = loss + l2_reg

            loss.backward()

            opt.step()

            sch.step()

            self.log_dict(
                {
                    "l2_reg": l2_reg,
                },
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        else:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            if not self.hparams.gradient_accumulation:
                opt.zero_grad()
            x, y = batch
            x_a, x_b = self.augmentation(x, pair=True)

            predict_a = self.forward(x_a)
            predict_b = self.forward(x_b)
            # predict_a, features_a = self.forward(x_a)
            # predict_b, features_b = self.forward(x_b)

            loss = self.train_loss(predict_a, predict_b)

            params = self._get_params()
            l1_reg = self.hparams.l1_lambda * torch.norm(params, 1)
            l2_reg = self.hparams.l2_lambda * torch.norm(params, 2)

            total_loss = loss + l1_reg + l2_reg

            # # scale losses by 1/N (for N batches of gradient accumulation)
            # loss = loss / ACCUMULATION_STEPS
            # self.manual_backward(loss)
            # # accumulate gradients of N batches
            # if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            #     opt.step()
            #     opt.zero_grad()
            #     sch = self.lr_schedulers()
            #     sch.step()
            if not self.hparams.gradient_accumulation:
                # total_loss.backward()
                # self.manual_backward(total_loss)
                # if self.hparams.upsteamtask:
                # clip gradients
                self.clip_gradients(
                    opt,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm",
                )
                opt.step()
                # sch.step(epoch=self.global_step, loss=loss)
                # if the sch is ReduceLROnPlateau, then use loss as the metric
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(loss)
                elif isinstance(sch, CosineWarmupScheduler) or isinstance(
                    sch, pl.tuner.lr_finder._ExponentialLR
                ):
                    sch.step()
                else:
                    print(f"{type(sch)}")
                    raise NotImplementedError

                # sch.step(loss)
            else:
                loss = loss / ACCUMULATION_STEPS
                self.clip_gradients(
                    opt,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm",
                )

                total_loss.backward()
                # self.manual_backward(total_loss)
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    opt.step()
                    opt.zero_grad()

                    sch.step(epoch=self.global_step, loss=loss)

            self.log("l1_reg/train", l1_reg, on_epoch=True, sync_dist=True)
            self.log("l2_reg/train", l2_reg, on_epoch=True, sync_dist=True)
            self.log(
                "total_loss/train",
                total_loss,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
            )

        if not self.hparams.upsteamtask:
            for key, value in self.train_metrics.items():
                value.update(predict.to(value.device), y.to(value.device))
                if "confusion_matrix" in key:
                    continue
                self.log(
                    f"{key}",
                    value.compute(),
                    on_epoch=True,
                    sync_dist=True,
                )

            self.log_dict(
                {
                    "loss/train": loss,
                    # "auc/train": auc,
                    # "f1/train": f1,
                    # "acc/train": acc,
                    # "precision/train": precision,
                    # "recall/train": recall,
                },
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log_dict(
                {
                    "loss/train": loss,
                },
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if not self.hparams.upsteamtask:
            x, y = batch
            # x["time"] = self.augmentation.lowpass(x["time"])
            # y = torch.argmax(y, dim=-1)
            predict = self.forward(x)
            # predict, features = self.forward(x)
            loss = self.val_loss(predict, y)
            # self.val_distribution.append(otherLoss.pop("error_distribution/val"))
            # loss = loss["loss"]
            # y = torch.argmax(y, dim=-1)
        else:
            x, y = batch
            x_a, x_b = self.augmentation(x, pair=True)
            predict_a = self.forward(x_a)
            predict_b = self.forward(x_b)

            # predict_a, features_a = self.forward(x_a)
            # predict_b, features_b = self.forward(x_b)
            loss = self.val_loss(predict_a, predict_b)

        params = self._get_params()
        l1_reg = self.hparams.l1_lambda * torch.norm(params, 1)
        l2_reg = self.hparams.l2_lambda * torch.norm(params, 2)
        total_loss = loss + l1_reg + l2_reg

        self.log("l1_reg/val", l1_reg, on_epoch=True, sync_dist=True)
        self.log("l2_reg/val", l2_reg, on_epoch=True, sync_dist=True)
        self.log("total_loss/val", total_loss, on_epoch=True, sync_dist=True)

        if not self.hparams.upsteamtask:
            for key, value in self.val_metrics.items():
                value.update(predict.to(value.device), y.to(value.device))
                if "confusion_matrix" in key:
                    continue
                self.log(
                    f"{key}",
                    value.compute(),
                    on_epoch=True,
                    sync_dist=True,
                )

            self.log_dict(
                {
                    "loss/val": loss,
                    # "auc/val": auc,
                    # "f1/val": f1,
                    # "acc/val": acc,
                    # "precision/val": precision,
                    # "recall/val": recall,
                },
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.log_dict(
                {
                    "loss/val": loss,
                },
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
            )

        # self.logger.experiment.add_embedding(
        #     features,
        #     metadata=y,
        #     label_img=self.makeImage(x["time"]),
        #     global_step=self.global_step,
        #     tag=f"val_embedding_{batch_idx:02d}",
        # )

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        # x["time"] = self.augmentation.lowpass(x["time"])
        # y = torch.argmax(y, dim=-1)
        if not self.hparams.upsteamtask:
            predict = self.forward(x)
            # predict, features = self.forward(x)
            loss = self.test_loss(predict, y)
            # otherLoss = {
            #     key + "/test": value for key, value in loss.items() if key != "loss"
            # }
            # self.test_distribution.append(otherLoss.pop("error_distribution/test"))
            # loss = loss["loss"]
            # get the index of the max as the label
            # y = torch.argmax(y, dim=-1)
            for key, value in self.test_metrics.items():
                value.update(predict.to(value.device), y.to(value.device))
                if "confusion_matrix" in key:
                    continue
                self.log(
                    f"{key}",
                    value.compute(),
                    on_epoch=True,
                    sync_dist=True,
                )
        else:
            x_a, x_b = self.augmentation(x, pair=True)
            predict_a = self.forward(x_a)
            predict_b = self.forward(x_b)

            # predict_a, features_a = self.forward(x_a)
            # predict_b, features_b = self.forward(x_b)
            loss = self.test_loss(predict_a, predict_b)
            # features = features_a

        self.log("loss/test", loss, sync_dist=True, on_epoch=True, prog_bar=True)
        # self.logger.experiment.add_embedding(
        #     features,
        #     metadata=y,
        #     label_img=self.makeImage(x["time"]),
        #     global_step=self.global_step,
        #     tag=f"test_embedding_{batch_idx:02d}",
        # )

        return loss

    # def plot_wave(self,batch,):

    def log_cmimage(self, cm, mode=None):
        fig = plotConfusionMatrix(cm, labels=ACTIONS)
        figure = plot_to_image(fig)
        # store the confusion matrix to tensorboard
        self.logger.experiment.add_image(
            f"confusion_matrix/{mode}",
            figure,
            self.current_epoch,
            dataformats="CHW",
        )
        # print(cm)

    def makeImage(self, tensor):
        bottom, top = 0.01, 0.99
        # make tensor to type torch.float32
        tensor = tensor.float()
        tensor = tensor.unsqueeze(1)

        # acc
        tensor[:, :, :3, :] = torch.clip(
            (torch.clip(tensor[:, :, :3, :], -1, 1) + 1) / 2, bottom, top
        )
        tensor[:, :, 3, :] = torch.clip(tensor[:, :, 3, :], bottom, top)

        # gyr
        tensor[:, :, 4:7, :] = torch.clip(
            (torch.clip(tensor[:, :, 4:7, :], -1, 1) + 1) / 2, bottom, top
        )
        tensor[:, :, 7, :] = torch.clip(tensor[:, :, 7, :], bottom, top)

        # mag
        tensor[:, :, 8:11, :] = torch.clip(
            (torch.clip(tensor[:, :, 8:11, :], -1, 1) + 1) / 2, bottom, top
        )
        # clear other channels
        tensor[:, :, 11, :] = torch.clip(tensor[:, :, 11, :], bottom, top)

        tensor = tensor.reshape(-1, 3, 4, WINDOW_SIZE)

        return tensor

    def on_train_epoch_end(self):
        if not self.hparams.upsteamtask:
            # log confusion matrix
            cm = self.train_metrics["confusion_matrix/train"].compute()
            self.log_cmimage(cm, mode="train")

        for key, value in self.train_metrics.items():
            value.reset()

        # if not self.hparams.upsteamtask:
        #     self.train_distribution = torch.cat(self.train_distribution, dim=0)
        #     self.logger.experiment.add_histogram(
        #         "error_distribution/train",
        #         self.train_distribution,
        #         global_step=self.global_step,
        #     )
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        if not self.hparams.upsteamtask:
            # log confusion matrix
            cm = self.val_metrics["confusion_matrix/val"].compute()
            self.log_cmimage(cm, mode="val")
        for key, value in self.val_metrics.items():
            value.reset()

        # if not self.hparams.upsteamtask:
        #     self.val_distribution = torch.cat(self.val_distribution, dim=0)
        #     self.logger.experiment.add_histogram(
        #         "error_distribution/val",
        #         self.val_distribution,
        #         global_step=self.global_step,
        #     )
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        if not self.hparams.upsteamtask:
            # log confusion matrix
            cm = self.test_metrics["confusion_matrix/test"].compute()
            self.log_cmimage(cm, mode="test")

        for key, value in self.test_metrics.items():
            value.reset()

        # if not self.hparams.upsteamtask:
        #     self.test_distribution = torch.cat(self.test_distribution, dim=0)
        #     self.logger.experiment.add_histogram(
        #         "error_distribution/test",
        #         self.test_distribution,
        #         global_step=self.global_step,
        #     )
        return super().on_test_epoch_end()

    # TODO:legacy code, remove later
    # def train_dataloader(self):
    #     # list all files in datasets folder with .pt
    #     files = list(Path(self.datasetPath).glob("*.pt"))
    #     randomize = True
    #     data = []
    #     for file in files:
    #         data.append(torch.load(file))

    #     if randomize:
    #         # shuffle the files
    #         np.random.shuffle(data)

    #     # convert to datasets
    #     splitpoint = len(data) // 10 * 8
    #     self.train_dataset = torch.utils.data.ConcatDataset(data[:splitpoint])
    #     self.val_dataset = torch.utils.data.ConcatDataset(data[splitpoint:])
    #     # construct the dataloader
    #     loader = DataLoader(
    #         self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6
    #     )
    #     return loader

    # def val_dataloader(self):
    #     # construct the dataloader
    #     loader = DataLoader(
    #         self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    #     )
    #     return loader

    # def test_dataloader(self):
    #     return test_loader

    @torch.no_grad()
    def predict(self, batch):
        # make both xF and xT to be a batch of size 1 and on the same device as the model
        # grep all x and y
        x, y = batch
        # x["freq"] = x["freq"].to(self.device)
        # x["time"] = x["time"].to(self.device)
        prediction = self.forward(x)
        loss = self.test_loss(prediction, y)

        # check key 'error_distribution' in loss or not if yes pop
        if "error_distribution" in loss.keys():
            loss.pop("error_distribution")
        pprint(loss)
        return prediction
