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
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MulticlassConfusionMatrix,
    Precision,
    Recall,
)
from tqdm import tqdm

from utils.augmentation import Augmentation
from utils.ConvNeXt import ConvNeXtV2, convnextv2_base
from utils.correction import rotateToWorldFrame
from utils.geoPreprocessor import GravityRemoval, magRescale
from utils.mLoss import CISSL, DISSL, ContrastiveLoss, StudentNet, reweightMSELoss
from utils.mnetwork import (
    channelShffuleNet,
    depthwiseSeparableConvolution2d,
    mResnet1d,
    mResnet2d,
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


class IODModel(pl.LightningModule):
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
            # reweight = [756, 844]

            # print(
            #     f"{cl.Fore.yellow}Secondary reweighting with ratio {reweight} {cl.Style.reset}"
            # )
            # train_weight[0] *= reweight[0]
            # train_weight[1] *= reweight[1]
            # train_weight /= train_weight.sum()

            self.train_loss = nn.CrossEntropyLoss(
                weight=torch.tensor(train_weight, dtype=torch.float32),
                # label_smoothing=0.07,
                # reduction='sum',
            )
            self.val_loss = nn.CrossEntropyLoss(
                weight=torch.tensor(val_weight, dtype=torch.float32),
            )
            self.test_loss = nn.CrossEntropyLoss()

            # self.train_loss = nn.KLDivLoss(
            #     log_target=True,
            #     # weight=torch.tensor(train_weight, dtype=torch.float32),
            #     # label_smoothing=0.07,
            #     # reduction='sum',
            # )
            # self.val_loss = nn.KLDivLoss(
            #     log_target=True,
            #     # weight=torch.tensor(val_weight, dtype=torch.float32),
            # )
            # self.test_loss = nn.KLDivLoss(log_target=True,)

            # self.train_loss = reweightMSELoss(
            #     weight=torch.tensor(train_weight, dtype=torch.float32)
            # )
            # self.val_loss = reweightMSELoss(
            #     weight=torch.tensor(val_weight, dtype=torch.float32)
            # )
            # self.test_loss = reweightMSELoss()
        else:
            # self.train_loss = ContrastiveLoss(BATCH_SIZE)
            # self.val_loss = ContrastiveLoss(BATCH_SIZE)
            # self.test_loss = ContrastiveLoss(BATCH_SIZE)

            # self.train_loss = self.val_loss = self.test_loss = CISSL(
            #     z_dim=1024,
            #     proj_dim=int(BATCH_SIZE * int(os.getenv("WORLD_SIZE", 1))),
            #     hidden_dim=int(BATCH_SIZE * int(os.getenv("WORLD_SIZE", 1))),
            #     # n_equiv=1024,
            # )
            self.train_loss = self.val_loss = self.test_loss = CISSL(
                z_dim=1024,
                proj_dim=int(BATCH_SIZE * int(os.getenv("WORLD_SIZE", 1))),
                hidden_dim=int(BATCH_SIZE * int(os.getenv("WORLD_SIZE", 1))),
                # n_equiv=1024,
            )

        # Train metrics
        self.val_AUC = AUROC(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.train_AUC = AUROC(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.train_F1 = F1Score(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.train_Accuracy = Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.train_Precision = Precision(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.train_Recall = Recall(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.train_ConfusionMatrix = MulticlassConfusionMatrix(
            num_classes=self.hparams.num_classes,
            normalize="true",
        )
        if train_weight is not None:
            assert len(train_weight) == self.hparams.num_classes
        else:
            train_weight = np.ones(self.hparams.num_classes) / self.hparams.num_classes

        # Validation metrics
        self.val_AUC = AUROC(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.val_AUC = AUROC(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.val_F1 = F1Score(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.val_Accuracy = Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.val_Precision = Precision(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.val_Recall = Recall(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.val_ConfusionMatrix = MulticlassConfusionMatrix(
            num_classes=self.hparams.num_classes,
            normalize="true",
        )
        if val_weight is not None:
            assert len(val_weight) == self.hparams.num_classes
        else:
            val_weight = np.ones(self.hparams.num_classes) / self.hparams.num_classes

        # Test metrics
        self.val_AUC = AUROC(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.test_AUC = AUROC(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.test_F1 = F1Score(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.test_Accuracy = Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.test_Precision = Precision(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.test_Recall = Recall(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            average="macro",
        )
        self.test_ConfusionMatrix = MulticlassConfusionMatrix(
            num_classes=self.hparams.num_classes,
            normalize="true",
        )

        self.augmentation = Augmentation(
            upstreamtask=self.hparams.upsteamtask,
            mode="classification",
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
        self.magF_encoder = nn.Sequential(
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
            nn.InstanceNorm2d(4),
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
            nn.InstanceNorm2d(4),
            Rearrange("b c t f -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            Rearrange(
                "b t c f -> b t (c f)", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
        )
        self.magT_decoder = nn.TransformerDecoderLayer(
            d_model=(WINDOW_SIZE // 2 + 1) * 4,
            nhead=12,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.magT_decoder_norm = nn.Sequential(
            Rearrange(
                "b t (c f) -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
            Rearrange("b t c f -> b c t f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            nn.InstanceNorm2d(4),
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
            nn.InstanceNorm2d(4),
            Rearrange("b c t f -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            Rearrange(
                "b t c f -> b t (c f)", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
        )

        self.magMotionDecoder = nn.TransformerDecoderLayer(
            d_model=(WINDOW_SIZE // 2 + 1) * 4,
            nhead=12,
            dim_feedforward=512,
            dropout=0.1,
            activation=F.sigmoid,
            batch_first=True,
            norm_first=True,
        )

        self.magMotionDecoder_norm = nn.Sequential(
            Rearrange(
                "b t (c f) -> b t c f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4
            ),
            Rearrange("b t c f -> b c t f", t=WINDOW_SIZE, f=WINDOW_SIZE // 2 + 1, c=4),
            nn.InstanceNorm2d(4),
        )

        self.featureExtraction = ConvNeXtV2(
            in_chans=4,
            num_classes=1024,
            drop_path_rate=0.1,
            depths=[2, 2, 6, 2],
            dims=[40, 80, 160, 320],
            head_init_scale=1.0,
        )

        if self.hparams.upsteamtask:
            pass
        else:
            if self.hparams.include_top:
                self.top = StudentNet(in_dim=4802, out_dim=1024, hidden_dim=2048)
                for param in self.top.parameters():
                    param.requires_grad = False
            # freeze
            for param in self.accF_encoder.parameters():
                param.requires_grad = False
            for param in self.gyrF_encoder.parameters():
                param.requires_grad = False
            for param in self.magF_encoder.parameters():
                param.requires_grad = False

            for param in self.accT_decoder.parameters():
                param.requires_grad = False
            for param in self.gyrT_decoder.parameters():
                param.requires_grad = False
            for param in self.magT_decoder.parameters():
                param.requires_grad = False

            for param in self.accT_decoder_norm.parameters():
                param.requires_grad = False
            for param in self.gyrT_decoder_norm.parameters():
                param.requires_grad = False
            for param in self.magT_decoder_norm.parameters():
                param.requires_grad = False

            for param in self.accgyrDecoder.parameters():
                param.requires_grad = False
            for param in self.magMotionDecoder.parameters():
                param.requires_grad = False

            for param in self.accgyrDecoder_norm.parameters():
                param.requires_grad = False
            for param in self.magMotionDecoder_norm.parameters():
                param.requires_grad = False

            # for param in self.featureExtraction.parameters():
            #     param.requires_grad = False

            hidden = 128
            self.projection = nn.Sequential(
                nn.LayerNorm(1024),
                # DropPath(0.1),
                nn.Dropout(0.1),
                nn.Linear(1024, hidden),
                nn.LayerNorm(hidden),
            )

            self.mlp_1 = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.1),
                # DropPath(0.1),
                nn.Linear(hidden // 2, hidden),
                nn.BatchNorm1d(hidden),
                nn.PReLU(),
                #
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )

            self.mlp_2 = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.1),
                # DropPath(0.1),
                nn.Linear(hidden // 2, hidden),
                nn.BatchNorm1d(hidden),
                nn.PReLU(),
                #
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )

            self.mlp_3 = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.1),
                # DropPath(0.1),
                nn.Linear(hidden // 2, 2),
                nn.Softmax(dim=-1),
            )
            # for block in [self.projection, self.mlp_1, self.mlp_2, self.mlp_3]:
            #     for layer in block:
            #         if isinstance(layer, nn.Linear):
            #             nn.init.kaiming_normal(
            #                 layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            #             )
            #             nn.init.zeros_(layer.bias)
            #         elif isinstance(layer, nn.BatchNorm1d):
            #             nn.init.normal(layer.weight)
            #             nn.init.zeros_(layer.bias)

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
        magT = rearrange(xT[:, 8:, :], "b c t -> b t c").repeat(
            1, 1, WINDOW_SIZE // 2 + 1
        )

        # shape : (batch, 2, (WINDOW_SIZE // 2 + 1)*4)
        accF = rearrange(
            xF[:, :4, :, :], "b c f t -> b t (f c)", t=2, f=WINDOW_SIZE // 2 + 1, c=4
        )
        gyrF = rearrange(
            xF[:, 4:8, :, :], "b c f t -> b t (f c)", t=2, f=WINDOW_SIZE // 2 + 1, c=4
        )
        magF = rearrange(
            xF[:, 8:, :, :], "b c f t -> b t (f c)", t=2, f=WINDOW_SIZE // 2 + 1, c=4
        )

        ###############################################################
        accF = self.accF_encoder(accF)
        gyrF = self.gyrF_encoder(gyrF)
        magF = self.magF_encoder(magF)

        accT = self.accT_decoder_norm(self.accT_decoder(accT, accF))
        gyrT = self.gyrT_decoder_norm(self.gyrT_decoder(gyrT, gyrF))
        magT = self.magT_decoder_norm(self.magT_decoder(magT, magF))

        ###############################################################
        motion = self.accgyrDecoder(accT, gyrT)
        motion = self.accgyrDecoder_norm(motion)
        magT = self.magMotionDecoder(magT, motion)
        magT = self.magMotionDecoder_norm(magT)
        ###############################################################
        features = self.featureExtraction(magT)

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
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            print(f"Using Adam optimizer with lr={self.hparams.lr}")
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            print(f"Using Adam optimizer with lr={self.hparams.lr}")

        lr_scheduler = {
            "scheduler": CosineWarmupScheduler(
                opt,
                warmup=self.hparams.warmup,
                max_iters=self.hparams.cycle,
            ),
            "monitor": "loss/train_step",
            "interval": "epoch",
        }
        return [opt], [lr_scheduler]

    def on_train_start(self) -> None:
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
        # reset metrics
        self.train_Accuracy.reset()
        self.train_AUC.reset()
        self.train_F1.reset()
        self.train_Precision.reset()
        self.train_Recall.reset()
        self.train_ConfusionMatrix.reset()

        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        # reset metrics
        self.val_Accuracy.reset()
        self.val_AUC.reset()
        self.val_F1.reset()
        self.val_Precision.reset()
        self.val_Recall.reset()
        self.val_ConfusionMatrix.reset()

        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        # reset metrics
        self.test_Accuracy.reset()
        self.test_AUC.reset()
        self.test_F1.reset()
        self.test_Precision.reset()
        self.test_Recall.reset()
        self.test_ConfusionMatrix.reset()

        return super().on_test_epoch_start()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        if not self.hparams.gradient_accumulation:
            opt.zero_grad()

        if not self.hparams.upsteamtask:
            x, y = batch
            y = torch.argmax(y, dim=-1)
            x_ = self.augmentation(x)
            predict = self.forward(x_)
            # predict, features = self.forward(x)
            loss = self.train_loss(predict, y)
        else:
            x, y = batch
            x_a, x_b = self.augmentation(x, pair=True)

            predict_a = self.forward(x_a)
            predict_b = self.forward(x_b)

            loss = self.train_loss(predict_a, predict_b)

        if self.hparams.l1_lambda != 0.0:
            # all_L1_params = torch.cat([x.view(-1) for x in self.parameters()])
            # grep all decoder final layer weight
            all_L1_params = torch.cat(
                [x.view(-1) for x in self.accT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.gyrT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.magT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.accgyrDecoder_norm.parameters()]
                + [x.view(-1) for x in self.magMotionDecoder_norm.parameters()]
                + [x.view(-1) for x in self.featureExtraction.parameters()]
            )
            l1_reg = self.hparams.l1_lambda * torch.norm(all_L1_params, 1)
        else:
            l1_reg = 0.0

        if self.hparams.l2_lambda != 0.0:
            # all_L2_params = torch.cat([x.view(-1) for x in self.parameters()])
            all_L2_params = torch.cat(
                [x.view(-1) for x in self.accT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.gyrT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.magT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.accgyrDecoder_norm.parameters()]
                + [x.view(-1) for x in self.magMotionDecoder_norm.parameters()]
                + [x.view(-1) for x in self.featureExtraction.parameters()]
            )
            l2_reg = self.hparams.l2_lambda * torch.norm(all_L2_params, 2)
        else:
            l2_reg = 0.0

        total_loss = loss + l1_reg + l2_reg

        if not self.hparams.gradient_accumulation:
            total_loss.backward()
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
            "total_loss/train", total_loss, on_epoch=True, on_step=True, sync_dist=True
        )

        if not self.hparams.upsteamtask:
            self.train_Accuracy.update(predict, y)
            self.train_AUC.update(predict, y)
            self.train_F1.update(predict, y)
            self.train_Precision.update(predict, y)
            self.train_Recall.update(predict, y)
            self.train_ConfusionMatrix.update(predict, y)

            acc = self.train_Accuracy.compute()
            auc = self.train_AUC.compute()
            f1 = self.train_F1.compute()
            precision = self.train_Precision.compute()
            recall = self.train_Recall.compute()
            cm = self.train_ConfusionMatrix.compute()

            self.log_dict(
                {
                    "loss/train": loss,
                    "auc/train": auc,
                    "f1/train": f1,
                    "acc/train": acc,
                    "precision/train": precision,
                    "recall/train": recall,
                },
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("TO/train", cm[0, 0], on_epoch=True, sync_dist=True)
            self.log("TI/train", cm[1, 1], on_epoch=True, sync_dist=True)
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
            y = torch.argmax(y, dim=-1)
            predict = self.forward(x)
            # predict, features = self.forward(x)
            loss = self.val_loss(predict, y)
            # y = torch.argmax(y, dim=-1)
        else:
            x, y = batch
            x_a, x_b = self.augmentation(x, pair=True)
            predict_a = self.forward(x_a)
            predict_b = self.forward(x_b)

            # predict_a, features_a = self.forward(x_a)
            # predict_b, features_b = self.forward(x_b)
            loss = self.val_loss(predict_a, predict_b)

        if self.hparams.l1_lambda != 0.0:
            # all_L1_params = torch.cat([x.view(-1) for x in self.parameters()])
            all_L1_params = torch.cat(
                [x.view(-1) for x in self.accT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.gyrT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.magT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.accgyrDecoder_norm.parameters()]
                + [x.view(-1) for x in self.magMotionDecoder_norm.parameters()]
                + [x.view(-1) for x in self.featureExtraction.parameters()]
            )
            l1_reg = self.hparams.l1_lambda * torch.norm(all_L1_params, 1)
        else:
            l1_reg = 0.0

        if self.hparams.l2_lambda != 0.0:
            # all_L2_params = torch.cat([x.view(-1) for x in self.parameters()])
            all_L2_params = torch.cat(
                [x.view(-1) for x in self.accT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.gyrT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.magT_decoder_norm.parameters()]
                + [x.view(-1) for x in self.accgyrDecoder_norm.parameters()]
                + [x.view(-1) for x in self.magMotionDecoder_norm.parameters()]
                + [x.view(-1) for x in self.featureExtraction.parameters()]
            )
            l2_reg = self.hparams.l2_lambda * torch.norm(all_L2_params, 2)
        else:
            l2_reg = 0.0

        total_loss = loss + l1_reg + l2_reg

        self.log("l1_reg/val", l1_reg, on_epoch=True, sync_dist=True)
        self.log("l2_reg/val", l2_reg, on_epoch=True, sync_dist=True)
        self.log("total_loss/val", total_loss, on_epoch=True, sync_dist=True)

        if not self.hparams.upsteamtask:
            self.val_Accuracy.update(predict, y)
            self.val_AUC.update(predict, y)
            self.val_F1.update(predict, y)
            self.val_Precision.update(predict, y)
            self.val_Recall.update(predict, y)
            self.val_ConfusionMatrix.update(predict, y)

            acc = self.val_Accuracy.compute()
            auc = self.val_AUC.compute()
            f1 = self.val_F1.compute()
            precision = self.val_Precision.compute()
            recall = self.val_Recall.compute()
            cm = self.val_ConfusionMatrix.compute()

            self.log_dict(
                {
                    "loss/val": loss,
                    "auc/val": auc,
                    "f1/val": f1,
                    "acc/val": acc,
                    "precision/val": precision,
                    "recall/val": recall,
                },
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("TO/val", cm[0, 0], on_epoch=True, sync_dist=True)
            self.log("TI/val", cm[1, 1], on_epoch=True, sync_dist=True)
        else:
            self.log_dict(
                {
                    "loss/val": loss,
                },
                sync_dist=True,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        # x["time"] = self.augmentation.lowpass(x["time"])
        y = torch.argmax(y, dim=-1)
        if not self.hparams.upsteamtask:
            predict = self.forward(x)
            # predict, features = self.forward(x)
            loss = self.test_loss(predict, y)
            # get the index of the max as the label
            # y = torch.argmax(y, dim=-1)
        else:
            x_a, x_b = self.augmentation(x, pair=True)
            predict_a = self.forward(x_a)
            predict_b = self.forward(x_b)

            # predict_a, features_a = self.forward(x_a)
            # predict_b, features_b = self.forward(x_b)
            loss = self.test_loss(predict_a, predict_b)
            # features = features_a

        if not self.hparams.upsteamtask:
            self.test_Accuracy.update(predict, y)
            self.test_AUC.update(predict, y)
            self.test_F1.update(predict, y)
            self.test_Precision.update(predict, y)
            self.test_Recall.update(predict, y)
            self.test_ConfusionMatrix.update(predict, y)

        self.log("loss/test", loss, sync_dist=True, on_epoch=True, prog_bar=True)

        return loss

    # def plot_wave(self,batch,):

    def log_cmimage(self, cm, mode=None):
        fig = plotConfusionMatrix(cm)
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
            cm = self.train_ConfusionMatrix.compute()
            self.log_cmimage(cm, mode="train")

        # reset
        self.train_Accuracy.reset()
        self.train_AUC.reset()
        self.train_F1.reset()
        self.train_Precision.reset()
        self.train_Recall.reset()
        self.train_ConfusionMatrix.reset()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        if not self.hparams.upsteamtask:
            # log confusion matrix
            cm = self.val_ConfusionMatrix.compute()
            self.log_cmimage(cm, mode="val")

        # reset
        self.val_Accuracy.reset()
        self.val_AUC.reset()
        self.val_F1.reset()
        self.val_Precision.reset()
        self.val_Recall.reset()
        self.val_ConfusionMatrix.reset()
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        if not self.hparams.upsteamtask:
            # log confusion matrix
            cm = self.test_ConfusionMatrix.compute()
            self.log_cmimage(cm, mode="test")

            acc = self.test_Accuracy.compute()
            auc = self.test_AUC.compute()
            f1 = self.test_F1.compute()
            precision = self.test_Precision.compute()
            recall = self.test_Recall.compute()

            metrics = {
                "auc/test": auc,
                "f1/test": f1,
                "acc/test": acc,
                "precision/test": precision,
                "recall/test": recall,
                "task": 2,
            }
            self.log_dict(
                metrics,
                sync_dist=True,
                prog_bar=True,
            )
            self.log("TO/test", cm[0, 0], on_epoch=True, sync_dist=True)
            self.log("TI/test", cm[1, 1], on_epoch=True, sync_dist=True)

            self.logger.log_hyperparams(params=self.hparams, metrics=metrics)

        # reset
        self.test_Accuracy.reset()
        self.test_AUC.reset()
        self.test_F1.reset()
        self.test_Precision.reset()
        self.test_Recall.reset()
        self.test_ConfusionMatrix.reset()
        return super().on_test_epoch_end()


    @torch.no_grad()
    def predict(self, batch):
        # make both xF and xT to be a batch of size 1 and on the same device as the model
        # grep all x and y
        x, y = batch
        x["freq"] = x["freq"].to(self.device)
        x["time"] = x["time"].to(self.device)
        return self.forward(x)


class IndoorDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        window_size: int = WINDOW_SIZE,
        samplingRate: int = 100,
        dice: str = "step",
        mode: str = "train",
        label: bool = True,
        whole_testing: bool = False,
    ):
        self._labels = label
        self._whole_testing = whole_testing
        self.root_dir = root_dir
        self.dice = dice
        # list only the csv files in the folder
        self.files = [file for file in os.listdir(root_dir) if file.endswith(".csv")]
        print(f"Found {len(self.files)} files in {root_dir}")
        # sort
        self.files.sort()
        # np.random.shuffle(self.files)
        # random pick 6 files

        # #old sample
        # index = np.arange(3).tolist()
        # self.files = np.array(self.files)
        # self.test_files = self.files[index].tolist() + self.files[::-1][index].tolist()
        # mask = np.isin(self.files, self.test_files, invert=True)
        # self.train_files = self.files[mask].tolist()

        if self._whole_testing:
            self.test_files = self.files
            self.train_files = []
        else:
            if "migration" in self.root_dir:
                self.test_files = self.files[:3] + self.files[-3:]
                self.train_files = self.files[3:-3]
            else:
                self.train_files = self.files

            # # random set 20% as test and 80% as train
            # self.test_files = np.random.choice(
            #     self.files, size=int(len(self.files) * 0.2), replace=False
            # )
            # self.train_files = np.array(self.files)[
            #     np.isin(self.files, self.test_files, invert=True)
            # ].tolist()

        # remain = self.files[6:]
        # self.train_files = remain[:int(len(remain)*0.8)]
        # self.val_files = remain[int(len(remain)*0.8):]

        self.mode = None
        self.window_size = window_size
        self.samplingRate = samplingRate
        self.toFreq = torchaudio.transforms.Spectrogram(
            win_length=WINDOW_SIZE,
            n_fft=WINDOW_SIZE,
            hop_length=WINDOW_SIZE,
            # pad_mode='replicate',
            # center=False,
            # onesided=False,
            # power=4,
            # normalized=True,
        )
        # if self._whole_testing:
        #     self.load_data("test")
        # else:
        #     self.load_data("train")
        self.load_data(mode)

    def load_data(self, mode: str = "train"):
        if mode is None:
            mode = "train"

        if mode == self.mode:
            return
        self.mode = mode

        self.data = []
        self.splitPoint = [0] if mode == "pred" else None
        self.time = []

        if self.mode == "train":
            target_files = self.train_files
        elif self.mode == "test":
            target_files = self.test_files
        elif self.mode == "pred" and "migration" in self.root_dir:
            target_files = self.test_files
        elif self.mode == "pred" and "migration" not in self.root_dir:
            target_files = self.files
        else:
            raise ValueError(f"Invalid mode {self.mode}")

        for file in target_files:
            # print(f'Loading {file}')
            file_path = os.path.join(self.root_dir, file)
            df = pd.read_csv(file_path)

            if "migration" in self.root_dir or "nolabeled" in self.root_dir:
                # ex : 2023-02-20-11:17:17.csv
                date = df["timestamp"].iloc[0]
                date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
                location = "migration"
            elif "advio" in self.root_dir:
                date = 2018.5
                location = "advio"
            elif "MotionID" in self.root_dir:
                date = df["timestamp"].iloc[0]
                # convert to datetime from str ex :2021-03-16 07:45:14.890
                date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
                location = "MotionID"
            else:
                raise ValueError("Invalid dataset for background date.")

            acc = df[["acc.X", "acc.Y", "acc.Z"]].to_numpy()[DROPFRONT:]
            gyr = df[["gyr.X", "gyr.Y", "gyr.Z"]].to_numpy()[DROPFRONT:]
            mag = df[["mag.X", "mag.Y", "mag.Z"]].to_numpy()[DROPFRONT:]
            # acc, gyr, mag = rotateToWorldFrame(acc, gyr, mag)

            acc_norm = np.linalg.norm(acc, axis=1)
            if self.dice == "step":
                stepStart, stepEnd = stepDetection(acc_norm)

            time = df["timestamp"].to_numpy()[DROPFRONT:]
            assert len(mag) == len(acc) == len(gyr) == len(time)

            if self._labels:
                indoor = df["indoor"].to_numpy().astype(np.bool_)[DROPFRONT:]
                assert len(time) == len(indoor)

            # expand acc form 3 to 4
            acc = np.concatenate((acc, np.zeros((len(acc), 1))), axis=1)
            acc = GravityRemoval(acc=acc, location=location, date=date, rescale=False)
            acc = acc[:, :3]
            gyr = gyr  # / np.pi
            mag = magRescale(mag, location=location, date=date)

            if self.dice == "step":
                print(f"Loaded {file} with {len(stepStart)} steps.")
                # for End in stepEnd:
                for index in range(len(stepEnd)):
                    # if the window is overlap with previous window, skip
                    End = stepEnd[index]
                    Start = stepEnd[index] - self.window_size
                    # Start = stepStart[index]
                    # pad = self.window_size - (End - Start)
                    if End < self.window_size:
                        continue

                    accT = acc[Start:End].swapaxes(0, 1).reshape(3, -1)
                    gyrT = gyr[Start:End].swapaxes(0, 1).reshape(3, -1)
                    magT = mag[Start:End].swapaxes(0, 1).reshape(3, -1)

                    # accT = np.pad(accT, pad_width=((0, 0), (pad, 0)), mode='constant', constant_values=0) # noqa
                    # gyrT = np.pad(gyrT, pad_width=((0, 0), (pad, 0)), mode='constant', constant_values=0) # noqa
                    # magT = np.pad(magT, pad_width=((0, 0), (pad, 0)), mode='constant', constant_values=0) # noqa

                    # pad the acc to window size
                    pad_size = self.window_size - accT.shape[1]
                    if pad_size < 0:
                        continue

                    # check all three are the same as the window size
                    assert (
                        accT.shape[1]
                        == gyrT.shape[1]
                        == magT.shape[1]
                        == self.window_size
                    )

                    # accT[2, :] = accT[2, :] - Gravity
                    # accT /= abs(Gravity)
                    # gyrT /= np.pi
                    # magT[0, :] /= wmm.F * 1e-5  # abs(wmm.X)
                    # magT[1, :] /= wmm.F * 1e-5  # abs(wmm.Y)
                    # magT[2, :] /= wmm.F * 1e-5  # abs(wmm.Z)

                    new_dataT = np.concatenate(
                        (
                            accT,
                            np.linalg.norm(accT, axis=0, keepdims=True),
                            gyrT,
                            np.linalg.norm(gyrT, axis=0, keepdims=True),
                            magT,
                            np.linalg.norm(magT, axis=0, keepdims=True),
                        ),
                        axis=0,
                    )
                    new_dataT = torch.from_numpy(new_dataT).float()
                    new_dataF = self.toFreq(new_dataT)

                    new_data = dict(
                        time=new_dataT,
                        freq=new_dataF,
                    )
                    if self._labels:
                        # NOTE: This is for the remove uneven label step
                        # if (indoor[Start:End] != indoor[Start]).any():
                        #     continue
                        # # majority vote of the indoor label
                        # new_label = np.bincount(indoor[Start:End]).argmax()
                        # # # use average of the indoor label
                        # # new_label = np.mean(indoor[i-self.window_size:i])
                        # new_label = torch.from_numpy(np.array(new_label)).long()

                        # NOTE: This is for the handle uneven label step
                        # count how much zeros and ones respetively within the window
                        zeros = (
                            np.count_nonzero(indoor[Start:End] == 0) / self.window_size
                        )
                        ones = (
                            np.count_nonzero(indoor[Start:End] == 1) / self.window_size
                        )
                        new_label = np.array([zeros, ones], dtype=np.float32)
                        new_label = torch.from_numpy(new_label)

                    else:
                        new_label = torch.tensor([0.0, 0.0])

                    self.data.append((new_data, new_label))
                    if self.mode == "pred":
                        self.time.append(time[End])
            elif self.dice == "dice":
                acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
                acc = np.concatenate((acc, acc_norm), axis=1)
                acc = (
                    acc[: acc.shape[0] // self.window_size * self.window_size]
                    .reshape(-1, self.window_size, 4)
                    .swapaxes(1, 2)
                )

                gyr_norm = np.linalg.norm(gyr, axis=1, keepdims=True)
                gyr = np.concatenate((gyr, gyr_norm), axis=1)
                gyr = (
                    gyr[: gyr.shape[0] // self.window_size * self.window_size]
                    .reshape(-1, self.window_size, 4)
                    .swapaxes(1, 2)
                )

                mag_norm = np.linalg.norm(mag, axis=1, keepdims=True)
                mag = np.concatenate((mag, mag_norm), axis=1)
                mag = (
                    mag[: mag.shape[0] // self.window_size * self.window_size]
                    .reshape(-1, self.window_size, 4)
                    .swapaxes(1, 2)
                )
                if self._labels:
                    indoor = indoor[
                        : indoor.shape[0] // self.window_size * self.window_size
                    ].reshape(-1, self.window_size)

                imu_T = np.concatenate((acc, gyr, mag), axis=1)
                imu_F = self.toFreq(
                    torch.from_numpy(imu_T).float().reshape(-1, self.window_size)
                ).reshape(-1, 12, self.window_size // 2 + 1, self.window_size)

                # TODO:complete the freq transform
                for I, (T, F) in zip(indoor, zip(imu_T, imu_F)):
                    new_data = dict(
                        time=T,
                        freq=F,
                    )
                    if self._labels:
                        new_label = torch.from_numpy(np.array(I)).long()

            else:
                raise ValueError("Invalid dice method")
            # add split point
            if self.mode == "pred":
                self.splitPoint.append(len(self.data))

        if self.dice == "step":
            if self._labels:
                # compute the number of indoor and outdoor
                indoor_count = 0
                outdoor_count = 0
                for _, label in self.data:
                    if label[1] < 0.5:
                        outdoor_count += 1
                    elif label[1] <= 1.0:
                        indoor_count += 1
                    else:
                        print(label)
                        raise ValueError("Invalid label")
                # print(f"indoor : {indoor_count}, outdoor : {outdoor_count}")
                # print(f"Loaded {len(self.data)} samples for {self.mode.upper()}.")
                # prin
                if len(self.data) == 0:
                    print(f"Loaded 0 samples for {self.mode.upper()}.")
                else:
                    print(
                        f"Loaded {len(self.data)} samples for {self.mode.upper()}, with {indoor_count} indoor({indoor_count/len(self.data):.2%}) and {outdoor_count} outdoor({outdoor_count/len(self.data):.2%}) samples."
                    )
            else:
                print(f"Loaded {len(self.data)} samples for {self.mode.upper()}.")

        # if self.mode != 'pred':
        #     # shuffle the data
        #     for _ in range(10):
        #         np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IODDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "datasets",
        batch_size: int = BATCH_SIZE,
        pin_memory: bool = True,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        label: bool = True,
        whole_testing: bool = False,
    ):
        super().__init__()
        # self.data_dir = data_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        if os.environ.get("NODE_RANK") is None:
            self.shuffle = shuffle
        else:
            self.shuffle = False
            print(
                f"{cl.Fore.yellow}Disable shuffle for distributed training.{cl.Style.reset}"
            )
        self.drop_last = drop_last

        # self.save_hyperparameters()
        self.datasets = IndoorDataset(
            self.data_dir,
            mode="train" if not whole_testing else "test",
            label=label,
            whole_testing=whole_testing,
        )
        if whole_testing:
            self.datasets.load_data("test")
        else:
            self.setup("fit")

    # def configure_optimizers(self):
    #     opt = optim.Adam(self.parameters(), lr=self.hparams.lr)

    #     return opt

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.datasets.load_data("train")
            # load the train and val dataset
            train_val = self.datasets

            # # shuffle the data and split into train and val
            # self.IO_train, self.IO_val = torch.utils.data.random_split(
            #     train_val, [0.8, 0.2]
            # )
            # self.IO_train = self.IO_train.dataset

            # # expand
            # self.IO_train = torch.utils.data.ConcatDataset(
            #     [self.IO_train.dataset] * 3)

            # grep fist 10% and last 10% as val
            front = int(len(train_val) * 0.07)
            back = int(len(train_val) * 0.13)
            self.IO_val = train_val[:front] + train_val[-back:]
            # grep the rest as train
            self.IO_train = train_val[front:-back]

            # np.random.shuffle(self.IO_train)
            # np.random.shuffle(self.IO_val)

        elif stage == "test":
            self.datasets.load_data("test")
            self.IO_test = self.datasets

        else:
            raise ValueError("Invalid stage")

    def get_balanced_weight(self, stage: str = "fit"):
        distribution = self.get_distribution(stage)
        distribution = distribution / distribution.sum()  # normalize
        weight = 1 / distribution
        weight = weight / weight.sum()  # normalize

        return weight

    def get_distribution(self, stage: str = "fit"):
        labels = []
        if stage == "train" or stage is None:
            for _, label in self.IO_train:
                labels.append(label)
        elif stage == "val":
            for _, label in self.IO_val:
                labels.append(label)
        elif stage == "test":
            for _, label in self.IO_test:
                labels.append(label)
        else:
            raise ValueError("Invalid stage")

        # distribution = np.bincount(labels)
        distribution = np.sum(labels, axis=0) / np.sum(labels)

        return distribution

    def train_dataloader(self):
        return DataLoader(
            self.IO_train,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.IO_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=False,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.IO_test,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=False,
            drop_last=self.drop_last,
        )

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...


class MotionIDDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # total_member = int(os.environ.get("WORLD_SIZE"))
        # memberid = int(os.environ.get("NODE_RANK"))
        # print(f"Loading dataset for member {memberid} of {total_member}.")

        # """
        # load for 4 node for multi-computer
        # """
        # rank = int(os.environ.get("NODE_RANK"))
        # if rank == 2:
        #     self.datasetPath = "/mnt/secondary/MotionID_01"
        # else:
        #     self.datasetPath = "/mnt/MotionID"

        # self.datasetPath = Path(self.datasetPath)
        # self.files = list(self.datasetPath.glob("*.pt"))
        # self.files.sort()
        # print(f"Loaded {len(self.files)} files.")
        # sample = BATCH_SIZE * 5 * 4
        # # estimate the number of step per epoch
        # step = sample * len(self.files) // (BATCH_SIZE)
        # print(f"Estimated step per epoch : {step}")

        # sample = int(sample / 8 * 10)
        # sample = int(1024 * 1024 // 4)
        # print(f"Use {sample/(1024 * 1024 // 4)*100:.2f}% of datasets")
        # self.length = np.arange(len(self.files+1), dtype=np.int32) * int(sample)
        # self.length = torch.tensor(self.length)

        """
        load for 2 node one computer
        """
        self.datasetPath = "/mnt/Volume02/MotionID_split_withoutscale"
        self.datasetPath = Path(self.datasetPath)
        self.files = list(self.datasetPath.glob("*.pt"))
        self.files.remove(self.datasetPath / "total.pt")
        self.files.sort()
        from pprint import pprint

        print("Load files : ", cl.Fore.green)
        pprint(self.files)
        print(cl.Style.reset)
        print(f"Loaded {len(self.files)} files.")
        self.length = torch.load(self.datasetPath / "total.pt")
        self.length = torch.cat([torch.tensor([0]), self.length])
        self.tranform = torchaudio.transforms.Spectrogram(
            win_length=WINDOW_SIZE,
            n_fft=WINDOW_SIZE,
            hop_length=WINDOW_SIZE,
            # pad_mode='replicate',
            # center=False,
            # onesided=False,
            # power=4,
            # normalized=True,
        )

        self.files = [str(file) for file in self.files]

        self.data_chunks = []
        for file in self.files:
            data = torch.load(file, mmap=True)
            self.data_chunks.append(data)
            del data

    def __len__(self):
        return self.length[-1]

    def __getitem__(self, idx):
        # find the file
        file_idx = torch.searchsorted(self.length, idx, right=True) - 1
        data_idx = idx - self.length[file_idx]
        # target = self.files[file_idx]
        # data = torch.load(target, mmap=True)
        data = self.data_chunks[file_idx]

        x = dict(
            time=data[data_idx],
            freq=self.tranform(data[data_idx]),
        )
        return x, False

    # def __getitem__(self, index):
    #     x = dict(
    #         time=self.time_data[index],
    #         # freq=self.tranform(self.time_data[index]),
    #     )
    #     return x


from torch.utils.data import DistributedSampler


class MotionIDModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        # self.datasetPath = "/mnt/Volume01/MotionID_Processed_tensor"
        self.batch_size = BATCH_SIZE
        self.pin_memory = True
        self.num_workers = int(
            torch.get_num_threads()
            / torch.cuda.device_count()
            # / int(torch.cuda.mem_get_info()[1] / 24892145664)
        )
        self.shuffle = True
        # self.save_hyperparameters()
        self.datasets = MotionIDDataset()
        # check if distributed
        # self.distributed = os.environ.get("WORLD_SIZE") is not None
        self.setup("fit")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            print("Loading train and val dataset.")
            self.train, self.val = torch.utils.data.random_split(
                self.datasets, [0.8, 0.2]
            )
            print(f"Loaded {len(self.train)} train samples.")
            print(f"Loaded {len(self.val)} val samples.")
            print("Completed spliting train and val dataset.")
            # if self.distributed:
            #     self.train_sampler = DistributedSampler(self.train)
            #     self.val_sampler = DistributedSampler(self.val)
            # else:
            #     self.train_sampler = None
            #     self.val_sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train,
            # sampler=self.train_sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=self.shuffle,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            # sampler=self.val_sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=False,
            # drop_last=True,
        )


"""
generate random noise for testing
"""


class NoiseDataset(Dataset):
    def __init__(self, length: int = 10000) -> None:
        super().__init__()
        self.length = length
        self.data = torch.randn((length, 12, WINDOW_SIZE))
        self.tranform = torchaudio.transforms.Spectrogram(
            win_length=WINDOW_SIZE,
            n_fft=WINDOW_SIZE,
            hop_length=WINDOW_SIZE,
            # pad_mode='replicate',
            # center=False,
            # onesided=False,
            # power=4,
            # normalized=True,
        )
        label = torch.rand((length, 1))
        self.label = torch.cat([label, 1 - label], dim=-1)
        # freq = torchaudio.transforms.Spectrogram(
        #     win_length=WINDOW_SIZE,
        #     n_fft=WINDOW_SIZE,
        #     hop_length=WINDOW_SIZE,
        #     # pad_mode='replicate',
        #     # center=False,
        #     # onesided=False,
        #     # power=4,
        #     # normalized=True,
        # )(time)
        # self.data = []
        # for i in range(length):
        #     self.data.append(dict(time=time[i], freq=freq[i]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        time = self.data[idx]
        freq = self.tranform(time)

        return (dict(time=time, freq=freq), self.label[idx])


class NoiseModule(pl.LightningDataModule):
    def __init__(self, length: int = 10000) -> None:
        super().__init__()
        self.batch_size = BATCH_SIZE
        self.pin_memory = True
        self.num_workers = int(
            torch.get_num_threads()
            / torch.cuda.device_count()
            # / int(torch.cuda.mem_get_info()[1] / 24892145664)
        )
        self.shuffle = True
        # self.save_hyperparameters()
        self.datasets = NoiseDataset(length=length)
        # check if distributed
        # self.distributed = os.environ.get("WORLD_SIZE") is not None
        self.setup("test")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            print("Loading train and val dataset.")
            self.train, self.val = torch.utils.data.random_split(
                self.datasets, [0.8, 0.2]
            )
            print(f"Loaded {len(self.train)} train samples.")
            print(f"Loaded {len(self.val)} val samples.")
            print("Completed spliting train and val dataset.")

        elif stage == "test":
            print("Loading test dataset.")
            self.test = self.datasets
            print(f"Loaded {len(self.test)} test samples.")
            print("Completed loading test dataset.")
        else:
            raise ValueError("Invalid stage")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            # sampler=self.train_sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=self.shuffle,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            # sampler=self.val_sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            # sampler=self.val_sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=False,
            drop_last=True,
        )


class multiDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # self.datasets1 = MotionIDDataset()
        self.datasets1 = IndoorDataset(
            "/tmp/trainer/spectrums/migration_wm/",
            mode="train",
            label=True,
            whole_testing=False,
            # mode="fit",
        )

        self.datasets2 = IndoorDataset(
            "/mnt/RogueOne/Documents/verilog/ASIC/PDR/spectrums/nolabeled_wm/",
            mode="train",
            label=False,
            whole_testing=False,
        )

        # min_len = min(len(self.datasets1), len(self.datasets2))
        # # randomly keep only min_len samples
        # self.datasets1 = torch.utils.data.Subset(
        #     self.datasets1, torch.randperm(len(self.datasets1))[:min_len]
        # )
        # self.datasets2 = torch.utils.data.Subset(
        #     self.datasets2, torch.randperm(len(self.datasets2))[:min_len]
        # )

        # min_len = min(len(self.datasets1), len(self.datasets2) // 4)
        # node = os.environ.get("NODE_RANK")
        # self.datasets2 = torch.utils.data.Subset(
        #     self.datasets2,
        #     torch.arange(
        #         len(self.datasets2) // 4 * int(node),
        #         len(self.datasets2) // 4 * (int(node) + 1),
        #     ),
        # )

        # self.datasets1 = torch.utils.data.Subset(
        #     self.datasets1, torch.randperm(len(self.datasets1))[: min_len * 100]
        # )
        self.merge = torch.utils.data.ConcatDataset([self.datasets1, self.datasets2])
        print(
            f"{cl.Fore.yellow}MultiDataset Ratio: {len(self.datasets1)/len(self.merge)*100:.2f}% {len(self.datasets2)/len(self.merge)*100:.2f}%{cl.Style.reset}"
        )

    def __len__(self):
        return len(self.merge)

    def __getitem__(self, idx):
        return self.merge[idx]


class multiDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.batch_size = BATCH_SIZE
        self.pin_memory = True
        self.num_workers = int(
            torch.get_num_threads()
            / torch.cuda.device_count()
            # / int(torch.cuda.mem_get_info()[1] / 24892145664)
        )
        self.shuffle = True
        # self.save_hyperparameters()

        self.datasets = multiDataset()

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.datasets, [0.8, 0.2]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=self.shuffle,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=int(self.num_workers),
            shuffle=False,
            drop_last=True,
        )


# class Augment:
#     """
#     old version, very slow, deprecated when switch to generative datasets generation method
#     """

#     def __init__(self, upstreamtask: bool = False, mode: str = "classify") -> None:
#         self._mode = mode
#         self._upstreamtask = upstreamtask
#         if mode == "classify":
#             if self._upstreamtask:
#                 self._flip = 0.5
#                 self._yredirect = 0.0
#                 self._channel_permute = 0.0

#                 self._scale = 1.0
#                 self._shift = 1.0
#                 self._gaussian = 1.0
#                 # self._gaussian_std = 0.01

#                 self._rotation = 1.0

#                 self._base = (1 / 3) * 0.9
#                 self._channel = 3
#                 self._mask = 0.9

#             else:
#                 self._flip = 0.5
#                 self._yredirect = 0.5
#                 self._channel_permute = 0.5

#                 self._scale = 0.0
#                 self._shift = 0.0
#                 self._gaussian = 0.0
#                 # self._gaussian_std = 0.05

#                 self._rotation = 1.0

#                 self._base = (1 / 3) * 0.9
#                 self._channel = 3
#                 self._mask = 0.6

#         elif mode == "generate":
#             self._flip = 0.5
#             self._yredirect = 0.0
#             self._channel_permute = 0.0

#             self._scale = 1.0
#             self._shift = 1.0
#             self._gaussian = 0.5
#             # self._gaussian_std = 0.05

#             self._rotation = 1.0

#             self._base = 0.0
#             self._channel = 0
#             self._mask = 0.0
#         else:
#             raise ValueError("Invalid mode for Augment")

#         self.timeMask = torchaudio.transforms.TimeMasking(
#             time_mask_param=int(self._mask * WINDOW_SIZE),
#             # time_mask_param=3
#             # iid_masks=True,
#             p=self._mask,
#         )
#         self.freqMask = torchaudio.transforms.FrequencyMasking(
#             freq_mask_param=int(self._mask * (WINDOW_SIZE // 2 + 1)),
#             # iid_masks=True,
#         )

#     def rotateMatrixGenerator(
#         self,
#         mode: str = None,
#     ) -> torch.Tensor:
#         # yaw = yaw * np.pi / 180
#         # pitch = pitch * np.pi / 180
#         # roll = roll * np.pi / 180

#         # mode = "rotate_axis"
#         # mode = "wobbling"
#         # mode = "random"
#         if mode is None:
#             if self._upstreamtask:
#                 mode = "wobbling"
#             else:
#                 mode = "wobbling"

#         # if self._upstreamtask:
#         #     percentage = torch.rand(1).item()
#         #     if percentage < 0.8:
#         #         mode = "wobbling"
#         #     else:
#         #         mode = "rotate_axis"
#         # else:
#         #     mode = "rotate_axis"

#         if mode == "rotate_axis":
#             yaw, pitch, roll = torch.randint(4, (3,)) * 90 / 180 * np.pi

#         elif mode == "wobbling":
#             degree = 6
#             yaw, pitch, roll = (torch.rand((3,)) * 2 - 1) * degree / 180 * np.pi

#         elif mode == "random":
#             yaw, pitch, roll = torch.rand((3,)) * 2 * np.pi

#         else:
#             raise ValueError(
#                 "Data Augementation: Invalid mode for rotateMatrixGenerator"
#             )

#         # Calculate the rotation matrix
#         rotation_x = torch.tensor(
#             [
#                 [1, 0, 0],
#                 [0, np.cos(roll), -np.sin(roll)],
#                 [0, np.sin(roll), np.cos(roll)],
#             ]
#         )
#         rotation_y = torch.tensor(
#             [
#                 [np.cos(pitch), 0, np.sin(pitch)],
#                 [0, 1, 0],
#                 [-np.sin(pitch), 0, np.cos(pitch)],
#             ]
#         )
#         rotation_z = torch.tensor(
#             [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
#         )
#         rotation_matrix = torch.matmul(torch.matmul(rotation_z, rotation_y), rotation_x)

#         return rotation_matrix.float().T

#     def iphone_sim(self, x_time):
#         mag = x_time[8:]
#         mag = mag[:3]

#         percentage = torch.rand(1).item()
#         mag_x = mag[0]
#         mag_z = mag[2]

#         x_time[8] = torch.linalg.norm(
#             torch.stack([mag_x, mag_z * percentage], dim=-1), dim=-1
#         )
#         x_time[10] = mag_z * (1 - percentage)

#         return x_time

#     def bach_transform(self, x, masking=True):
#         # convert the dict of tenwor to tensor of dict
#         time = copy.deepcopy(x["time"])
#         freq = torch.zeros(
#             time.shape[0], 12, WINDOW_SIZE // 2 + 1, 2, device=time.device
#         )
#         for i in range(time.shape[0]):
#             # time[i] = self.iphone_sim(time[i])
#             time[i] = self.train_transform_time(time[i])

#             # 5 out of 6 chance to flip xyz axis
#             percentage = torch.rand(1).item()
#             if percentage < self._rotation:
#                 matrix = self.rotateMatrixGenerator()
#                 matrix = matrix.to(time.device)
#                 time[i][:3] = torch.matmul(
#                     time[i][:3].swapaxes(0, -1), matrix
#                 ).swapaxes(0, -1)
#                 time[i][4:7] = torch.matmul(
#                     time[i][4:7].swapaxes(0, -1), matrix
#                 ).swapaxes(0, -1)
#                 time[i][8:11] = torch.matmul(
#                     time[i][8:11].swapaxes(0, -1), matrix
#                 ).swapaxes(0, -1)

#             freq[i] = self.train_transform_freq(time[i])

#             if masking:
#                 percentage = torch.rand(1).item()

#                 # random masking time
#                 if percentage < (self._base * 1):
#                     time[i] = self.timeMask(time[i])

#                 # random masking freq
#                 elif percentage < (self._base * 2):
#                     freq[i] = self.freqMask(freq[i])

#                 # elif percentage < 0.75:
#                 #     # random masking time
#                 #     time[i] = torchaudio.transforms.TimeMasking(
#                 #         time_mask_param=99,
#                 #         # iid_masks=True,
#                 #     )(time[i])
#                 #     freq[i] = (
#                 #         torch.stft(
#                 #             time[i],
#                 #             n_fft=WINDOW_SIZE,
#                 #             hop_length=WINDOW_SIZE,
#                 #             win_length=WINDOW_SIZE,
#                 #             window=torch.hann_window(WINDOW_SIZE).cuda(),
#                 #             return_complex=False,
#                 #         )
#                 #         .pow(2)
#                 #         .sum(-1)
#                 #     )

#                 # random masking channel
#                 elif percentage < (self._base * 3):
#                     if self._channel != 0:
#                         for sensor in range(0, 12, 4):
#                             num = torch.randint(1, self._channel + 1, (1,)).item()
#                             channel = torch.randint(0, 4, (num,)) + sensor

#                             # channel = torch.randint(0, 12, (1,)).item()
#                             time[i, channel] = 0
#                             freq[i, channel] = 0
#                 else:
#                     pass

#         return dict(time=time, freq=freq)

#     def train_transform_time(self, x):
#         acc = x[:3, :]
#         gyr = x[4:7, :]
#         mag = x[9:, :]

#         # flip
#         percentage = torch.rand(1).item()
#         if percentage < self._flip:
#             acc = torch.flip(acc, [1])
#             gyr = torch.flip(gyr, [1])
#             mag = torch.flip(mag, [1])

#         percentage = torch.rand(1).item()
#         if percentage < self._yredirect:
#             mag[1] = -mag[1]

#         percentage = torch.rand(1).item()
#         if percentage < self._channel_permute:
#             order = torch.randperm(3)
#             acc = acc[order]
#             gyr = gyr[order]
#             mag = mag[order]

#         # scale
#         percentage = torch.rand(1).item()
#         if percentage < self._scale:
#             for i in range(3):
#                 acc[i] = acc[i] * (torch.rand(1).item() * 0.2 + 0.9)  # noqa
#                 gyr[i] = gyr[i] * (torch.rand(1).item() * 0.2 + 0.9)  # noqa
#                 mag[i] = mag[i] * (torch.rand(1).item() * 0.2 + 0.9)  # noqa

#         # shift
#         percentage = torch.rand(1).item()
#         if percentage < self._shift:
#             for i in range(3):
#                 acc[i] = acc[i] + torch.rand(1).item() * 0.1 - 0.05  # noqa
#                 gyr[i] = gyr[i] + torch.rand(1).item() * 0.05 - 0.025  # noqa
#                 mag[i] = mag[i] + torch.rand(1).item() * 1e-4 - 1e-4 / 2  # noqa

#         # gaussian noise
#         percentage = torch.rand(1).item()
#         if percentage < self._gaussian:
#             std_list = (
#                 0.026033755765741105 / 9.8 / 2,
#                 0.004937662039113619 / np.pi / 2,
#                 0.006771730824538068 / 0.5 / 2,
#             )
#             acc = torch.normal(acc, std=std_list[0]).to(acc.device)
#             gyr = torch.normal(gyr, std=std_list[1]).to(gyr.device)
#             mag = torch.normal(mag, std=std_list[2]).to(mag.device)

#             # acc = torch.normal(acc, std=self._gaussian_std).to(acc.device)
#             # gyr = torch.normal(gyr, std=self._gaussian_std).to(gyr.device)
#             # mag = torch.normal(mag, std=self._gaussian_std).to(mag.device)
#         # percentage = torch.rand(1).item()
#         # if percentage < 0.5:
#         #     # cutoff = torch.randint(0, 3, (1,), device=x.device).item()
#         #     cutoff = torch.rand(1).item() * 2
#         #     acc = torchaudio.functional.highpass_biquad(
#         #         acc, sample_rate=100, cutoff_freq=cutoff
#         #     ).to(acc.device)
#         #     gyr = torchaudio.functional.highpass_biquad(
#         #         gyr, sample_rate=100, cutoff_freq=cutoff
#         #     ).to(gyr.device)

#         # percentage = torch.rand(1).item()
#         # if percentage < 0.5:
#         #     cutoff = torch.rand(1).item() * 30 + 30
#         #     acc = torchaudio.functional.lowpass_biquad(
#         #         acc, sample_rate=100, cutoff_freq=cutoff
#         #     ).to(acc.device)

#         #     cutoff = torch.rand(1).item() * 30 + 30
#         #     gyr = torchaudio.functional.lowpass_biquad(
#         #         gyr, sample_rate=100, cutoff_freq=cutoff
#         #     ).to(gyr.device)

#         #     cutoff = torch.rand(1).item() * 30 + 30
#         #     mag = torchaudio.functional.lowpass_biquad(
#         #         mag, sample_rate=100, cutoff_freq=cutoff
#         #     ).to(mag.device)

#         accNorm = torch.linalg.norm(acc, dim=0, keepdim=True)
#         gyrNorm = torch.linalg.norm(gyr, dim=0, keepdim=True)
#         magNorm = torch.linalg.norm(mag, dim=0, keepdim=True)

#         # stack the norm to the end of the tensor
#         time = torch.concatenate(
#             [
#                 acc,
#                 accNorm,
#                 gyr,
#                 gyrNorm,
#                 mag,
#                 magNorm,
#             ],
#             dim=0,
#         )
#         return time

#     def train_transform_freq(self, x):
#         freq = torch.stft(
#             x,
#             n_fft=WINDOW_SIZE,
#             hop_length=WINDOW_SIZE,
#             win_length=WINDOW_SIZE,
#             window=torch.hann_window(WINDOW_SIZE).cuda(),
#             return_complex=True,
#         )
#         # make complex to real
#         freq = torch.stack([freq.real, freq.imag], dim=-1)
#         freq = freq.pow(2).sum(-1)

#         # # percentage = torch.rand(1).item()
#         # # if percentage < 1.0:
#         # freq = torchaudio.transforms.FrequencyMasking(
#         #     freq_mask_param=30,
#         #     iid_masks=True,
#         # )(freq)
#         return freq

#     def __call__(self, x, masking=True, pair=False):
#         for i in range(len(x)):
#             matrix = self.rotateMatrixGenerator(mode="rotate_axis")
#             time = x["time"][i]

#             matrix = matrix.to(time.device)
#             time[i][:3] = torch.matmul(time[i][:3].swapaxes(0, -1), matrix).swapaxes(
#                 0, -1
#             )
#             time[i][4:7] = torch.matmul(time[i][4:7].swapaxes(0, -1), matrix).swapaxes(
#                 0, -1
#             )
#             time[i][8:11] = torch.matmul(
#                 time[i][8:11].swapaxes(0, -1), matrix
#             ).swapaxes(0, -1)
#             x["time"][i] = time

#         if pair:
#             # return self.bach_transform(x, masking), self.bach_transform(x, masking)
#             return self.bach_transform(x, False), self.bach_transform(x, masking)
#         else:
#             return self.bach_transform(x, masking)
