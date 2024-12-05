import copy
import datetime
import glob
import io
import os
import sys
from pathlib import Path
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchaudio.models import Conformer
import colored as cl
import torchaudio

WINDOW_SIZE = 100


class Augmentation:
    def __init__(
        self,
        upstreamtask=True,
        mode="classification",
    ):
        if mode not in ["classification", "generate", "regression"]:
            raise ValueError(
                "mode should be one of ['classification', 'generate','regression']"
            )

        self._upstream = upstreamtask
        self._mode = mode

        if mode == "classification":
            if upstreamtask:
                self._ymag_reverse_percentage = 0.0
                self._y_reverse_percentage = 0.0
                self._channel_permute_percentage = 0.0

                self._flip_percentage = 0.5
                self._scale_percentage = 1.0
                self._shift_percentage = 1.0
                self._gaussian_noise_percentage = 1.0

                self._rotate_percentage = 1.0

                self._base = (1 / 3) * 0.9
                self._time_masking_percentage = 1.0
                self._frequency_masking_percentage = 1.0
                self._channel_masking_percentage = 1.0
                self._mask = 0.9
                self._num_channel = 3
            else:
                """
                here is the default setting for downstream task
                """
                self._ymag_reverse_percentage = 0.5
                self._y_reverse_percentage = 0.0
                self._channel_permute_percentage = 0.0

                self._flip_percentage = 0.5
                self._scale_percentage = 1.0
                self._shift_percentage = 1.0
                self._gaussian_noise_percentage = 1.0
                self._rotate_percentage = 1.0

                self._base = (1 / 3) * 3 / 4
                self._time_masking_percentage = 1.0
                self._frequency_masking_percentage = 1.0
                self._channel_masking_percentage = 1.0
                self._mask = 0.9
                self._num_channel = 3

        elif mode == "generate":
            if upstreamtask:
                self._ymag_reverse_percentage = 0.0
                self._y_reverse_percentage = 0.0
                self._channel_permute_percentage = 0.0

                self._flip_percentage = 0.0
                self._scale_percentage = 1.0
                self._shift_percentage = 1.0
                self._gaussian_noise_percentage = 1.0

                self._rotate_percentage = 0.0

                self._base = (1 / 3) * 0.9
                self._time_masking_percentage = 0.0
                self._frequency_masking_percentage = 0.0
                self._channel_masking_percentage = 0.0
                self._mask = 0.0
                self._num_channel = 0
            else:
                raise ValueError(
                    "In augmentation, under mode 'generate', upstreamtask should be True."
                )
        elif mode == "regression":
            if upstreamtask:
                self._ymag_reverse_percentage = 0.0
                self._y_reverse_percentage = 0.0
                self._channel_permute_percentage = 0.0

                self._flip_percentage = 0.5
                self._scale_percentage = 1.0
                self._shift_percentage = 1.0
                self._gaussian_noise_percentage = 1.0

                self._rotate_percentage = 1.0

                self._base = (1 / 3) * 0.9
                self._time_masking_percentage = 1.0
                self._frequency_masking_percentage = 1.0
                self._channel_masking_percentage = 1.0
                self._mask = 0.9
                self._num_channel = 3
            else:
                """
                here is the default setting for downstream task
                """
                self._ymag_reverse_percentage = 0.0
                self._y_reverse_percentage = 0.0
                self._channel_permute_percentage = 0.0

                self._flip_percentage = 0.5
                self._scale_percentage = 1.0
                self._shift_percentage = 1.0
                self._gaussian_noise_percentage = 1.0
                self._rotate_percentage = 1.0

                self._base = (1 / 3) * 3 / 4
                self._time_masking_percentage = 0.0
                self._frequency_masking_percentage = 0.0
                self._channel_masking_percentage = 0.0
                self._mask = 0.0
                self._num_channel = 0
        else:
            raise ValueError(
                "mode should be one of ['classification', 'generate', 'regression']"
            )

    def __call__(self, data, pair=False, label=None):
        return self.forward(data, pair, label)

    def forward(self, data, pair, label=None):
        time = data["time"].cuda()
        if label is not None:
            original_device = label.device
            label = torch.clone(label).to(original_device)
        bach_size = time.shape[0]
        # freq = data["freq"].cuda()

        # time domain process
        time = self.twelve2Nine(time)
        time = self.ymag_reverse(time)
        time = self.channel_permute(time)

        # # rotate
        if label is not None:
            time, label = self.rotate(time, mode="random_only_roll", label=label)
        else:
            time = self.rotate(time, mode="random")

        # other
        if label is not None:
            time1, label = self.flip(time, label=label)
            # time1, label = self.reverse(data=time1, percentage=0.0, label=label)
        else:
            time1 = self.flip(time)
            # time1 = self.reverse(time1)
        time1 = self.scale(time1)
        time1 = self.shift(time1)
        time1 = self.gaussianNoise(time1)
        time1 = self.rotate(time1, mode="wobble")
        # time1 = self.accgyr_reverse(time1, label=label)
        time1 = self.nine2Twelve(time1)

        freq1 = self.genFreq(time1)
        # time1 = self.lowpass(time1)

        percentage = torch.rand(bach_size)

        filter_time = percentage < self._base
        filter_freq = torch.logical_and(
            percentage < self._base * 2, percentage >= self._base
        )
        filter_channel = torch.logical_and(
            percentage < self._base * 3, percentage >= self._base * 2
        )
        del percentage

        time1 = self.timeMasking(time1, filter_time)
        freq1 = self.frequencyMasking(freq1, filter_freq)
        time1, freq1 = self.channelMasking(time1, freq1, filter_channel)

        if not pair:
            if label is not None:
                return {"time": time1, "freq": freq1}, label
            return {"time": time1, "freq": freq1}

        # other
        time2 = self.flip(time)
        time2 = self.scale(time2)
        time2 = self.shift(time2)
        time2 = self.gaussianNoise(time2)
        time2 = self.rotate(time2, mode="wobble")
        time2 = self.nine2Twelve(time2)

        freq2 = self.genFreq(time2)
        # time2 = self.lowpass(time2)

        return {"time": time1, "freq": freq1}, {"time": time2, "freq": freq2}

    def lowpass(self, data, cutoff_hz=30):
        size = data.shape[0]

        data = data.reshape(size * 12, 100)
        data = torchaudio.functional.lowpass_biquad(
            data, sample_rate=100, cutoff_freq=cutoff_hz, Q=0.707
        )
        data = data.reshape(size, 12, 100)

        return data

    def nine2Twelve(self, data):
        size = data.shape[0]

        acc = data[:, :3, :]
        gyr = data[:, 3:6, :]
        mag = data[:, 6:9, :]

        acc_norm = torch.norm(acc, dim=1, keepdim=True)
        gyr_norm = torch.norm(gyr, dim=1, keepdim=True)
        mag_norm = torch.norm(mag, dim=1, keepdim=True)

        new_data = torch.cat([acc, acc_norm, gyr, gyr_norm, mag, mag_norm], dim=1)

        return new_data

    def twelve2Nine(self, data):
        size = data.shape[0]

        acc = data[:, :3, :]
        gyr = data[:, 4:7, :]
        mag = data[:, 8:11, :]

        new_data = torch.cat([acc, gyr, mag], dim=1)

        return new_data

    def ymag_reverse(self, data: torch.tensor, percentage: float = None):
        if percentage is None:
            percentage = self._ymag_reverse_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage

        data[mask][7] = -data[mask][7]
        return data

    def accgyr_reverse(self, data: torch.tensor, percentage: float = None, label=None):
        if percentage is None:
            percentage = self._y_reverse_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        # reverse x-axis
        for i in range(3):
            probability = torch.rand(size)
            mask = probability < percentage
            data[mask][i] = -data[mask][i]
            # data[mask][i + 3] = -data[mask][i + 3]
            label[mask][:, i] = -label[mask][:, i]

        return data

    # def gyr_reverse(self, data: torch.tensor, percentage: float = None, label=None):

    def channel_permute(self, data, percentage: float = None):
        if percentage is None:
            percentage = self._channel_permute_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage

        # TODO
        device = data.device
        order = torch.stack([torch.randperm(3) for _ in range(mask.sum(0))]).to(device)
        length = order.shape[0]
        data[mask] = torch.gather(
            data[mask].reshape(length, 3, 3, 100),
            2,
            order.unsqueeze(1).unsqueeze(-1).repeat(1, 3, 1, 100),
        ).reshape(length, 9, 100)

        return data

    def reverse(self, data: torch.tensor, percentage: float = None, label=None):
        if percentage is None:
            percentage = self._reverse_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage

        data[mask] = -data[mask]
        if label is not None:
            label[mask] = -label[mask]
            return data, label

        return data

    def flip(self, data: torch.tensor, percentage: float = None, label=None):
        if percentage is None:
            percentage = self._flip_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            if label is not None:
                return data, label
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage

        data[mask] = torch.flip(data[mask], dims=[-1])
        if label is not None:
            label[mask] = -label[mask]
            data[mask] = -data[mask]
            return data, label

        return data

    def scale(self, data, percentage: float = None):
        if percentage is None:
            percentage = self._scale_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage

        # TODO
        data[mask, :3, :] = data[mask, :3, :] * (
            torch.rand((data[mask].shape[0], 3, 1), device=data.device) * 0.2 + 0.9
        )

        data[mask, 3:-3, :] = data[mask, 3:-3, :] * (
            torch.rand((data[mask].shape[0], 3, 1), device=data.device) * 0.2 + 0.9
        )

        data[mask, -3:, :] = data[mask, -3:, :] * (
            torch.rand((data[mask].shape[0], 3, 1), device=data.device) * 0.2 + 0.9
        )

        return data

    def shift(self, data, percentage: float = None):
        if percentage is None:
            percentage = self._shift_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage

        # TODO
        data[mask, :3, :] = data[mask, :3, :] + (
            torch.rand((data[mask].shape[0], 3, 1), device=data.device) * 0.1 - 0.05
        )

        data[mask, 3:-3, :] = data[mask, 3:-3, :] + (
            torch.rand((data[mask].shape[0], 3, 1), device=data.device) * 0.05 - 0.025
        )

        data[mask, -3:, :] = data[mask, -3:, :] + (
            torch.rand((data[mask].shape[0], 3, 1), device=data.device) * 0.2 - 0.1
        )

        return data

    def gaussianNoise(self, data, percentage: float = None):
        if percentage is None:
            percentage = self._gaussian_noise_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage
        std_list = [
            0.026033755765741105 / 2,  # / 9.8 / 2,
            0.004937662039113619 / 2,  # / np.pi / 2,
            0.006771730824538068 / 0.5 / 2,
        ]

        # TODO
        data[mask, :3, :] = torch.normal(data[mask, :3, :], std=std_list[0])
        data[mask, 3:-3, :] = torch.normal(data[mask, 3:-3, :], std=std_list[1])
        data[mask, -3:, :] = torch.normal(data[mask, -3:, :], std=std_list[2])
        return data

    def genRotaionMatrix(self, mode, size: int):
        if mode == "random":
            roll, pitch, yaw = torch.rand(3, size) * 2 * torch.pi
            roll[:] = 0.0
            pitch[:] = 0.0
        elif mode == "axis":
            roll, pitch, yaw = (
                torch.randint(low=0, high=4, size=(3, size)).float() * torch.pi / 2
            )
            roll[:] = 0.0
            pitch[:] = 0.0
        elif mode == "wobble":
            roll, pitch, yaw = torch.rand(3, size) * 6 / 180 * torch.pi
        elif mode == "random_only_roll":
            # roll, pitch, yaw = torch.rand(3, size) * 2 * torch.pi
            roll, pitch, yaw = (torch.rand(3, size) - 0.5) * 2 * torch.pi * 20 / 180
            roll[:] = 0.0
            pitch[:] = 0.0
        elif mode == "axis_only_roll":
            roll, pitch, yaw = (
                torch.randint(low=0, high=4, size=(3, size)).float() * torch.pi / 2
            )
            roll[:] = 0.0
            pitch[:] = 0.0
        else:
            raise ValueError("mode should be one of ['random', 'axis', 'wobble']")
        Rx = torch.zeros((size, 3, 3)).cuda()
        # x-axis
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = torch.cos(roll)
        Rx[:, 1, 2] = -torch.sin(roll)
        Rx[:, 2, 1] = torch.sin(roll)
        Rx[:, 2, 2] = torch.cos(roll)

        # y-axis
        Ry = torch.zeros((size, 3, 3)).cuda()
        Ry[:, 1, 1] = 1
        Ry[:, 0, 0] = torch.cos(pitch)
        Ry[:, 0, 2] = torch.sin(pitch)
        Ry[:, 2, 0] = -torch.sin(pitch)
        Ry[:, 2, 2] = torch.cos(pitch)

        # z-axis
        Rz = torch.zeros((size, 3, 3)).cuda()
        Rz[:, 2, 2] = 1
        Rz[:, 0, 0] = torch.cos(yaw)
        Rz[:, 0, 1] = -torch.sin(yaw)
        Rz[:, 1, 0] = torch.sin(yaw)
        Rz[:, 1, 1] = torch.cos(yaw)

        matrix = torch.matmul(torch.matmul(Rx, Ry), Rz)
        return matrix

    def rotate(self, data, percentage: float = None, mode: str = "random", label=None):
        if percentage is None:
            percentage = self._rotate_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data.shape[0]

        probability = torch.rand(size)

        mask = probability < percentage

        matrix = self.genRotaionMatrix(mode=mode, size=mask.sum(0))

        # TODO
        data[mask] = (
            torch.matmul(
                data[mask].reshape((-1, 3, 3, WINDOW_SIZE)).permute(0, 3, 1, 2),
                matrix.reshape(-1, 1, 3, 3),
            )
            .permute(0, 2, 3, 1)
            .reshape((-1, 9, WINDOW_SIZE))
        )
        if label is not None:
            label[mask] = (
                torch.matmul(
                    label[mask].reshape(-1, 1, 3) * 1000,
                    matrix.reshape(-1, 3, 3),
                ).reshape(-1, 3)
                / 1000
            )
            return data, label

        return data

    def genFreq(self, data, num_channel: int = 12):
        batch_size = data.shape[0]
        freq = torch.stft(
            data.reshape(-1, WINDOW_SIZE),
            n_fft=WINDOW_SIZE,
            hop_length=WINDOW_SIZE,
            win_length=WINDOW_SIZE,
            window=torch.hann_window(WINDOW_SIZE).to(data.device),
            return_complex=True,
        )
        # make complex to real
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        freq = freq.pow(2).sum(-1)
        freq = freq.reshape(batch_size, num_channel, WINDOW_SIZE // 2 + 1, 2)
        return freq

    def timeMasking(self, data, filter: torch.tensor, percentage: float = None):
        if percentage is None:
            percentage = self._time_masking_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data[filter].shape[0]

        probability = torch.rand(size)
        mask = probability < percentage

        timemask = torchaudio.transforms.TimeMasking(
            time_mask_param=int(self._mask * WINDOW_SIZE),
            # time_mask_param=3
            # iid_masks=True,
            p=self._mask,
        )

        data[filter][mask] = timemask(data[filter][mask])
        # data[mask] = data[mask] + torch.rand(data[mask].shape[0])
        return data

    def frequencyMasking(self, data, filter: torch.tensor, percentage: float = None):
        if percentage is None:
            percentage = self._frequency_masking_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return data

        size = data[filter].shape[0]

        probability = torch.rand(size)
        mask = probability < percentage

        freqmask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=int(self._mask * (WINDOW_SIZE // 2 + 1)),
        )

        data[filter][mask] = freqmask(data[filter][mask])
        return data

    def channelMasking(
        self, time, freq, filter: torch.tensor, percentage: float = None
    ):
        if percentage is None:
            percentage = self._channel_masking_percentage

        # check percentage is number
        assert isinstance(percentage, (int, float))

        if percentage == 0:
            return time, freq

        size = time[filter].shape[0]

        probability = torch.rand(size)
        mask = probability < percentage
        indices = (
            torch.arange(mask.sum(0))
            .reshape(-1, 1)
            .expand(-1, self._num_channel)
            .reshape(-1)
        )
        order = torch.randint(0, 4, (mask.sum(0), self._num_channel)).reshape(-1)

        # time
        temp = time[filter][mask]
        temp = temp.reshape(-1, 3, 4, WINDOW_SIZE)
        temp[indices, :, order] = 0
        temp = temp.reshape(-1, 12, WINDOW_SIZE)
        time[filter][mask] = temp

        # freq
        temp = freq[filter][mask]
        temp = temp.reshape(-1, 3, 4, WINDOW_SIZE // 2 + 1, 2)
        temp[indices, :, order] = 0
        temp = temp.reshape(-1, 12, WINDOW_SIZE // 2 + 1, 2)
        freq[filter][mask] = temp

        return time, freq
