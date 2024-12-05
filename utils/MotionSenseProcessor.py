import glob
import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Union

import colored as cl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils
import torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ACTIONS = {
    "dws": 0,
    "jog": 1,
    "sit": 2,
    "std": 3,
    "ups": 4,
    "wlk": 5,
}


class MotionSenseProcessor:
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)

    def list_files(self, path: Path = None):
        if path is None:
            path = self.data_path
        files = list(path.glob("**/*.csv"))
        return files

    def load_data(self, file_name: Path):
        df = pd.read_csv(file_name)
        df["index"] = df.index
        df.drop(columns=["Unnamed: 0"], inplace=True)
        df = self.resample(df, "10ms")

        assert not df.isnull().values.any(), "DataFrame contains null values 2"

        labels = file_name.parent.name[:3]
        labels = ACTIONS[labels]
        return df, labels

    def process_data(self):
        files = self.list_files()
        total = []
        for file in tqdm(files):
            df, labels = self.load_data(file)
            acc, gyr = self.grepColumns(df)
            total.extend(self.convert2SingleArray(acc, gyr, labels))

        return total

    def grepColumns(self, df):
        acc = (
            df[
                ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]
            ].to_numpy()
            * 9.8
        )[1000:-1000]
        gyr = df[["rotationRate.x", "rotationRate.y", "rotationRate.z"]].to_numpy()[
            1000:-1000
        ]
        assert np.isnan(acc).sum() == 0, "acc contains nan value"
        assert np.isnan(gyr).sum() == 0, "gyr contains nan value"
        acc = np.concatenate([acc, np.linalg.norm(acc, axis=-1, keepdims=True)], axis=1)
        gyr = np.concatenate([gyr, np.linalg.norm(gyr, axis=-1, keepdims=True)], axis=1)
        return acc, gyr

    def convert2SingleArray(self, acc, gyr, label):
        assert len(acc) == len(gyr), "Length of acc and gyr should be the same"
        assert np.isnan(acc).sum() == 0, "acc contains nan value"
        assert np.isnan(gyr).sum() == 0, "gyr contains nan value"
        stride = 100
        window_size = 100
        total = []
        for i in range(0, len(acc) - window_size, stride):
            acc_window = acc[i : i + window_size]
            gyr_window = gyr[i : i + window_size]
            label_window = label
            new_window = np.concatenate((acc_window, gyr_window), axis=1).swapaxes(0, 1)
            total.append((new_window, label_window))
        return total

    def resample(self, df, freq):
        # assert if nan value exist
        assert not df.isnull().values.any(), "DataFrame contains null values"
        # add timestamp column base on index with sample rate 50hz
        df["timestamp"] = pd.date_range(start="1/1/2000", periods=len(df), freq="20ms")
        # resample from 50hz to 100hz
        df.set_index("timestamp", inplace=True)
        assert not df.isnull().values.any(), "DataFrame contains null values 1"
        df = df.resample(freq).mean()
        df.ffill(inplace=True)

        return df


class MotionSneseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.processor = MotionSenseProcessor(data_path)
        self.data = self.processor.process_data()
        self.transform = transform
        self.toFreq = torchaudio.transforms.Spectrogram(
            win_length=100,
            n_fft=100,
            hop_length=100,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        x, y = sample
        x = torch.from_numpy(x).float()
        return (
            dict(
                time=x,
                freq=self.toFreq(x),
            ),
            y,
        )


class MotionSenseDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset = MotionSneseDataset(data_path)
        ratio = [0.8, 0.1]
        ratio = [int(len(self.dataset) * r) for r in ratio]
        ratio.append(len(self.dataset) - sum(ratio))
        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(
                self.dataset,
                ratio,
            )
        )

        # labels = [y for _, y in self.dataset]
        # # Split the dataset into train, validation, and test sets with stratification
        # train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        #     range(len(self.dataset)), labels, stratify=labels, test_size=0.2, random_state=42
        # )
        # val_idx, test_idx, val_labels, test_labels = train_test_split(
        #     temp_idx, temp_labels, stratify=temp_labels, test_size=0.5, random_state=42
        # )
        
        # self.train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        # self.val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        # self.test_dataset = torch.utils.data.Subset(self.dataset, test_idx)

        # # split the dataset based on the label so that the ratio of the label is the same

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
            )
        if stage == "test" or stage is None:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def _get_label_weights(self, datasets):
        labels = torch.tensor([y for _, y in datasets])
        label_weights = torch.tensor([1 / labels.eq(i).sum().item() for i in range(6)])
        label_weights = label_weights / label_weights.sum()
        return label_weights

    # -----------------
    def get_train_label_weights(self):
        return self._get_label_weights(self.train_dataset)

    def get_val_label_weights(self):
        return self._get_label_weights(self.val_dataset)

    def get_test_label_weights(self):
        return self._get_label_weights(self.test_dataset)
