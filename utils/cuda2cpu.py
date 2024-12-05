import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_folder", type=str, help="target folder")
    args = parser.parse_args()

    target_folder = Path(args.target_folder)

    files = list(target_folder.glob("*.pt"))
    print(f"Found {len(files)} files in {target_folder}")
    for file in tqdm.tqdm(files):
        data = torch.load(file)
        data = data.cpu()
        torch.save(data, file)

        # # randomly sample 10 of the data and plot
        # fig, axes = plt.subplots(3, 10, figsize=(20, 6))
        # for i in range(10):
        #     idx = np.random.randint(0, data.size(0))
        #     axes[0, i].plot(data[idx, 3], label="acc")
        #     axes[1, i].plot(data[idx, 7], label="gyr")
        #     axes[2, i].plot(data[idx, 11], label="mag")
        # plt.tight_layout()
        # plt.show()
        # c = input("Continue? (Y/n)")
        # if c.lower() == "n":
        #     break
    print("Done")
