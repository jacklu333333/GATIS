import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import colored as cl

if __name__ == "__main__":
    # path = Path("/mnt/Volume01/MotionID_Processed_tensor_withoutscale/")
    # path = Path("/mnt/Volume02/MotionID_split_withoutscale/")
    path = Path("/mnt/Volume02/MotionID_split_withoutscale_9channel")

    # path = Path("/mnt/Volume01/MotionID_Processed_tensor/")
    # path = Path("/mnt/Volume02/MotionID_split/")
    # path = Path(
    #     "/home/jack/Documents/verilog/ASIC/PDR/spectrums/upperstream/RoNIN/Processed/"
    # )
    files = glob.glob(f"{path}/**/*.pt", recursive=True)
    files.sort()
    total = []
    for file in tqdm(files):
        # for file in files:
        # tensor = torch.load(file)["time_data"]
        try:
            tensor = torch.load(file)
            # total.append(tensor['time'].shape[0])
            total.append(len(tensor))
            del tensor
        except:
            print(f"{cl.Fore.red}Error: {file}{cl.Style.reset}")
            sys.exit(1)

    total = np.cumsum(total)
    total = torch.from_numpy(total)

    torch.save(total, path / "total.pt")
