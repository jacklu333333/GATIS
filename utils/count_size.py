import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if __name__ == "__main__":
    path = Path("/mnt/Volume01/MotionID_Processed_tensor/")
    files = glob.glob(f"{path}/**/*.pt", recursive=True)
    files.sort()
    files.remove(str(path / "total.pt"))
    size = dict()
    for file in tqdm(files):
        # get the file size
        size[file.split("/")[-1]] = os.path.getsize(file)

    df = pd.DataFrame.from_dict(size, orient="index", columns=["size[Byte]"])
    # sort
    df = df.sort_values(by="size[Byte]", ascending=False)
    df.to_csv("size_of_tensorfile.csv")
    print(df.head(30))
    # total = np.cumsum(total)
    # total = torch.from_numpy(total)

    # torch.save(total, path / "total.pt")
