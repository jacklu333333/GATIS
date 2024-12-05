import glob
import os
from pathlib import Path

import numpy as np
import torch

CHUNK_SIZE = 1024


def save_chunks(data, chunk_size, output_dir):
    num_chunks = len(data) // chunk_size
    remainder = len(data) % chunk_size

    for i in range(num_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        file_path = output_dir / f"chunk_{i}.pt"
        torch.save(chunk, file_path)

    if remainder > 0:
        last_chunk = data[-remainder:]
        file_path = output_dir / f"chunk_{num_chunks}.pt"
        torch.save(last_chunk, file_path)


if __name__ == "__main__":
    path = Path("/mnt/Volume01/MotionID_Processed_tensor")
    files = glob.glob(os.path.join(path, "*.pt"))
    count = 0

    for file in files:
        data = torch.load(file)
        save_chunks(data, 1024, path)
