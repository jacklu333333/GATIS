import glob

import torch
from tqdm import tqdm

folder = "/mnt/Volume02/MotionID_split/"

files = glob.glob(folder + "*.pt")
# print(len(files))
for file in tqdm(files):
    data = torch.load(file)
    # save the data
    torch.save(data, file)
    # print(file)
