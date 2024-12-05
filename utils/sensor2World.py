import argparse
import datetime
import glob
import multiprocessing as mp
import pickle
from pathlib import Path

import ahrs
import numpy as np
import pandas as pd
from tqdm import tqdm

from .correction import rotateToWorldFrame
from .geoPreprocessor import GravityRemoval, MagneticRemoval


def sensor2WorldFrame(file, folder, save_path):
    if "migration" in folder:
        location = "migration"
    elif "advio" in folder:
        location = "advio"
    elif "MotionID" in folder:
        location = "MotionID"
    else:
        raise ValueError("folder name must be migration, advio or MotionID")

    df = pd.read_csv(file)
    # if len(df) < 3100:
    #     return
    if location == "migration":
        date = df["timestamp"].iloc[0]
        # convert to datetime from str ex :2021-03-16 07:45:14.890
        date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    elif location == "advio":
        date = 2018.5
    elif location == "MotionID":
        date = df["timestamp"].iloc[0]
        # example:2021-02-08 14:25:46.670
        date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    else:
        raise ValueError("location name must be migration, advio or MotionID")

    acc = df[["acc.X", "acc.Y", "acc.Z"]].to_numpy()
    gyr = df[["gyr.X", "gyr.Y", "gyr.Z"]].to_numpy()
    mag = df[["mag.X", "mag.Y", "mag.Z"]].to_numpy()
    acc, gyr, mag = rotateToWorldFrame(acc, gyr, mag)
    # mag = MagneticRemoval(mag=mag, location=location, date=date)[:, :3]
    # acc = GravityRemoval(location, acc, date)

    df[["acc.X", "acc.Y", "acc.Z"]] = acc
    df[["gyr.X", "gyr.Y", "gyr.Z"]] = gyr
    df[["mag.X", "mag.Y", "mag.Z"]] = mag

    df.to_csv(save_path / Path(file).name, index=False)
    print(f"{file} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--portion", type=int, default=0)

    portion = parser.parse_args().portion
    print(f"portion: {portion}")
    # exit()
    folder = "/mnt/Volume01/MotionID_Processed"

    save_path = Path(folder + "_wm")
    # files = glob.glob(f"{folder}/*.csv")
    df = pd.read_csv("count.csv")
    # sort file by column 'count'
    df = df.sort_values(by=["count"], ascending=False)
    # keep only row which count > 3100
    df = df[df["count"] > 3100]
    files = df["filename"].to_list()

    save_path.mkdir(exist_ok=True, parents=True)
    already_done = glob.glob(f"{save_path}/*.csv")
    # if name is same, remove from files
    # files = [file.split("/")[-1] for file in files]
    already_done = [file.split("/")[-1] for file in already_done]
    files = list(set(files) - set(already_done))
    # add prefix back

    files = [folder + "/" + file for file in files]

    # with open("toosmallfiles.pickle", "rb") as f:
    #     toosmall = pickle.load(f)
    # toosmall = []
    # for file in files:
    #     df = pd.read_csv(file)
    #     if len(df) < 3100:
    #         toosmall.append(file)

    # files = list(set(files) - set(toosmall))
    # files = toosmall

    # # save with pickle
    # with open("toosmallfiles.pickle", "wb") as f:
    #     pickle.dump(, f)

    # for file in tqdm(files):
    #     sensor2WorldFrame(file)

    # modify to use multiprocessing
    # limit only use 16 core to avoid memory error
    # files = files[-100:]

    print(f"total {len(files)} files")
    # with mp.Pool(processes=24, maxtasksperchild=1024) as pool:
    #     for file in files:
    #         pool.apply_async(
    #             sensor2WorldFrame,
    #             args=(file, folder, save_path),
    #         )

    #     pool.close()
    #     pool.join()

    # equaly split files to 4 portion
    # files = np.array_split(files, 4)[portion].tolist()
    files = files
    print(f"portion {portion} has {len(files)} files")
    for file in tqdm(files):
        sensor2WorldFrame(file, folder, save_path)
