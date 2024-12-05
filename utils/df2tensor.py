import argparse
import datetime
import glob
import multiprocessing as mp
import os
from pathlib import Path
from typing import Union

import colored as cl
import numpy as np
import pandas as pd
import torch
from geoPreprocessor import GravityRemoval, magRescale
from tqdm import tqdm

FRONTDROP = 3000


def df2tensor(file: Union[Path, str], save_dir: Union[Path, str]) -> None:
    """
    Convert csv file to tensor file and store as .pt file
    ----------------------------------------------------------------
    Input:
        file: csv file path
        save_dir: directory to save the tensor file
    Output:
        None
    """
    try:
        # check file size if more than 10G then skip
        #                                 G      M      K
        if os.path.getsize(file) > 1 * 1024 * 1024 * 1024:
            print(
                f"{cl.Fore.yellow} file too large: {file}: size:{os.path.getsize(file)/1024 / 1024 / 1024:.2f}G {cl.Style.reset}"
            )
            return
        df = pd.read_csv(file)
        date = df["timestamp"].iloc[0]  # ex: 2021-02-08 14:25:46.600
        # date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")

        acc = df[["acc.X", "acc.Y", "acc.Z", "acc"]].to_numpy()[FRONTDROP:]
        gyr = df[["gyr.X", "gyr.Y", "gyr.Z", "gyr"]].to_numpy()[FRONTDROP:]
        mag = df[["mag.X", "mag.Y", "mag.Z", "mag"]].to_numpy()[FRONTDROP:]

        acc = torch.from_numpy(acc).float().cuda()
        gyr = torch.from_numpy(gyr).float().cuda()
        mag = torch.from_numpy(mag).float().cuda()

        assert acc.shape == gyr.shape == mag.shape
        if len(acc) < 100:
            return
        # rescale
        acc = GravityRemoval(acc, location="MotionID", date=date, rescale=False)
        gyr = gyr  # / np.pi
        mag = magRescale(mag, location="MotionID", date=date)

        # slice every 100 datapoint
        length = len(acc)
        keep = int(length // 100 * 100)
        acc, gyr, mag = acc[:keep], gyr[:keep], mag[:keep]

        acc = acc.reshape(-1, 100, 4).swapaxes(1, 2)
        gyr = gyr.reshape(-1, 100, 4).swapaxes(1, 2)
        mag = mag.reshape(-1, 100, 4).swapaxes(1, 2)

        # acc = torch.from_numpy(acc).float()
        # gyr = torch.from_numpy(gyr).float()
        # mag = torch.from_numpy(mag).float()

        if "indoor" in df.columns:
            indoor = df[["indoor"]].to_numpy()[FRONTDROP:]
            assert len(indoor) == length
            # indoor = torch.from_numpy(indoor).float()
            indoor = indoor[:keep]
            indoor = indoor.reshape(-1, 100, 1)
            indoor = torch.from_numpy(indoor).float()
            # output = dict(
            #     acc=acc,
            #     gyr=gyr,
            #     mag=mag,
            #     indoor=indoor,
            # )
            raise NotImplementedError
        else:
            # merge the acc gyr mag by axis=1
            time_data = torch.cat([acc, gyr, mag], dim=1)
            batch, channel, length = time_data.shape
            # freq_data = torch.stft(
            #     time_data.reshape(-1, length),
            #     n_fft=100,
            #     hop_length=100,
            #     win_length=100,
            #     window=torch.hann_window(100),
            #     return_complex=False,
            # )
            # # convert complex to real
            # # freq_data = torch.cat([freq_data.real, freq_data.imag], dim=-1)
            # freq_data = freq_data.pow(2).sum(-1).reshape(batch, -1, 51, 2)

            # output = dict(
            #     time_data=time_data,
            #     freq_data=freq_data,
            # )
            output = time_data
        torch.save(output, save_dir / Path(file).name.replace(".csv", ".pt"))
    except Exception as e:
        print(f"{cl.Fore.red} {e} {cl.Style.reset}")
        print(f"file name: {cl.Fore.red} {file} {cl.Style.reset}")


if __name__ == "__main__":
    # argument parser
    # add default
    parser = argparse.ArgumentParser()
    parser.add_argument("--portion", type=int, default=0)
    # set to variable
    portion = parser.parse_args().portion

    # dir
    folder = "/mnt/Volume01/MotionID_Processed_wm"
    save_dir = Path(folder.replace("_wm", "_tensor_withoutscale"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # list all csv files in the folder
    files = glob.glob(os.path.join(folder, "*.csv"))
    files.sort()
    files = [Path(f).name for f in files]
    # portion size
    portion_size = int(len(files) / 4)
    if portion != 0:
        if portion < 5:
            files = files[portion_size * (portion - 1) : portion_size * portion]
        else:
            raise ValueError("portion should be in range 1-4")

    # remove the files that are already done
    done_files = glob.glob(os.path.join(save_dir, "*.pt"))
    done_files = [Path(f).name for f in done_files]
    done_files = [f.replace(".pt", ".csv") for f in done_files]

    original_len = len(files)
    files = list(set(files) - set(done_files))
    print(f"original_len: {original_len}")
    print(f"len(done_files): {len(done_files)}")
    print(f"len(files): {len(files)}")

    files = [folder + "/" + file for file in files]

    # # loop over all files
    # files = [f.split("/")[-1] for f in files]

    for file in tqdm(files):
        df2tensor(file, Path(save_dir))

    # with mp.Pool(processes=mp.cpu_count() // 4, maxtasksperchild=1) as pool:
    #     for file in tqdm(files):
    #         pool.apply_async(df2tensor, args=(file, save_dir))
    #     pool.close()
    #     pool.join()
    print("All Done")
