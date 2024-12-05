import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


path = Path("/mnt/Volume01/MotionID_Processed/")
files = glob.glob(str(path) + "/*.csv")

df_count = pd.DataFrame()

for file in tqdm(files):
    df = pd.read_csv(file)
    acc = df[["acc"]].to_numpy()
    gyr = df[["gyr"]].to_numpy()
    mag = df[["mag"]].to_numpy()
    new_row = pd.DataFrame(
        {
            "file": file.split("/")[-1],
            #
            "acc_mean": np.mean(acc, axis=0),
            "acc_std": np.std(acc, axis=0),
            'acc_max': np.max(acc, axis=0),
            'acc_min': np.min(acc, axis=0),
            #
            "gyr_mean": np.mean(gyr, axis=0),
            "gyr_std": np.std(gyr, axis=0),
            'gyr_max': np.max(gyr, axis=0),
            'gyr_min': np.min(gyr, axis=0),
            #
            "mag_mean": np.mean(mag, axis=0),
            "mag_std": np.std(mag, axis=0),
            'mag_max': np.max(mag, axis=0),
            'mag_min': np.min(mag, axis=0),

            "count": len(df),
        },
        index=[0],
    )
    df_count = pd.concat([df_count, new_row], ignore_index=True)

save_path = path / "metadata"
save_path.mkdir(parents=True, exist_ok=True)
df_count.to_csv(save_path / (str(path).split("/")[-1] + ".csv"), index=False)
