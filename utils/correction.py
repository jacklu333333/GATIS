import datetime

import ahrs
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as R


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def rotateToWorldFrame(
    acc: np.array,
    gyr: np.array,
    mag: np.array,
    cutoff: float = 50.0,
    name: str = "",
    sample_rate: float = 100.0,
    return_ori: bool = False,
) -> (np.array, np.array, np.array):
    """
    This function rotates the sensor frame to world frame using the Madgwick algorithm
    Input:
        acc: nx3 numpy array of accelerometer data
        gyr: nx3 numpy array of gyroscope data
        mag: nx3 numpy array of magnetometer data
    Output:
        acc: nx3 numpy array of accelerometer data in world frame
        gyr: nx3 numpy array of gyroscope data in world frame
        mag: nx3 numpy array of magnetometer data in world frame
    """
    # # Filter requirements.
    # order = 4
    # fs = 100.0  # sample rate, Hz
    # # cutoff = 50  # desired cutoff frequency of the filter, Hz
    # b, a = butter_lowpass(cutoff, fs, order)
    # acc = butter_lowpass_filter(acc, cutoff, fs, order)
    # gyr = butter_lowpass_filter(gyr, cutoff, fs, order)
    # mag = butter_lowpass_filter(mag, cutoff, fs, order)

    # Initialize the filter
    filter = ahrs.filters.Madgwick(
        acc=acc, gyr=gyr, mag=mag / 10, frequency=sample_rate, gain=0.1
    )

    # if "migration" in name:
    #     lat = 36.36907833333333
    #     lon = 127.36171166666666
    #     date = name.split("/")[-1].replace(".csv", "")
    #     date = datetime.datetime.strptime(date, "%Y-%m-%d-%H:%M:%S")
    #     noise = [
    #         0.004937662039113619**2,
    #         0.026033755765741105**2,
    #         0.006771730824538068**2,
    #     ]

    # elif "advio" in name:
    #     lat = 61.44971833184961
    #     lon = 23.858864487427393
    #     date = 2018.5
    #     noise = [0.3**2, 0.5**2, 0.8**2]  # default
    # else:
    #     raise ValueError("Wrong dataset name")

    # Gravity = ahrs.utils.WGS()
    # Gravity.normal_gravity(lat, 0)
    # Magnetic = ahrs.utils.WMM()
    # Magnetic.magnetic_field(
    #     lat,
    #     lon,
    #     date=date,
    # )
    # print(Gravity.normal_gravity(lat, 0), Magnetic.F)

    # filter = ahrs.filters.EKF(
    #     gyr=gyr,
    #     acc=acc,
    #     mag=mag * 1e5,
    #     frequency=100.0,
    #     magnetic_ref=[
    #         Magnetic.X * 1e-3,
    #         Magnetic.Y * 1e-3,
    #         Magnetic.Z * 1e-3,
    #     ],
    #     # noise=noise,
    #     gyr_var=noise[0],
    #     acc_var=noise[1],
    #     mag_var=noise[2],
    #     frame="NED",
    # )

    # get the inverse quaternion
    quad = filter.Q
    quad = np.concatenate((quad[:, 1:], quad[:, :1]), axis=1)
    r = R.from_quat(quad)
    inv = r.inv()
    acc = r.apply(acc)
    gyr = r.apply(gyr)
    mag = r.apply(mag)
    if return_ori:
        return acc, gyr, mag, r
    return acc, gyr, mag
