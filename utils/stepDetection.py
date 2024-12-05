from typing import Union

import numpy as np
import torch
from tqdm import tqdm

# make a function of step detection
# ilterate over the acc vairable
# if the acc is greater than 1.5 then it is a step
# then grab 100 data points before and after the step

shift = 0.6
UPTHRESHOLD = 9.81 + shift
DOWNTHRESHOLD = 9.81 - shift


def stepDetection(acc: np.array) -> np.array:
    stepStart = []
    stepEnd = []
    reachTop = False
    for i in range(len(acc)):
        if acc[i] > UPTHRESHOLD and reachTop == False:
            stepStart.append(i)
            reachTop = True

        # elif acc[i] < 8 and reachTop == True:
        #     reachTop = False
        #     # remove the stepStart last element
        #     stepStart.pop()

        elif acc[i] < DOWNTHRESHOLD and reachTop == True:
            stepEnd.append(i)
            reachTop = False
            # reverse trace the stepStart
            stepStart.pop()
            reverseTop = False
            for j in range(i, i - 100, -1):
                # print(j,acc[j], reverseTop)
                if acc[j] > UPTHRESHOLD and reverseTop == False:
                    reverseTop = True
                elif acc[j] < UPTHRESHOLD and reverseTop == True:
                    reverseTop = False
                    stepStart.append(j)
                    break

            if len(stepStart) != len(stepEnd):
                # remove start last element
                stepEnd.pop()
            assert len(stepStart) == len(stepEnd)
            # if add step start and end distance is larger than 100 then remove the last step
            if len(stepStart) != 0:
                if stepEnd[-1] - stepStart[-1] >= 100:
                    stepStart.pop()
                    stepEnd.pop()

    if len(stepStart) != len(stepEnd):
        # remove start last element
        stepStart.pop()

    assert len(stepStart) == len(stepEnd)
    # print("Number of steps: ", len(stepStart))
    return np.array(stepStart), np.array(stepEnd)


def endStepBackwardSampling(
    acc: Union[np.array, torch.tensor],
    gyr: Union[np.array, torch.tensor],
    mag: Union[np.array, torch.tensor],
    stepEnd: Union[np.array, torch.tensor],
    WINDOW_SIZE: int = 100,
) -> Union[np.array, torch.tensor]:
    # check tensor or numpy
    if isinstance(acc, torch.Tensor):
        is_torch = True
    else:
        is_torch = False
    if not is_torch:
        acc = torch.from_numpy(acc).float()
        gyr = torch.from_numpy(gyr).float()
        mag = torch.from_numpy(mag).float()
        stepEnd = torch.from_numpy(stepEnd).float()

    # check the shape
    assert acc.shape[-1] == 4
    assert gyr.shape[-1] == 4
    assert mag.shape[-1] == 4
    assert acc.shape[0] == gyr.shape[0] == mag.shape[0]

    raw = torch.cat((acc, gyr, mag), dim=1)

    data = []
    for index in stepEnd:
        if index - WINDOW_SIZE < 0:
            continue
        newdata = raw[index - WINDOW_SIZE : index + WINDOW_SIZE].swapaxes(0, 1)
        if not is_torch:
            newdata = newdata.numpy()
        data.append(newdata)

    return torch.stack(data) if is_torch else np.stack(data)


def startStepForwardSampling(
    acc: Union[np.array, torch.tensor],
    gyr: Union[np.array, torch.tensor],
    mag: Union[np.array, torch.tensor],
    stepStart: Union[np.array, torch.tensor],
    WINDOW_SIZE: int = 100,
) -> Union[np.array, torch.tensor]:
    # check tensor or numpy
    if isinstance(acc, torch.Tensor):
        is_torch = True
    else:
        is_torch = False
    if not is_torch:
        acc = torch.from_numpy(acc).float()
        gyr = torch.from_numpy(gyr).float()
        mag = torch.from_numpy(mag).float()
        stepStart = torch.from_numpy(stepStart).float()

    # check the shape
    assert acc.shape[-1] == 4
    assert gyr.shape[-1] == 4
    assert mag.shape[-1] == 4
    assert acc.shape[0] == gyr.shape[0] == mag.shape[0]

    raw = torch.cat((acc, gyr, mag), dim=1)

    data = []
    for index in stepStart:
        if index + WINDOW_SIZE > len(raw):
            continue
        newdata = raw[index - WINDOW_SIZE : index + WINDOW_SIZE].swapaxes(0, 1)
        if not is_torch:
            newdata = newdata.numpy()
        data.append(newdata)

    return torch.stack(data) if is_torch else np.stack(data)
