import torch
import numpy as np
import os
import glob
from pprint import pprint

if __name__ == "__main__":
    folder = "/mnt/Volume02/MotionID_split/"
    files = glob.glob(f"{folder}/*.pt")
    files.sort()
    size_dict = {}

    for f in files:
        name = f.split("/")[-1]
        size = os.path.getsize(f)
        size_dict[name] = size

    # sort the size_dict
    sorted_size_dict = sorted(size_dict.items(), key=lambda x: x[1], reverse=True)
    # convert the size to GB
    sorted_size_dict = [(x[0], x[1] / 1024 / 1024 / 1024) for x in sorted_size_dict]
    # pprint(sorted_size_dict)
    # divide into 102 files as a group
    groups = []
    for i in range(0, len(sorted_size_dict), 46):
        groups.append(sorted_size_dict[i : i + 46])

    # groups the 2th and 3rd odd index as a group even index as a group
    reorder = groups.copy()
    new_odd = []
    new_even = []
    for i in range(0, 46, 2):
        new_odd.append(groups[1][i])
        new_odd.append(groups[2][i])
        new_even.append(groups[1][i + 1])
        new_even.append(groups[2][i + 1])
    groups[1] = new_odd
    groups[2] = new_even

    # print groups size
    for i, g in enumerate(groups):
        total_size = sum([x[1] for x in g])
        target_list = [x[0] for x in g]
        print(target_list)
        assert len(target_list) == 46
        print(f"Group {i}: {total_size:.2f}G")
        print("====================================================")

    # # print every 102 files
    # reorder = []
    # for i in range(0, len(sorted_size_dict), 102):
    #     pprint(sorted_size_dict[i : i + 102])
    #     # append the name to reorder
    #     reorder.append([x[0] for x in sorted_size_dict[i : i + 102]])

    #     # print total size
    #     total_size = sum([x[1] for x in sorted_size_dict[i : i + 102]])
    #     print(f"Total size: {total_size}")
    #     print(reorder[-1])
    #     print("====================================================")


    
