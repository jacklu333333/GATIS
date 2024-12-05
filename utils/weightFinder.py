import os
import sys
import glob
from pathlib import Path
import colored as cl


class WeightFinder:
    def __init__(self, path, keyword):
        self.path = path
        self.keyword = keyword
        self.weight_path = self.find_weight_path()
        self.weight_name = self.find_weight_name()
        # self.weight = self.find_weight()

    def find_weight_path(self):
        # list all dir in path
        folders = os.listdir(self.path)
        folders.sort(reverse=True)
        # find the first dir that contains the keyword
        for folder in folders:
            if self.keyword in folder:
                # find the first weight file in the dir
               # weight_path = glob.glob(os.path.join(self.path, folder, "*.pth"))
                # weight_path.sort(reverse=True)
                return folder

    def find_weight_name(self):
        # find all the file end with .ckpt recursivel
        print(self.weight_path)
        files = [
            file
            for file in glob.glob(
                os.path.join(self.path,self.weight_path,"**", "*.ckpt"), recursive=True
            )
        ]
        files.sort()
        print(f"Find {len(files)} files in {self.weight_path}")
        for file in files:
            print(f'    {cl.fg("green")}{file}{cl.attr("reset")}')
        return files

    # def find_weight(self):
    #     weight = os.path.getsize(self.weight_path[0])
    #     return weight

    # def print_weight(self):
    #     print("Weight path: ", self.weight_path)
    #     print("Weight name: ", self.weight_name)
    #     print("Weight size: ", self.weight)
