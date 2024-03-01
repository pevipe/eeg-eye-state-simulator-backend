import numpy as np
from math import floor
import os

from window import Window
from constants import dataset_path, fs, window_size


def load_csv(file):
    data = np.loadtxt(open(file, "rb"), delimiter=",")
    data = data[:, 1:4]
    return data


# Generates n windows of size window_size from the data
def all_windowing(total_data, total_time):
    n = floor(total_time / window_size)

    windows_list = []

    for i in range(n):
        start_time = i * window_size * fs
        end_time = (i + 1) * window_size * fs
        windows_list.append(Window(total_data, start_time, end_time, fs))

    return windows_list


if __name__ == '__main__':
    files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    fs = 200
    window_size = 10
    windows = []

    for f in files:
        data = load_csv(dataset_path+"/"+f)
        data = data - np.mean(data)
        total_time = 600
        windows = windows + all_windowing(data, total_time)

    print(len(windows))
