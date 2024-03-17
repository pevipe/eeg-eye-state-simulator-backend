import numpy as np
from math import floor
import os

from src.feature_extraction.rates import Rate
from src.feature_extraction.window import Window
from src.feature_extraction.constants import dataset_path, fs, window_size, alpha_lowcut, alpha_highcut, beta_lowcut, beta_highcut


def load_csv(file):
    data = np.loadtxt(open(file, "rb"), delimiter=",")
    data = data[:, 1:4]
    return data


def normalize(unnormalized_data):
    normalized = unnormalized_data[:, 0:2] - np.mean(unnormalized_data[:, 0:2])
    return np.append(normalized, unnormalized_data[:, 2].reshape(-1, 1), axis=1)


# Generates n windows of size window_size from the data
def all_windowing(total_data, total_time, win_size):
    n = floor(total_time / win_size)

    windows_list = []

    for i in range(n):
        start_time = i * win_size
        end_time = (i + 1) * win_size
        windows_list.append(Window(total_data, start_time, end_time, fs))

    return windows_list


# Performs all necessary operations to transform csv into entries for a classificator (rates and target)
def load_dataset(win_size):
    files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

    windows = []
    for f in files:
        data = load_csv(dataset_path + "/" + f)
        data = normalize(data)
        total_time = 600
        windows = windows + all_windowing(data, total_time, win_size)

    dataset = []
    for w in windows:
        dataset.append(Rate(w, alpha_lowcut, alpha_highcut, beta_lowcut, beta_highcut, fs).to_classificator_entry())
    return np.array(dataset)


if __name__ == '__main__':
    load_dataset(window_size)
