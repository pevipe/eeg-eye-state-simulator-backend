import numpy as np
from math import floor
import os

from src.feature_extraction.rates import Rate
from src.feature_extraction.window import Window
from src.feature_extraction.constants import alpha_lowcut, alpha_highcut, beta_lowcut, beta_highcut


class DataLoader:
    def __init__(self, dataset_path, fs, win_size, alpha_lowcut=alpha_lowcut, alpha_highcut=alpha_highcut,
                 beta_lowcut=beta_lowcut, beta_highcut=beta_highcut):
        self.dataset_path = dataset_path
        self.fs = fs
        self.window_size = win_size

        #############
        # Constants #
        #############
        self.alpha_lowcut = alpha_lowcut
        self.alpha_highcut = alpha_highcut
        self.beta_lowcut = beta_lowcut
        self.beta_highcut = beta_highcut
        self.dataset = None

    @staticmethod
    def _normalize(unnormalized_data):
        normalized = unnormalized_data[:, 0:2] - np.mean(unnormalized_data[:, 0:2])
        return np.append(normalized, unnormalized_data[:, 2].reshape(-1, 1), axis=1)

    @staticmethod
    def _all_windowing(total_data, total_time, win_size, fs):
        n = floor(total_time / win_size)

        windows_list = []

        for i in range(n):
            start_time = i * win_size
            end_time = (i + 1) * win_size
            windows_list.append(Window(total_data, start_time, end_time, fs))

        return windows_list


class ProvidedDatasetLoader(DataLoader):
    def __init__(self, dataset_path, fs, win_size):
        super().__init__(dataset_path, fs, win_size)
        self.dataset = None

    @staticmethod
    def _load_csv(file):
        data = np.loadtxt(open(file, "rb"), delimiter=",")
        data = data[:, 1:4]
        return data

    def load_dataset(self, exact_targets=False):
        files = [f for f in os.listdir(self.dataset_path) if f.endswith(".csv")]

        windows = []
        for f in files:
            data = ProvidedDatasetLoader._load_csv(self.dataset_path + "/" + f)
            data = self._normalize(data)
            total_time = 600
            windows = windows + self._all_windowing(data, total_time, self.window_size, self.fs)

        dataset = []
        for w in windows:
            dataset.append(Rate(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                self.fs).to_classificator_entry(exact_targets))
        self.dataset = np.array(dataset)
        return self.dataset


class DHBWDatasetLoader(DataLoader):
    def __init__(self, dataset_path, fs, win_size, total_time):
        super().__init__(dataset_path, fs, win_size)
        self.dataset = None
        self.total_time = total_time

    @staticmethod
    def _load_csv(file):
        data = np.genfromtxt(file, delimiter=",", skip_header=1)
        data = data[:, [6, 7, 14]]
        return data

    def load_dataset(self, exact_targets=False):
        data = DHBWDatasetLoader._load_csv(self.dataset_path)
        data = self._normalize(data)

        windows = self._all_windowing(data, self.total_time, self.window_size, self.fs)

        dataset = []
        for w in windows:
            dataset.append(Rate(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                self.fs).to_classificator_entry(exact_targets))
        self.dataset = np.array(dataset)
        return self.dataset
