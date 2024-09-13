# Copyright 2024 Pelayo Vieites Pérez
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

import numpy as np
import os

from src.domain.feature_extraction.ratio import Ratio
from src.domain.feature_extraction.window import Window


class DataLoader:
    def __init__(self, dataset_path: str, output_path: str, win_size: int,
                 exact_windows_path: str, fs: int = 200) -> None:
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.window_size = win_size
        self.overlap = self.window_size - 2
        self.exact_windows_path = exact_windows_path
        self.fs = fs

        #############
        # Constants #
        #############
        self.alpha_lowcut = 7.0
        self.alpha_highcut = 12.0
        self.beta_lowcut = 14.0
        self.beta_highcut = 25.0
        self.total_time = 600  # 10 minutes

        #############################
        # Attribute initializations #
        #############################
        self.dataset = None
        self.exact_windows_indexes = None

        #####################
        # Method invocation #
        #####################
        self.load_a_dataset()  # Load the dataset and exact window indexes
        self.export_dataset()  # Export the data so it can be used next time

    @staticmethod
    def _all_windowing(total_data: np.ndarray, total_time: int, win_size: int, fs: int, overlap: int = 0) -> list[Window]:
        """
        Window the given data, creating a list of Window objects.
        :param total_data: array with the data. Each row is a sample, and the columns are [signal_o1, signal_o2, target]
        :param total_time: total recorded time, in seconds
        :param win_size: size of the time window
        :param fs: frequency of sampling (samples per second)
        :param overlap: overlap between time windows
        :return: list of Window objects
        """
        windows_list = []
        start_time = 0
        end_time = win_size

        while end_time <= total_time:
            windows_list.append(Window(total_data, start_time, end_time, fs))
            start_time = start_time + win_size - overlap
            end_time = start_time + win_size

        return windows_list

    @staticmethod
    def _load_csv(file: str) -> np.ndarray:
        """
        Load a CSV file with the specified format (timestamp;signal_o1;singal_o2;target) into an array with the data
        :param file: path to the csv file
        :return: the data, in an array with columns [signal_o1, signal_o2, target] and a sample per row
        """
        data = np.loadtxt(open(file, "rb"), delimiter=",")
        data = data[:, 1:4]
        return data

    def load_a_dataset(self):
        """
        Load a dataset from a CSV file and window it and obtain the ratios.
        Convert csv into the input of the classifiers -> columns [ratio_o1, ratio_o1, target] and a window per row
        """
        # Clear the dataset in case it contained something
        self.dataset = None
        self.exact_windows_indexes = None

        # If the dataset had been previously loaded and saved -> load from the output path
        if os.path.exists(self.output_path):
            self.dataset = np.loadtxt(open(self.output_path, "rb"), delimiter=",")

        if os.path.exists(self.exact_windows_path):
            # load the dataset from the output path
            self.exact_windows_indexes = np.loadtxt(open(self.exact_windows_path, "rb"), delimiter=",")

        if self.dataset is not None and self.exact_windows_indexes is not None:
            return

        # Load the individual subject specified
        data = self._load_csv(self.dataset_path)

        # Take the samples with overlapping (sliding time window each 2 seconds)
        windows = self._all_windowing(data, self.total_time, self.window_size, self.fs, overlap=self.overlap)

        dataset = []
        self.exact_windows_indexes = []
        for i, w in enumerate(windows):
            dataset.append(
                Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                      self.fs).to_classificator_entry())
            if 0 < w.mean_targets < 1:  # Window does not contain unique state
                pass
            else:
                self.exact_windows_indexes.append(i)

        self.dataset = np.array(dataset)

    def export_dataset(self):
        """
        Save the dataset and exact windows indexes to a CSV file
        """
        if not os.path.exists(self.output_path):
            np.savetxt(self.output_path, self.dataset, delimiter=",")
        if not os.path.exists(self.exact_windows_path):
            np.savetxt(self.exact_windows_path, self.exact_windows_indexes, delimiter=",")
