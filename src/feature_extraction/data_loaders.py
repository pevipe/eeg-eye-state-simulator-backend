import numpy as np
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
    def _all_windowing(total_data, total_time, win_size, fs, overlap=0):
        windows_list = []
        start_time = 0
        end_time = win_size

        while end_time <= total_time:
            windows_list.append(Window(total_data, start_time, end_time, fs))
            start_time = start_time + win_size - overlap
            end_time = start_time + win_size

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

    def load_dataset(self, exact_targets=False, normalize=False):
        files = [f for f in os.listdir(self.dataset_path) if f.endswith(".csv")]

        windows = []
        for f in files:
            data = ProvidedDatasetLoader._load_csv(self.dataset_path + "/" + f)
            if normalize:
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

    def load_dataset(self, exact_targets=False, normalize=False):
        data = DHBWDatasetLoader._load_csv(self.dataset_path)
        if normalize:
            data = self._normalize(data)

        windows = self._all_windowing(data, self.total_time, self.window_size, self.fs)

        dataset = []
        for w in windows:
            dataset.append(Rate(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                self.fs).to_classificator_entry(exact_targets))
        self.dataset = np.array(dataset)
        return self.dataset


class ProvidedDatasetIndividualLoader(ProvidedDatasetLoader):
    def __init__(self, dataset_path, output_path, fs, win_size):
        super().__init__(dataset_path, fs, win_size)
        self.dataset = None
        self.output_path = output_path

    def load_all_datasets(self, overlap=0, exact_targets=False, normalize=False):
        """
            Load the datasets from each individual subject. Returns the list with the datasets.
            Windowing is now done with overlapping, sliding the time window each 2 seconds.
        """
        self.dataset = []  # Clear the dataset in case it contained something
        files = [f for f in os.listdir(self.dataset_path) if f.endswith(".csv")]

        # Check if the datasets have already been calculated
        if os.path.exists(self.output_path):
            out_files = [f for f in os.listdir(self.output_path) if f.endswith(".csv")]
            if len(out_files) == len(files):
                # Load the dataset that had already been calculated
                for f in out_files:
                    self.dataset.append(np.loadtxt(open(self.output_path + "/" + f, "rb"), delimiter=","))
                print("Datasets loaded from the output path.")
                return self.dataset

        # If not, calculate the datasets and export them so then can be imported later
        for f in files:
            data = ProvidedDatasetLoader._load_csv(self.dataset_path + "/" + f)
            if normalize:
                data = self._normalize(data)
            # Take the samples with overlapping (sliding time window each 2 seconds)
            total_time = 600
            windows = self._all_windowing(data, total_time, self.window_size, self.fs, overlap=overlap)

            dataset = []
            for w in windows:
                dataset.append(Rate(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                    self.fs).to_classificator_entry(exact_targets))
            self.dataset.append(np.array(dataset))

        # Export the datasets so that time can be saved next time
        self.export_datasets()

        return self.dataset

    def load_single_dataset(self, subject_number, overlap=0, exact_targets=False, normalize=False):
        self.dataset = None  # Clear the dataset in case it contained something

        # Load the individual subject specified
        data = ProvidedDatasetLoader._load_csv(self.dataset_path + "/Sujeto_" + str(subject_number) + ".csv")
        if normalize:
            data = self._normalize(data)

        # Take the samples with overlapping (sliding time window each 2 seconds)
        total_time = 600
        windows = self._all_windowing(data, total_time, self.window_size, self.fs, overlap=overlap)

        dataset = []
        for w in windows:
            dataset.append(Rate(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                self.fs).to_classificator_entry(exact_targets))
        self.dataset = np.array(dataset)

        return self.dataset

    def export_datasets(self, route=None):
        if route is None:
            route = self.output_path
        # Create the directory if it does not exist
        if not os.path.exists(route):
            os.makedirs(route)

        # Save the dataset to CSV files
        for i, dataset in enumerate(self.dataset):
            np.savetxt(route + "/subject_" + str(i + 1) + ".csv", dataset, delimiter=",")

        print("Datasets exported to the output path.")
