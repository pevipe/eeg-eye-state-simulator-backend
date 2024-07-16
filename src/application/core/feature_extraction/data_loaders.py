import numpy as np
import os

from src.application.core.feature_extraction.ratio import Ratio
from src.application.core.feature_extraction.window import Window


class DataLoader:
    def __init__(self, dataset_path, fs, win_size):
        self.dataset_path = dataset_path
        self.fs = fs
        self.window_size = win_size

        #############
        # Constants #
        #############
        self.alpha_lowcut = 7.0
        self.alpha_highcut = 12.0
        self.beta_lowcut = 14.0
        self.beta_highcut = 25.0
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

    def load_dataset(self, overlap=0, normalize=False, pure_windows=False):
        files = [f for f in os.listdir(self.dataset_path) if f.endswith(".csv")]

        windows = []
        for f in files:
            data = ProvidedDatasetLoader._load_csv(self.dataset_path + "/" + f)
            if normalize:
                data = self._normalize(data)
            total_time = 600
            windows = windows + self._all_windowing(data, total_time, self.window_size, self.fs, overlap=overlap)

        dataset = []
        for w in windows:
            if pure_windows:
                if 0 < w.mean_targets < 1:
                    pass
                else:
                    dataset.append(Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                        self.fs).to_classificator_entry())
            dataset.append(Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                self.fs).to_classificator_entry())
        self.dataset = np.array(dataset)
        return self.dataset


class ProvidedDatasetIndividualLoader(ProvidedDatasetLoader):
    def __init__(self, dataset_path, output_path, fs, win_size):
        super().__init__(dataset_path, fs, win_size)
        self.dataset = None
        self.output_path = output_path

    def load_all_datasets(self, overlap=0, normalize=False, pure_windows=False):
        """
            Load the datasets from each individual subject. Returns the list with the datasets.
            Windowing is now done with the overlapping specified as parameter.
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
        else:
            os.makedirs(self.output_path)

        # If not, calculate the datasets and export them so then can be imported later
        for f in files:
            data = ProvidedDatasetLoader._load_csv(self.dataset_path + "/" + f)
            if normalize:
                data = self._normalize(data)
            # Take the samples with overlapping (sliding time window each 2 seconds)
            total_time = 600
            windows = self._all_windowing(data, total_time, self.window_size, self.fs, overlap=overlap)

            dataset = []
            if pure_windows:
                for w in windows:
                    if 0 < w.mean_targets < 1:  # Window does not contain unique state
                        pass
                    else:
                        dataset.append(Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut,
                                            self.beta_highcut, self.fs).to_classificator_entry())
                self.dataset.append(np.array(dataset))
            else:
                for w in windows:
                    dataset.append(Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                                        self.fs).to_classificator_entry())
                self.dataset.append(np.array(dataset))

        # Export the datasets so that time can be saved next time
        self.export_datasets()

        return self.dataset

    def load_single_dataset(self, subject_number, overlap=0, normalize=False, pure_windows=False):
        self.dataset = None  # Clear the dataset in case it contained something

        # Load the individual subject specified
        data = ProvidedDatasetLoader._load_csv(self.dataset_path + "/Sujeto_" + str(subject_number) + ".csv")
        if normalize:
            data = self._normalize(data)

        # Take the samples with overlapping (sliding time window each 2 seconds)
        total_time = 600
        windows = self._all_windowing(data, total_time, self.window_size, self.fs, overlap=overlap)

        dataset = []
        if pure_windows:
            for w in windows:
                if 0 < w.mean_targets < 1:  # Window does not contain unique state
                    pass
                else:
                    dataset.append(Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut,
                                        self.beta_highcut, self.fs).to_classificator_entry())
        else:
            for w in windows:
                dataset.append(
                    Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                         self.fs).to_classificator_entry())
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


class SingleDatasetLoader(ProvidedDatasetIndividualLoader):
    def __init__(self, dataset_path, output_path, win_size, exact_windows_path, fs=200):
        super().__init__(dataset_path, output_path, fs, win_size)
        self.exact_windows_indexes = None
        # self.exact_windows_path = output_path[:-4] + "exact_indexes.csv"
        self.exact_windows_path = exact_windows_path
        self.overlap = self.window_size - 2
        self.load_a_dataset(self.overlap)
        self.get_exact_windows_idx(self.overlap)
        self.export_dataset()

    def load_a_dataset(self, overlap, pure_windows=False):
        self.dataset = None  # Clear the dataset in case it contained something

        if os.path.exists(self.output_path):
            # load the dataset from the output path
            self.dataset = np.loadtxt(open(self.output_path, "rb"), delimiter=",")
            return self.dataset

        # Load the individual subject specified
        data = ProvidedDatasetLoader._load_csv(self.dataset_path)

        # Take the samples with overlapping (sliding time window each 2 seconds)
        total_time = 600
        windows = self._all_windowing(data, total_time, self.window_size, self.fs, overlap=overlap)

        dataset = []
        if pure_windows:
            for w in windows:
                if 0 < w.mean_targets < 1:  # Window does not contain unique state
                    pass
                else:
                    dataset.append(Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut,
                                        self.beta_highcut, self.fs).to_classificator_entry())
        else:
            for w in windows:
                dataset.append(
                    Ratio(w, self.alpha_lowcut, self.alpha_highcut, self.beta_lowcut, self.beta_highcut,
                         self.fs).to_classificator_entry())
        self.dataset = np.array(dataset)

        return self.dataset

    def export_dataset(self):
        # Save the dataset to CSV
        if os.path.exists(self.output_path):
            return
        np.savetxt(self.output_path, self.dataset, delimiter=",")

    def get_exact_windows_idx(self, overlap):
        self.exact_windows_indexes = []  # Clear the dataset in case it contained something

        if os.path.exists(self.exact_windows_path):
            # load the dataset from the output path
            self.exact_windows_indexes = np.loadtxt(open(self.exact_windows_path, "rb"), delimiter=",")
            return self.exact_windows_indexes

        # Load the individual subject specified
        data = ProvidedDatasetLoader._load_csv(self.dataset_path)

        # Take the samples with overlapping (sliding time window each 2 seconds)
        total_time = 600
        windows = self._all_windowing(data, total_time, self.window_size, self.fs, overlap=overlap)

        for i, w in enumerate(windows):
            if 0 < w.mean_targets < 1:  # Window does not contain unique state
                pass
            else:
                self.exact_windows_indexes.append(i)

        np.savetxt(self.exact_windows_path, self.exact_windows_indexes, delimiter=",")

        return self.exact_windows_indexes
