import numpy as np


class Window:
    def __init__(self, global_data: np.ndarray, start_time: int, end_time: int, fs: int) -> None:
        # Defining the start and end time for the window (seconds)
        self.start_time = start_time
        self.end_time = end_time
        # Sampling frequency
        self.fs = fs

        targets = self.windowing(global_data[:, 2])

        # Sensor data and targets
        self.data_O1 = self.windowing(global_data[:, 0])
        self.data_O2 = self.windowing(global_data[:, 1])
        self.mean_targets = np.mean(targets)

    def windowing(self, data: np.ndarray) -> np.ndarray:
        ini = self.start_time * self.fs
        fin = self.end_time * self.fs

        return data[ini:fin]

    def __str__(self) -> str:
        mean_O1 = str(round(np.mean(self.data_O1), 4))
        mean_O2 = str(round(np.mean(self.data_O2), 4))
        mean_targets = str(round(self.mean_targets, 2))

        return ("Time: " + str(self.start_time) + " - " + str(self.end_time) + " s. O1: " + mean_O1 + " O2: " +
                mean_O2 + ". State: " + mean_targets + ".")
