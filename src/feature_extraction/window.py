import numpy as np


class Window:
    def __init__(self, global_data, start_time, end_time, fs):
        # Defining the start and end time for the window (seconds)
        self.start_time = start_time
        self.end_time = end_time
        # Sampling frequency
        self.fs = fs

        targets = self.windowing(global_data[:, 2])

        self.data_O1 = [self.windowing(global_data[:, 0]), targets]
        self.data_O2 = [self.windowing(global_data[:, 1]), targets]

    def windowing(self, data):
        ini = self.start_time * self.fs
        fin = self.end_time * self.fs

        return data[ini:fin]

    def __str__(self):
        mean_O1 = str(round(np.mean(self.data_O1[0]), 4))
        mean_O2 = str(round(np.mean(self.data_O2[0]), 4))
        mean_targets = str(round(np.mean(self.data_O1[1]), 1))

        return ("Time: " + str(self.start_time) + " - " + str(self.end_time) + " s. O1: " + mean_O1 + " O2: " +
                mean_O2 + ". State: " + mean_targets + ".")
