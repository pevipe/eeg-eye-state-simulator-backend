
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

    