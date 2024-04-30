import numpy as np
from scipy.signal import butter, sosfilt

from src.application.core.feature_extraction.window import Window


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band',  output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def calculate_rate(signal_1, signal_2):
    pot_s1 = np.mean(np.power(signal_1, 2))
    pot_s2 = np.mean(np.power(signal_2, 2))

    ratio = pot_s1/pot_s2

    return ratio


class Rate:
    def __init__(self, window: Window, alpha_lowcut, alpha_highcut, beta_lowcut, beta_highcut, fs):
        self.window = window

        # 1. Apply butter bandpass filter
        alpha_o1 = butter_bandpass_filter(self.window.data_O1, alpha_lowcut, alpha_highcut, fs)
        beta_o1 = butter_bandpass_filter(self.window.data_O1, beta_lowcut, beta_highcut, fs)
        alpha_o2 = butter_bandpass_filter(self.window.data_O2, alpha_lowcut, alpha_highcut, fs)
        beta_o2 = butter_bandpass_filter(self.window.data_O2, beta_lowcut, beta_highcut, fs)

        # 2. Calculate rates for sensors O1 and O2
        self.rate_O1 = calculate_rate(alpha_o1, alpha_o2)
        self.rate_O2 = calculate_rate(beta_o1, beta_o2)

    def __str__(self):
        o1 = str(round(self.rate_O1, 3))
        o2 = str(round(self.rate_O2, 3))
        return ("Rates for window on period " + str(self.window.start_time) + "s to " + str(self.window.end_time) +
                "s is: O1 = " + o1 + ", O2 = " + o2)

    def to_classificator_entry(self):
        return [self.rate_O1, self.rate_O2, np.rint(self.window.mean_targets)]
