# Copyright 2024 Pelayo Vieites Pérez
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

import numpy as np
from scipy.signal import butter, sosfilt

from src.domain.feature_extraction.window import Window


def _butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 5) -> tuple:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band',  output='sos')
    return sos


def _butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    sos = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def _calculate_ratio(signal_1: np.ndarray, signal_2: np.ndarray) -> float:
    pot_s1 = np.mean(np.power(signal_1, 2))
    pot_s2 = np.mean(np.power(signal_2, 2))

    ratio = pot_s1/pot_s2

    return ratio


class Ratio:
    def __init__(self, window: Window, alpha_lowcut: float, alpha_highcut: float,
                 beta_lowcut: float, beta_highcut: float, fs: int) -> None:
        self.window = window

        # 1. Apply butter bandpass filter
        alpha_o1 = _butter_bandpass_filter(self.window.data_O1, alpha_lowcut, alpha_highcut, fs)
        beta_o1 = _butter_bandpass_filter(self.window.data_O1, beta_lowcut, beta_highcut, fs)
        alpha_o2 = _butter_bandpass_filter(self.window.data_O2, alpha_lowcut, alpha_highcut, fs)
        beta_o2 = _butter_bandpass_filter(self.window.data_O2, beta_lowcut, beta_highcut, fs)

        # 2. Calculate ratios for sensors O1 and O2
        self.ratio_O1 = _calculate_ratio(alpha_o1, alpha_o2)
        self.ratio_O2 = _calculate_ratio(beta_o1, beta_o2)

    def __str__(self) -> str:
        o1 = str(round(self.ratio_O1, 3))
        o2 = str(round(self.ratio_O2, 3))
        return ("Ratios for window on period " + str(self.window.start_time) + "s to " + str(self.window.end_time) +
                "s is: O1 = " + o1 + ", O2 = " + o2)

    def to_classificator_entry(self) -> list:
        """
        Obtain the list that conforms the classificator entry for the time window.
        :return: list with ratio in each of the sensors and label associated to it.
        """
        return [self.ratio_O1, self.ratio_O2, np.rint(self.window.mean_targets)]
