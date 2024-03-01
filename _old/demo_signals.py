import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfreqz

filename = "Sujeto_1.csv"


def leerArchivos(file):
    data = np.loadtxt(open(file, "rb"), delimiter=",")

    data = data[:, 1:3]
    # print(data.shape[0])
    return data


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


def enventanado(data, sec_ini, sec_fin, fs):
    ini = sec_ini*fs
    fin = sec_fin*fs

    return data[ini:fin, :]


def calc_ratio(signal_1, signal_2):
    pot_s1 = np.mean(np.power(signal_1, 2))
    pot_s2 = np.mean(np.power(signal_2, 2))

    ratio = pot_s1/pot_s2

    return ratio


if __name__ == '__main__':

    # Caragamos los datos y los normalizamos.
    data = leerArchivos(filename)
    data = data - np.mean(data)

    # Fecuencia de muestreo y frecuencias del filtro (in Hz).
    alpha_lowcut = 7.0
    alpha_highcut = 12.0
    fs = 200

    # Mostramos la respuesta del filtro para distintos orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        sos = butter_bandpass(7, 12, fs, order=order)
        w, h = sosfreqz(sos, fs=fs, worN=2000)
        plt.plot(w, abs(h), label="order = %d" % order)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Respuesta del filtro')

    # Filtramos la señal y mostramos los primeros 10 segundos.
    T = 10
    nsamples = T * fs
    t = np.arange(0, nsamples) / fs

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, data[0:nsamples, 1], label='Noisy signal channel 1')
    axs[0].set_title('Sin filtrar')

    y = butter_bandpass_filter(data[:, 1], alpha_lowcut, alpha_highcut, fs, order=6)
    axs[1].plot(t, y[0:fs * T], label='Filtered signal channel 1')
    axs[1].set_title('Filtrada')
    axs[1].set_xlabel('Tiempo (s)')

    # Mostramos las senales en el dominio de la frecuencia.
    N = np.shape(data)[0]
    X_data = np.fft.fft(data[:, 1])
    freqs = fs * np.arange(0, int(N / 2)) / N

    Y_data = np.fft.fft(y) / N
    fig2, sec_axs = plt.subplots(2, 1)
    sec_axs[0].plot(freqs, np.abs(X_data.real[0:int(N / 2)]))
    sec_axs[0].set_title('Sin Filtrar')
    sec_axs[1].plot(freqs, np.abs(Y_data.real[0:int(N / 2)]))
    sec_axs[1].set_title('Filtrada')
    sec_axs[1].set_xlabel('Frecuencia (Hz)')

    # Seleccionamos partes de la senal correspondiente a ojos abiertos y cerrados y calculamos el ratio entre
    # distintas bandas.
    abiertos = enventanado(data, 0, 60, fs)
    alpha_ab = butter_bandpass_filter(abiertos[:, 1], alpha_lowcut, alpha_highcut, 200)
    beta_ab = butter_bandpass_filter(abiertos[:, 1], 14, 25, 200)

    cerrados = enventanado(data, 61, 120, fs)
    alpha_cr = butter_bandpass_filter(cerrados[:, 1], 7, 13, 200)
    beta_cr = butter_bandpass_filter(cerrados[:, 1], 14, 25, 200)

    ratio_ab = calc_ratio(alpha_ab, beta_ab)
    ratio_cr = calc_ratio(alpha_cr, beta_cr)

    _, axs3 = plt.subplots()
    axs3.bar(['Abiertos', 'Cerrados'], [ratio_ab, ratio_cr])
    axs3.set_title('Ratio')

    plt.show()
