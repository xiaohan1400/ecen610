import numpy as np
import matplotlib.pyplot as plt

Fs = 5e6
T = 1e-3
t = np.arange(0, T, 1 / Fs)
f_signal = 2e6
A_signal = 1
signal = A_signal * np.sin(2 * np.pi * f_signal * t)

SNR_dB = 50
P_signal = (A_signal ** 2) / 2
P_noise = P_signal / (10 ** (SNR_dB / 10))
sigma_noise = np.sqrt(P_noise)
noise = np.random.normal(0, sigma_noise, size=t.shape)
noisy_signal = signal + noise

def apply_window(signal, window_type):
    windows = {
        'hanning': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman
    }
    if window_type not in windows:
        raise ValueError("Unknown window type")
    return signal * windows[window_type](len(signal))

def calculate_psd(signal, Fs, window_type):
    windowed_signal = apply_window(signal, window_type)
    n = len(windowed_signal)
    frequencies = np.fft.fftfreq(n, d=1 / Fs)
    fft_values = np.fft.fft(windowed_signal)
    window_power = np.sum(windowed_signal ** 2)
    Pxx = np.abs(fft_values) ** 2 / (n * window_power)
    f = frequencies[:n // 2]
    Pxx = Pxx[:n // 2]
    return f, Pxx

window_types = ['hanning', 'hamming', 'blackman']
plt.figure(figsize=(12, 8))

for i, window_type in enumerate(window_types):
    f, Pxx = calculate_psd(noisy_signal, Fs, window_type)
    plt.subplot(3, 1, i + 1)
    plt.semilogy(f, Pxx, label=f'{window_type.capitalize()} Window')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Power Spectral Density (PSD) with {window_type.capitalize()} Window')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

def calculate_snr(Pxx, f, signal_band):
    P_signal_est = np.sum(Pxx[signal_band])
    P_noise_est = np.sum(Pxx[~signal_band])
    return 10 * np.log10(P_signal_est / P_noise_est)

for window_type in window_types:
    f, Pxx = calculate_psd(noisy_signal, Fs, window_type)
    signal_band = (f > 1.9e6) & (f < 2.1e6)
    SNR_est_dB = calculate_snr(Pxx, f, signal_band)
    print(f"{window_type.capitalize()} Window - Estimated SNR from PSD: {SNR_est_dB:.2f} dB")