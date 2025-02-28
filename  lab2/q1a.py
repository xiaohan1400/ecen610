import numpy as np
import matplotlib.pyplot as plt

Fs = 5e6
T = 1e-3
t = np.arange(0, T, 1/Fs)
f_signal = 2e6
A_signal = 1
signal = A_signal * np.sin(2 * np.pi * f_signal * t)

Fs_high = 500e6
t_high = np.arange(0, T, 1/Fs_high)
original_signal = A_signal * np.sin(2 * np.pi * f_signal * t_high)

SNR_dB = 50
P_signal = (A_signal ** 2) / 2
P_noise = P_signal / (10**(SNR_dB / 10))
sigma_noise = np.sqrt(P_noise)

noise = np.random.normal(0, sigma_noise, size=t.shape)
noisy_signal = signal + noise

t_cutoff = 10e-6
t_selected = t[t < t_cutoff]
signal_selected = signal[t < t_cutoff]
noisy_signal_selected = noisy_signal[t < t_cutoff]
t_high_selected = t_high[t_high < t_cutoff]
original_signal_selected = original_signal[t_high < t_cutoff]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_high_selected, original_signal_selected, label='Original Signal', color='lightgray', linestyle='-', linewidth=2)
plt.plot(t_selected, signal_selected, label='Sampled Signal', color='green', marker='o', linestyle='None')
plt.title('Original vs Sampled Signal (First 10 μs)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_high_selected, original_signal_selected, label='Original Signal', color='lightgray', linestyle='-', linewidth=2)
plt.plot(t_selected, noisy_signal_selected, label='Noisy Signal', color='red', marker='o', linestyle='None')
plt.title('Original vs Noisy Signal (First 10 μs)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

n = len(noisy_signal)
frequencies = np.fft.fftfreq(n, d=1/Fs)
fft_values = np.fft.fft(noisy_signal)
Pxx = np.abs(fft_values) ** 2 / n
f = frequencies[:n // 2]
Pxx = Pxx[:n // 2]

n = len(signal)
frequencies = np.fft.fftfreq(n, d=1/Fs)
fft_values = np.fft.fft(signal)
F_ori = np.abs(fft_values) ** 2 / n
f = frequencies[:n // 2]
F_ori = F_ori[:n // 2]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogy(f, Pxx, color='g')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('PSD of Noisy Signal')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(f, F_ori, color='r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('PSD of Original Signal')
plt.grid(True)

plt.tight_layout()
plt.show()

signal_band = (f > 1.9e6) & (f < 2.1e6)
P_signal_est = np.sum(Pxx[signal_band])
P_noise_est = np.sum(Pxx[~signal_band])
SNR_est_dB = 10 * np.log10(P_signal_est / P_noise_est)

print(f"Theoretical SNR: {SNR_dB} dB")
print(f"SNR from PSD: {SNR_est_dB:.2f} dB")

sigma_uniform = np.sqrt(3 * P_noise)
a_uniform = np.sqrt(3) * sigma_noise

print(f"Gaussian noise std: {sigma_noise:.5f}")
print(f"Uniform noise range: [-{a_uniform:.5f}, {a_uniform:.5f}]")