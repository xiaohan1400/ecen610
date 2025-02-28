import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a 2 MHz sine wave with a sampling rate of 5 MHz
Fs = 5e6  # Sampling rate of 5 MHz
T = 1e-3  # Signal duration of 1 ms
t = np.arange(0, T, 1/Fs)  # Time axis for sampling points
f_signal = 2e6  # Signal frequency of 2 MHz
A_signal = 1  # Signal amplitude of 1V
signal = A_signal * np.sin(2 * np.pi * f_signal * t)

# 2. Generate a high sampling rate original signal (background signal)
Fs_high = 500e6  # High sampling rate of 500 MHz
t_high = np.arange(0, T, 1/Fs_high)  # High sampling rate time axis
original_signal = A_signal * np.sin(2 * np.pi * f_signal * t_high)  # Original signal

# 3. Calculate the required noise variance
SNR_dB = 50  # Target SNR of 50 dB
P_signal = (A_signal ** 2) / 2  # Sine wave power = A^2 / 2
P_noise = P_signal / (10**(SNR_dB / 10))  # Calculate noise power based on SNR
sigma_noise = np.sqrt(P_noise)  # Standard deviation of Gaussian noise

# 4. Generate uniform distribution noise and add it to the signal
a_uniform = np.sqrt(3) * sigma_noise  # Uniform distribution range [-a, a]
noise = np.random.uniform(-a_uniform, a_uniform, size=t.shape)
noisy_signal = signal + noise

# 5. Only take the first 10 microseconds of data
t_cutoff = 10e-6  # 10 microseconds
t_selected = t[t < t_cutoff]  # Select time points within the first 10 microseconds
signal_selected = signal[t < t_cutoff]  # Select corresponding signal values
noisy_signal_selected = noisy_signal[t < t_cutoff]  # Select corresponding noisy signal values
t_high_selected = t_high[t_high < t_cutoff]  # Select high sampling rate time points
original_signal_selected = original_signal[t_high < t_cutoff]  # Select corresponding original signal values

# 6. Plot the time-domain signal waveforms
plt.figure(figsize=(10, 6))  # Adjust the overall figure size

# First subplot: Comparison before adding noise and before sampling
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.plot(t_high_selected, original_signal_selected, label='Original Signal (Background)', color='lightgray', linestyle='-', linewidth=2)
plt.plot(t_selected, signal_selected, label='Sampled Signal (Points)', color='green', marker='o', linestyle='None')
plt.title('Comparison of Original Signal and Sampled Signal (Before Adding Noise, First 10 μs)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)

# Second subplot: Comparison after adding noise and before sampling
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.plot(t_high_selected, original_signal_selected, label='Original Signal (Background)', color='lightgray', linestyle='-', linewidth=2)
plt.plot(t_selected, noisy_signal_selected, label='Noisy Signal (Points)', color='red', marker='o', linestyle='None')
plt.title('Comparison of Original Signal and Noisy Signal (After Adding Noise, First 10 μs)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)

# Display the figure
plt.tight_layout()  # Automatically adjust subplot spacing
plt.show()

# 5. Calculate and plot the Power Spectral Density (PSD)
# Calculate DFT of the noisy signal
n = len(noisy_signal)  # Signal length
frequencies = np.fft.fftfreq(n, d=1/Fs)  # Calculate frequency array
fft_values = np.fft.fft(noisy_signal)  # Calculate DFT

# Calculate Power Spectral Density
Pxx = np.abs(fft_values) ** 2 / n  # Power Spectral Density
f = frequencies[:n // 2]  # Only take the positive frequency part
Pxx = Pxx[:n // 2]  # Only take the Power Spectral Density corresponding to positive frequencies

# Calculate DFT of the original signal
n = len(signal)  # Signal length
frequencies = np.fft.fftfreq(n, d=1/Fs)  # Calculate frequency array
fft_values = np.fft.fft(signal)  # Calculate DFT

# Calculate Power Spectral Density
F_ori = np.abs(fft_values) ** 2 / n  # Power Spectral Density
f = frequencies[:n // 2]  # Only take the positive frequency part
F_ori = F_ori[:n // 2]  # Only take the Power Spectral Density corresponding to positive frequencies

# Output the original PSD and the PSD with noise
plt.figure(figsize=(10, 6))  # Adjust the overall figure size

# First plot: PSD of the noisy signal
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.semilogy(f, Pxx, color='g')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density (PSD) of Noisy Signal')
plt.grid(True)

# Second plot: PSD of the original signal
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.semilogy(f, F_ori, color='r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Original Power Spectral Density')
plt.title('Original Power Spectral Density (PSD)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Calculate signal power and noise power from PSD and verify SNR
signal_band = (f > 1.9e6) & (f < 2.1e6)  # Select frequencies near the signal frequency
P_signal_est = np.sum(Pxx[signal_band])  # Estimate signal power
P_noise_est = np.sum(Pxx[~signal_band])  # Estimate noise power
SNR_est_dB = 10 * np.log10(P_signal_est / P_noise_est)

print(f"Theoretical SNR: {SNR_dB} dB")
print(f"SNR calculated from PSD: {SNR_est_dB:.2f} dB")

# 7. Calculate uniform distribution noise variance
sigma_uniform = np.sqrt(3 * P_noise)
a_uniform = np.sqrt(3) * sigma_noise  # Uniform distribution range [-a, a]

print(f"Uniform noise range: [-{a_uniform:.5f}, {a_uniform:.5f}]")