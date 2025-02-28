import numpy as np
import matplotlib.pyplot as plt

fs = 401e6
fin = 200e6
num_bits = 6
L = 2**num_bits
A = 1
T = 1 / fs

def generate_sinewave(fin, fs, num_periods):
    num_samples = int(num_periods * fs / fin)
    t = np.arange(num_samples) * T
    x = A * np.sin(2 * np.pi * fin * t)
    return t, x

def quantize_signal(x, num_bits):
    L = 2**num_bits
    x_q = np.round(x * (L / 2 - 1)) / (L / 2 - 1)
    return x_q

def compute_snr(x, x_q):
    noise = x_q - x
    power_signal = np.mean(x**2)
    power_noise = np.mean(noise**2)
    snr = 10 * np.log10(power_signal / power_noise)
    return snr

def compute_psd(x, fs):
    X = np.fft.fft(x)
    X_mag = np.abs(X[:len(X)//2])**2
    freqs = np.fft.fftfreq(len(x), d=1/fs)[:len(X)//2]
    return freqs, X_mag

t_30, x_30 = generate_sinewave(fin, fs, 30)
x_q_30 = quantize_signal(x_30, num_bits)
snr_30 = compute_snr(x_30, x_q_30)
freqs_30, psd_30 = compute_psd(x_q_30, fs)

t_100, x_100 = generate_sinewave(fin, fs, 100)
x_q_100 = quantize_signal(x_100, num_bits)
snr_100 = compute_snr(x_100, x_q_100)
freqs_100, psd_100 = compute_psd(x_q_100, fs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(freqs_30 / 1e6, 10 * np.log10(psd_30 + 1e-10), label="30 periods")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.title("PSD for 30 Periods")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(freqs_100 / 1e6, 10 * np.log10(psd_100 + 1e-10), label="100 periods", color='r')
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.title("PSD for 100 Periods")
plt.legend()

plt.tight_layout()
plt.show()

print(f"SNR for 30 periods: {snr_30:.2f} dB")
print(f"SNR for 100 periods: {snr_100:.2f} dB")