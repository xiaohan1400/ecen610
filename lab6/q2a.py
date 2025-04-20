import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 参数设置
fs = 500e6         # 采样率 500 MHz
fin = 200e6        # 输入频率 200 MHz
bits = 13          # ADC 分辨率
N = 8192           # 采样点数，足够长以保证频谱分辨率

# 时间轴
t = np.arange(N) / fs

# 输入信号（单位幅度正弦波）
x_in = 0.9 * np.sin(2 * np.pi * fin * t)

# 理想 pipeline ADC 量化（这里直接用理想量化器）
full_scale = 1.0  # 假设输入范围 ±1V
q_levels = 2**bits
step = 2 * full_scale / q_levels
x_adc = np.round(x_in / step) * step
x_adc = np.clip(x_adc, -full_scale, full_scale - step)  # 保持在范围内

# 绘图：输入和输出波形
plt.figure(figsize=(12, 5))
plt.plot(t[:200], x_in[:200], label='Analog Input')
plt.plot(t[:200], x_adc[:200], label='ADC Output', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.title('Input and ADC Output (Zoomed In)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 频谱分析
window = np.hanning(N)
X = fft(x_adc * window)  # 加窗减少旁瓣
f = fftfreq(N, d=1/fs)

# 只看正频率部分
X_half = X[:N//2]
f_pos = f[:N//2]
X_magnitude = 20 * np.log10(np.abs(X_half))

# SNR计算（175MHz 到 220MHz 视为信号）
sig_band = (f_pos >= 199e6) & (f_pos <= 200.97e6)
noise_band = (f_pos > 0) & ~sig_band

signal_power = np.sum(np.abs(X_half[sig_band])**2)
noise_power = np.sum(np.abs(X_half[noise_band])**2)
snr = 10 * np.log10(signal_power / noise_power)

# 绘图：频谱
plt.figure(figsize=(12, 5))
plt.plot(f_pos / 1e6, X_magnitude)
plt.axvspan(199, 201, color='orange', alpha=0.3, label='Signal Band')
plt.title(f'FFT of ADC Output (SNR ≈ {snr:.2f} dB)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(snr)