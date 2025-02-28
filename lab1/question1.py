import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk

# 定义 FIR 和 IIR 滤波器的系数
# FIR 滤波器: H(z) = 1 + z^(-1) + z^(-2) + z^(-3) + z^(-4)
b_fir = [1, 1, 1, 1, 1]  # 分子系数
a_fir = [1]  # 分母系数 (FIR 滤波器的分母是 1)

# IIR 滤波器: H(z) = (1 + z^(-1)) / (1 - z^(-1))
b_iir = [1, 1]  # 分子系数
a_iir = [1, -1]  # 分母系数

# 创建一个大图，包含 2x2 的子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 计算并绘制 FIR 频率响应
w, h_fir = freqz(b_fir, a_fir, worN=1024)
axs[0, 0].plot(w / np.pi, 20 * np.log10(abs(h_fir)), label="FIR Filter")
axs[0, 0].set_title("FIR Frequency Response")
axs[0, 0].set_xlabel("Normalized Frequency (×π rad/sample)")
axs[0, 0].set_ylabel("Gain (dB)")
axs[0, 0].grid()
axs[0, 0].legend()

# 计算并绘制 IIR 频率响应
w, h_iir = freqz(b_iir, a_iir, worN=1024)
axs[0, 1].plot(w / np.pi, 20 * np.log10(abs(h_iir)), label="IIR Filter", color="b")
axs[0, 1].set_title("IIR Frequency Response")
axs[0, 1].set_xlabel("Normalized Frequency (×π rad/sample)")
axs[0, 1].set_ylabel("Gain (dB)")
axs[0, 1].grid()
axs[0, 1].legend()

# 计算 FIR 滤波器的零极点并绘制
zeros_fir, poles_fir, _ = tf2zpk(b_fir, a_fir)
axs[1, 0].scatter(np.real(zeros_fir), np.imag(zeros_fir), s=50, marker='o', label='Zeros', color='b')
axs[1, 0].scatter(np.real(poles_fir), np.imag(poles_fir), s=50, marker='x', label='Poles', color='r')
axs[1, 0].axhline(0, color='black', lw=0.5)
axs[1, 0].axvline(0, color='black', lw=0.5)
axs[1, 0].set_title("Pole-Zero Plot of FIR Filter")
axs[1, 0].set_xlabel("Real Part")
axs[1, 0].set_ylabel("Imaginary Part")
axs[1, 0].legend()
axs[1, 0].grid()

# 计算 IIR 滤波器的零极点并绘制
zeros_iir, poles_iir, _ = tf2zpk(b_iir, a_iir)
axs[1, 1].scatter(np.real(zeros_iir), np.imag(zeros_iir), s=50, marker='o', label='Zeros', color='b')
axs[1, 1].scatter(np.real(poles_iir), np.imag(poles_iir), s=50, marker='x', label='Poles', color='r')
axs[1, 1].axhline(0, color='black', lw=0.5)
axs[1, 1].axvline(0, color='black', lw=0.5)
axs[1, 1].set_title("Pole-Zero Plot of IIR Filter")
axs[1, 1].set_xlabel("Real Part")
axs[1, 1].set_ylabel("Imaginary Part")
axs[1, 1].legend()
axs[1, 1].grid()

# 调整子图之间的间距
plt.tight_layout()
plt.show()