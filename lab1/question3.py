import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# 为了便于观察，将画图过程写成一个函数
def plot_spectrum(signal, Fs, title="Spectrum", apply_window=False):
    """
    计算并绘制给定信号的幅度谱（只显示0~Fs/2部分）。
    signal: 输入离散信号（长度N）
    Fs: 采样率
    title: 图标题
    apply_window: 是否应用Blackman窗
    """
    N = len(signal)

    # 是否乘以Blackman窗
    if apply_window:
        window = np.blackman(N)
        signal_win = signal * window
    else:
        signal_win = signal

    # 计算FFT
    Signal_f = fft(signal_win)

    # 频率轴（从0到Fs*(N-1)/N，但只绘0~Fs/2）
    freqs = fftfreq(N, d=1 / Fs)  # 生成从负频到正频的频率序列
    # 只取前半部分
    half_N = N // 2
    freqs_half = freqs[:half_N]
    amplitude_half = np.abs(Signal_f)[:half_N]

    # 作图
    plt.figure()
    plt.stem(freqs_half, amplitude_half)  # 移除了 use_line_collection=True
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.show()


##############################################################################
# a) x(t) = cos(2π·F·t), F=2MHz, Fs=5MHz, 64点DFT
##############################################################################
def part_a():
    # 参数设置
    F = 2e6  # 2 MHz
    Fs = 5e6  # 5 MHz
    N = 64

    # 时间序列
    t = np.arange(N) / Fs

    # 生成信号 x(t)
    x = np.cos(2 * np.pi * F * t)

    # 画出幅度谱
    plot_spectrum(x, Fs, title="Part (a): F=2MHz, Fs=5MHz, N=64")


##############################################################################
# b) y(t) = cos(2π·F1·t) + cos(2π·F2·t)
#    F1=200MHz, F2=400MHz, Fs=1GHz, 64点DFT
##############################################################################
def part_b():
    F1 = 200e6
    F2 = 400e6
    Fs = 1e9
    N = 64

    t = np.arange(N) / Fs
    y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

    plot_spectrum(y, Fs, title="Part (b): F1=200MHz, F2=400MHz, Fs=1GHz, N=64")


##############################################################################
# c) 将上一步Fs改为500MHz，观察频谱
##############################################################################
def part_c():
    F1 = 200e6
    F2 = 400e6
    Fs = 500e6
    N = 64

    t = np.arange(N) / Fs
    y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

    plot_spectrum(y, Fs, title="Part (c): F1=200MHz, F2=400MHz, Fs=500MHz, N=64")


##############################################################################
# d) 对信号应用Blackman窗，再做DFT并比较
#    这里分别对 Part (a) 的 x(t) 和 Part (b)/(c) 的 y(t) 做示例
##############################################################################
def part_d():
    # 先对 Part (a) 的信号应用 Blackman 窗
    F = 2e6
    Fs = 5e6
    N = 64
    t = np.arange(N) / Fs
    x = np.cos(2 * np.pi * F * t)
    plot_spectrum(x, Fs, title="Part (d): x(t) without window", apply_window=False)
    plot_spectrum(x, Fs, title="Part (d): x(t) with Blackman window", apply_window=True)

    # 再对 Part (b) 的信号应用 Blackman 窗
    F1 = 200e6
    F2 = 400e6
    Fs_b = 1e9
    N_b = 64
    t_b = np.arange(N_b) / Fs_b
    y_b = np.cos(2 * np.pi * F1 * t_b) + np.cos(2 * np.pi * F2 * t_b)
    plot_spectrum(y_b, Fs_b, title="Part (d): y(t) without window (Fs=1GHz)", apply_window=False)
    plot_spectrum(y_b, Fs_b, title="Part (d): y(t) with Blackman window (Fs=1GHz)", apply_window=True)


if __name__ == "__main__":
    # 依次执行
    part_a()
    part_b()
    part_c()
    part_d()