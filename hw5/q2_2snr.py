import numpy as np
import matplotlib.pyplot as plt


def stage_2p5bit_noredundancy(x, vref=1.0):
    """
    2.5-bit流水线级（无冗余）量化处理
    :param x: 输入信号（已含偏移）
    :param vref: 参考电压，默认1.0V
    :return: 量化后的电平值
    """
    q = vref / 6
    d = np.floor(x / q)
    d = np.clip(d, 0, 5)
    return 6 * (x - d * q)


def generate_signals(fs, f_in, N, offset=0.0625, vref=1.0):
    """
    生成时域信号、应用2.5-bit量化处理，并返回输入与输出信号
    :param fs: 采样率 (Hz)
    :param f_in: 输入正弦波频率 (Hz)
    :param N: 采样点数
    :param offset: 量化前加入的偏移值
    :param vref: 参考电压
    :return: 时间向量、原始输入信号、量化输出信号
    """
    t = np.arange(N) / fs
    # Full-scale sine wave: 0.45V 正弦波加直流 0.5V（总幅度约0.9Vpp）
    vin = 0.5 + 0.45 * np.sin(2 * np.pi * f_in * t)
    # 加偏移后进入量化，并限制输出在[0,1]
    vout = np.clip(stage_2p5bit_noredundancy(vin + offset, vref), 0, 1)
    return t, vin, vout


def compute_fft(signal, fs, window_func=np.hanning):
    """
    对信号先加窗后进行FFT，返回单边归一化功率谱和对应频率
    :param signal: 输入时域信号
    :param fs: 采样率 (Hz)
    :param window_func: 窗函数生成函数，默认为np.hanning
    :return: 频率轴数组和归一化功率谱数组
    """
    N = len(signal)
    window = window_func(N)
    Vf = np.fft.fft(signal * window)
    half_N = N // 2
    power_spectrum = np.abs(Vf[:half_N]) ** 2
    freq = np.fft.fftfreq(N, d=1 / fs)[:half_N]
    # 归一化功率谱
    power_spectrum /= np.max(power_spectrum)
    return freq, power_spectrum


def estimate_snr(freq, power_spectrum, center=200e6, tolerance=50e6):
    """
    根据归一化功率谱估计信噪比(SNR)
    :param freq: 频率轴数组 (Hz)
    :param power_spectrum: 归一化功率谱
    :param center: 信号中心频率 (Hz)
    :param tolerance: 信号频带容限 (Hz)，信号频带为 [center-tolerance, center+tolerance]
    :return: SNR (dB), 信号下限频率, 信号上限频率
    """
    band_lower = center - tolerance
    band_upper = center + tolerance
    signal_mask = (freq >= band_lower) & (freq <= band_upper)
    P_signal = np.sum(power_spectrum[signal_mask])
    P_noise = np.sum(power_spectrum[~signal_mask])
    snr_dB = 10 * np.log10(P_signal / P_noise)
    return snr_dB, band_lower, band_upper


def plot_spectrum(freq, power_spectrum, band_lower, band_upper):
    """
    绘制输出功率谱图，并标注信号频带区域
    :param freq: 频率数组 (Hz)
    :param power_spectrum: 归一化功率谱
    :param band_lower: 信号频带下限 (Hz)
    :param band_upper: 信号频带上限 (Hz)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freq * 1e-6, 10 * np.log10(power_spectrum + 1e-12),
             label='Output Spectrum (dB)', linewidth=1.5)
    # 使用±50MHz标定信号带宽区域
    plt.axvspan(band_lower * 1e-6, band_upper * 1e-6, color='green', alpha=0.3,
                label='Signal Band (±50MHz)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB, normalized)')
    plt.title('Output Spectrum of 2.5-bit Stage (No Redundancy)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 参数设置
    fs = 1e9  # 采样率：1 GHz
    f_in = 200e6  # 输入正弦波频率：200 MHz
    N = 8192  # 采样点数（2的幂，便于FFT）

    # 生成信号：时域时间向量、输入信号、及经过2.5-bit量化输出信号
    t, vin, vout = generate_signals(fs, f_in, N, offset=0.0625, vref=1.0)

    # 对量化输出信号加窗并计算FFT，得到功率谱和频率轴
    freq, power_spectrum = compute_fft(vout, fs)

    # 计算信噪比 (SNR)：信号中心200MHz，带宽 ±50MHz
    snr_dB, band_lower, band_upper = estimate_snr(freq, power_spectrum,
                                                  center=200e6, tolerance=50e6)
    print(f"SNR for 2.5-bit (no redundancy): {snr_dB:.2f} dB")

    # 绘制输出功率谱图
    plot_spectrum(freq, power_spectrum, band_lower, band_upper)


if __name__ == "__main__":
    main()
