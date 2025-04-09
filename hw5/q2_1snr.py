import numpy as np
import matplotlib.pyplot as plt


def quantize_2bit(signal, vref=1.0):
    """2位量化器，不包括偏移处理"""
    quant_unit = vref / 4
    digital_level = np.floor(signal / quant_unit)
    digital_level = np.clip(digital_level, 0, 3)
    return 4 * (signal - digital_level * quant_unit)


def quantize_2bit_with_offset(signal, vref=1.0, offset=0.0625):
    """在输入上加上偏移后进行2位量化，并将输出限制在0到1之间"""
    quantized = quantize_2bit(signal + offset, vref)
    return np.clip(quantized, 0, 1)


def generate_input_signal(fs, f_signal, N):
    """
    生成输入信号：全幅0.9V的正弦波，中心电平0.5V
    fs: 采样率
    f_signal: 输入信号频率
    N: 信号点数
    """
    time_vector = np.arange(N) / fs
    signal = 0.5 + 0.45 * np.sin(2 * np.pi * f_signal * time_vector)
    return time_vector, signal


def compute_normalized_fft(signal, fs):
    """
    对信号进行FFT，采用汉宁窗，并返回归一化后的单边功率谱及对应频率
    """
    window = np.hanning(len(signal))
    fft_result = np.fft.fft(signal * window)
    half_length = len(signal) // 2
    power_spectrum = np.abs(fft_result[:half_length]) ** 2
    freqs = np.fft.fftfreq(len(signal), d=1 / fs)[:half_length]
    # 归一化功率谱
    power_spectrum /= np.max(power_spectrum)
    return freqs, power_spectrum


def calculate_snr(freqs, power_spectrum, band_start=150e6, band_end=250e6):
    """
    在指定频段内（默认150MHz到250MHz）计算信噪比 (SNR, 单位：dB)
    """
    signal_mask = (freqs >= band_start) & (freqs <= band_end)
    signal_power = np.sum(power_spectrum[signal_mask])
    noise_power = np.sum(power_spectrum[~signal_mask])
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def plot_spectrum(freqs, power_spectrum, band_start=150, band_end=250):
    """
    绘制输出频谱图，X轴单位转换为MHz，
    并标识出信号所在频段（默认150-250MHz）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freqs * 1e-6, 10 * np.log10(power_spectrum + 1e-12), color='blue', linewidth=1.5,
             label='Output Spectrum (dB)')
    plt.axvspan(band_start, band_end, color='green', alpha=0.3,
                label=f'Signal Band ({band_start} - {band_end} MHz)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB, normalized)')
    plt.title('Output Spectrum and SNR Estimation')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 参数设置
    fs = 1e9  # 采样率 1 GHz
    f_in = 200e6  # 输入频率 200 MHz
    N = 8192  # 数据点数（2的幂，便于 FFT）

    # 生成输入时域信号
    t, vin = generate_input_signal(fs, f_in, N)

    # 经过带偏移的2位量化处理
    vout = quantize_2bit_with_offset(vin)

    # 对处理后的信号进行 FFT 并得到归一化单边功率谱
    freqs, spectrum = compute_normalized_fft(vout, fs)

    # 计算信噪比（信号频段：150MHz ~ 250MHz）
    snr = calculate_snr(freqs, spectrum, band_start=150e6, band_end=250e6)
    print(f"SNR = {snr:.2f} dB")

    # 绘制功率谱图
    plot_spectrum(freqs, spectrum, band_start=150, band_end=250)


if __name__ == "__main__":
    main()
