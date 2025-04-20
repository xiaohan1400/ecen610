import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq


def configure_system_parameters():
    """设置系统参数并返回相关配置"""
    sample_rate = 512e6  # 采样频率，单位Hz
    bandwidth = 200e6  # 信号带宽，单位Hz
    subcarrier_count = 128  # 子载波数量
    sample_points = 4096  # 采样点数
    return sample_rate, bandwidth, subcarrier_count, sample_points


def generate_time_base(samples: int, rate: float) -> np.ndarray:
    """生成时间轴数组"""
    return np.arange(samples) / rate


def create_subcarrier_frequencies(bandwidth: float, num_tones: int) -> np.ndarray:
    """生成子载波频率分布"""
    freq_lower = -bandwidth / 2
    freq_upper = bandwidth / 2
    return np.linspace(freq_lower, freq_upper, num_tones)


def produce_bpsk_symbols(num_symbols: int) -> np.ndarray:
    """生成BPSK调制符号（±1）"""
    return np.random.choice([-1, 1], size=num_symbols)


def synthesize_signal(time: np.ndarray, freqs: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """合成复数信号并返回实部"""
    composite_signal = np.zeros_like(time, dtype=complex)
    for idx, (symbol, freq) in enumerate(zip(symbols, freqs)):
        composite_signal += symbol * np.exp(2j * np.pi * freq * time)
    return np.real(composite_signal)


def compute_frequency_spectrum(data: np.ndarray, samples: int, rate: float) -> tuple:
    """计算信号的频谱"""
    fft_result = fftshift(fft(data))
    freq_scale = fftshift(fftfreq(samples, 1 / rate))
    return freq_scale, fft_result


def plot_signal_analysis(time_data: np.ndarray, signal_data: np.ndarray,
                         freq_data: np.ndarray, spectrum_data: np.ndarray):
    """绘制时域和频域图，仅显示前1μs的时域数据"""
    # 计算前1μs对应的采样点数
    time_limit = 1e-6  # 1微秒
    samples_to_show = int(time_limit * 512e6)  # 512 MHz × 1e-6 s = 512点

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 时域图（前1μs）
    ax1.plot(time_data[:samples_to_show] * 1e6, signal_data[:samples_to_show], color='blue')
    ax1.set_title('Signal in Time Domain (First 1μs)')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Amplitude')

    # 频域图
    magnitude_db = 20 * np.log10(np.abs(spectrum_data) + 1e-12)
    ax2.plot(freq_data / 1e6, magnitude_db, color='green')
    ax2.set_title('Signal Spectrum (DFT)')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Power (dB)')

    plt.tight_layout()
    plt.show()


def main():
    """主函数，生成并分析BPSK信号"""
    # 获取参数
    fs, bw, tones, N = configure_system_parameters()

    # 生成信号
    time_vector = generate_time_base(N, fs)
    subcarrier_freqs = create_subcarrier_frequencies(bw, tones)
    modulation_symbols = produce_bpsk_symbols(tones)
    transmitted_signal = synthesize_signal(time_vector, subcarrier_freqs, modulation_symbols)

    # 计算频谱
    frequency_axis, spectrum_result = compute_frequency_spectrum(transmitted_signal, N, fs)

    # 可视化
    plot_signal_analysis(time_vector, transmitted_signal, frequency_axis, spectrum_result)


if __name__ == "__main__":
    main()