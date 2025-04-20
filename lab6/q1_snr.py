import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def configure_simulation_parameters():
    """配置模拟参数并返回相关设置"""
    sampling_rate = 500e6  # 采样频率，单位：Hz
    signal_frequency = 200e6  # 信号频率，单位：Hz
    adc_resolution = 13  # ADC位数
    sample_count = 8192  # 采样点总数
    return sampling_rate, signal_frequency, adc_resolution, sample_count


def generate_time_array(samples: int, rate: float) -> np.ndarray:
    """生成时间轴数组"""
    return np.arange(samples) / rate


def create_input_signal(time: np.ndarray, freq: float, amplitude: float = 0.9) -> np.ndarray:
    """生成正弦输入信号"""
    return amplitude * np.sin(2 * np.pi * freq * time)


def quantize_signal(signal: np.ndarray, resolution: int, max_range: float = 1.0) -> np.ndarray:
    """模拟ADC量化过程"""
    levels = 2 ** resolution
    quantization_step = 2 * max_range / levels
    quantized = np.round(signal / quantization_step) * quantization_step
    return np.clip(quantized, -max_range, max_range - quantization_step)


def plot_waveforms(time_data: np.ndarray, original: np.ndarray, quantized: np.ndarray):
    """绘制输入和量化后的波形对比图"""
    plt.figure(figsize=(12, 5))
    plt.plot(time_data[:50], original[:50], label='Original Signal', color='blue')
    plt.plot(time_data[:50], quantized[:50], label='Quantized Signal', linestyle='--', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (V)')
    plt.title('Signal  ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_spectrum(data: np.ndarray, samples: int, sample_rate: float) -> tuple:
    """计算信号的频谱"""
    window_function = np.hanning(samples)
    fft_result = fft(data * window_function)
    frequencies = fftfreq(samples, d=1 / sample_rate)
    return fft_result[:samples // 2], frequencies[:samples // 2]


def calculate_snr(magnitude: np.ndarray, freqs: np.ndarray,
                  signal_range: tuple = (199e6, 201e6)) -> float:
    """计算信噪比"""
    signal_mask = (freqs >= signal_range[0]) & (freqs <= signal_range[1])
    noise_mask = (freqs > 0) & ~signal_mask

    signal_energy = np.sum(np.abs(magnitude[signal_mask]) ** 2)
    noise_energy = np.sum(np.abs(magnitude[noise_mask]) ** 2)
    return 10 * np.log10(signal_energy / noise_energy)


def visualize_spectrum(frequencies: np.ndarray, magnitude: np.ndarray, snr_value: float):
    """绘制频谱图并标注SNR"""
    # plt.figure(figsize=(12, 5))
    # plt.plot(frequencies / 1e6, 20 * np.log10(np.abs(magnitude)), color='green')
    # plt.axvspan(199, 201, color='orange', alpha=0.3, label='Signal Region')
    # plt.title(f'Frequency Spectrum (SNR: {snr_value:.2f} dB)')
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Power (dB)')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


def main():
    """主函数，执行ADC模拟和分析"""
    # 获取参数
    fs, fin, bits, N = configure_simulation_parameters()

    # 生成信号
    time_axis = generate_time_array(N, fs)
    input_wave = create_input_signal(time_axis, fin)
    adc_output = quantize_signal(input_wave, bits)

    # 绘制波形
    plot_waveforms(time_axis, input_wave, adc_output)

    # 频谱分析
    spectrum, freq_axis = compute_spectrum(adc_output, N, fs)
    snr_result = calculate_snr(spectrum, freq_axis)

    # 显示结果
    visualize_spectrum(freq_axis, spectrum, snr_result)
    print(f"SNR: {snr_result:.2f} dB")


if __name__ == "__main__":
    main()