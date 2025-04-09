import numpy as np
import matplotlib.pyplot as plt


def redundant_2p5bit_stage(signal, vref=1.0):
    """
    带冗余的 2.5-bit 量化处理
    :param signal: 输入信号（已加偏移）
    :param vref: 参考电压（默认为 1.0V）
    :return: 量化后的信号
    """
    # 量化步长（8 个区间）
    step = vref / 8
    # 定义分割阈值
    thresholds = np.array([0.0, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 1.0])
    # 根据阈值分组（注意 np.digitize 返回的是 1 至 len(thresholds) 的索引）
    indices = np.digitize(signal, thresholds) - 1
    # 限制索引在有效范围内
    indices = np.clip(indices, 0, 6)
    # 计算量化误差并输出放大后的结果
    return 6 * (signal - indices * step)


def create_signal(sampling_rate, sine_freq, samples, amplitude=0.45, dc_offset=0.5):
    """
    生成正弦输入信号：全幅约 0.9V（中心电平 0.5V）
    :param sampling_rate: 采样率（Hz）
    :param sine_freq: 正弦波频率（Hz）
    :param samples: 样本数
    :param amplitude: 正弦幅度（默认为 0.45V，对应 0.9V_pp）
    :param dc_offset: 直流偏置（默认为 0.5V）
    :return: 时间向量和生成的输入信号
    """
    t_vector = np.arange(samples) / sampling_rate
    sine_wave = dc_offset + amplitude * np.sin(2 * np.pi * sine_freq * t_vector)
    return t_vector, sine_wave


def process_signal(signal, stage_func, offset=0.0625, clip_min=0, clip_max=1, **kwargs):
    """
    对输入信号先加偏移，再经过量化处理，最后裁剪到指定范围
    :param signal: 原始输入信号
    :param stage_func: 量化函数（如 redundant_2p5bit_stage）
    :param offset: 加入量化前的偏移量（默认为 0.0625）
    :param clip_min: 裁剪下限
    :param clip_max: 裁剪上限
    :param kwargs: 传递给量化函数的其它参数，例如 vref
    :return: 量化并裁剪后的输出信号
    """
    quantized_output = stage_func(signal + offset, **kwargs)
    return np.clip(quantized_output, clip_min, clip_max)


def compute_fft(signal, sampling_rate, window_func=np.hanning):
    """
    对信号进行加窗和 FFT 计算，并返回单边归一化的功率谱和对应频率轴
    :param signal: 输入时域信号
    :param sampling_rate: 采样率（Hz）
    :param window_func: 窗函数生成函数（默认为 np.hanning）
    :return: 频率数组和归一化的功率谱
    """
    N = len(signal)
    window = window_func(N)
    fft_result = np.fft.fft(signal * window)
    half_N = N // 2
    power_spec = np.abs(fft_result[:half_N]) ** 2
    freq_axis = np.fft.fftfreq(N, d=1 / sampling_rate)[:half_N]
    # 归一化功率谱（防止除以 0）
    if np.max(power_spec) != 0:
        power_spec /= np.max(power_spec)
    return freq_axis, power_spec


def estimate_snr(freq, power_spec, center_freq=200e6, band_margin=25e6):
    """
    在指定频带内估计信噪比 (SNR)
    :param freq: 频率数组（Hz）
    :param power_spec: 归一化功率谱
    :param center_freq: 信号中心频率（Hz），默认为 200 MHz
    :param band_margin: 信号带宽余量（Hz），默认为 ±25MHz
    :return: 计算得到的 SNR（dB），信号频带下限和上限
    """
    lower_bound = center_freq - band_margin
    upper_bound = center_freq + band_margin
    signal_mask = (freq >= lower_bound) & (freq <= upper_bound)
    signal_power = np.sum(power_spec[signal_mask])
    noise_power = np.sum(power_spec[~signal_mask])
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db, lower_bound, upper_bound


def plot_spectrum(freq, power_spec, band_limits, color_band='green'):
    """
    绘制功率谱，并高亮显示信号所在频带
    :param freq: 频率数组（Hz）
    :param power_spec: 归一化功率谱
    :param band_limits: 信号频带（下限， 上限）
    :param color_band: 信号频带高亮颜色
    """
    plt.figure(figsize=(10, 6))
    # 转换为 dB 单位（防止 log(0) 加上小量 1e-12）
    plt.plot(freq * 1e-6, 10 * np.log10(power_spec + 1e-12),
             label='Output Spectrum (dB)', linewidth=1.5)
    # 修改此处的标签为 ±50 MHz
    plt.axvspan(band_limits[0] * 1e-6, band_limits[1] * 1e-6,
                color=color_band, alpha=0.3,
                label='Signal Band (±50 MHz)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB, normalized)')
    plt.title('Output Spectrum of 2.5-bit Stage (With Redundancy)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 参数设置
    sampling_rate = 1e9  # 采样率 1 GHz
    sine_frequency = 200e6  # 输入信号频率 200 MHz
    sample_count = 8192  # 样本数（2 的幂，用于 FFT）

    # 生成输入时域信号和对应正弦波
    time_vec, vin = create_signal(sampling_rate, sine_frequency, sample_count)

    # 通过带冗余的 2.5-bit 量化处理（加偏移后进行量化）
    vout = process_signal(vin, redundant_2p5bit_stage, offset=0.0625, vref=1.0)

    # 计算 FFT（加窗处理）得到归一化的功率谱和频率轴
    freq_axis, power_spectrum = compute_fft(vout, sampling_rate)

    # 估计信噪比（信号频带：200MHz ±50MHz，即 [150, 250] MHz）
    snr, low_freq, high_freq = estimate_snr(freq_axis, power_spectrum,
                                            center_freq=200e6, band_margin=50e6)
    print(f"SNR for 2.5-bit (with redundancy): {snr:.2f} dB")

    # 绘制频谱图并标注信号频带
    plot_spectrum(freq_axis, power_spectrum, band_limits=(low_freq, high_freq))


if __name__ == "__main__":
    main()
