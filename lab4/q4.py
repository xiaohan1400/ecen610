import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.fft import fft, ifft

# 参数设置
amplitude = 0.5
data_rate = 10e9
bit_duration = 1 / data_rate
fs = 20e9
Ts = 1 / fs
samples_per_bit = int(fs / data_rate)
num_bits = 1024
t = np.arange(0, num_bits * bit_duration, Ts)

# 生成PRBS7 NRZ信号
def generate_prbs7(seed=0x7F, length=num_bits):
    state = seed & 0x7F
    prbs = []
    for _ in range(length):
        new_bit = ((state >> 6) ^ (state >> 5)) & 1
        state = ((state << 1) | new_bit) & 0x7F
        prbs.append(new_bit)
    return np.array(prbs)

bits = generate_prbs7()
nrz_signal = amplitude * np.repeat(bits, samples_per_bit)

# 应用通道失配
def apply_channel_mismatches(input_signal, time_skew=5e-12, offset=0.01, bw_ratio=0.9):
    delay_samples = int(time_skew / Ts)
    delayed_signal = np.roll(input_signal, delay_samples)
    delayed_signal[:delay_samples] = 0

    cutoff_freq_ch1 = data_rate
    cutoff_freq_ch2 = cutoff_freq_ch1 * bw_ratio
    Wn_ch1 = cutoff_freq_ch1 / (fs / 2)
    Wn_ch2 = cutoff_freq_ch2 / (fs / 2)

    b1, a1 = scipy.signal.butter(1, Wn_ch1, btype='low', analog=False, fs=fs)
    b2, a2 = scipy.signal.butter(1, Wn_ch2, btype='low', analog=False, fs=fs)

    filtered_ch1 = scipy.signal.lfilter(b1, a1, input_signal)
    filtered_ch2 = scipy.signal.lfilter(b2, a2, delayed_signal)

    return filtered_ch1, filtered_ch2 + offset

ch1, ch2 = apply_channel_mismatches(nrz_signal)

# 时间交织
ti_adc_output = np.zeros_like(ch1)
ti_adc_output[::2] = ch1[::2]
ti_adc_output[1::2] = ch2[1::2]

# 计算SNDR
def calculate_sndr(signal_data, fs, fundamental_bin):
    n = len(signal_data)
    fft_result = fft(signal_data) / n
    psd = np.abs(fft_result) ** 2
    signal_power = psd[fundamental_bin]
    total_power = np.sum(psd)
    noise_distortion_power = total_power - signal_power
    return 10 * np.log10(signal_power / noise_distortion_power)

# 校准实现
def calibrate_ti_adc(ti_output, calibration_bits=1000):
    offset_ch1 = np.mean(ti_output[::2][:calibration_bits])
    offset_ch2 = np.mean(ti_output[1::2][:calibration_bits])
    offset_corrected = ti_output.copy()
    offset_corrected[::2] -= offset_ch1
    offset_corrected[1::2] -= offset_ch2

    ch1_signal = offset_corrected[::2]
    ch2_signal = offset_corrected[1::2]
    corr = np.correlate(ch1_signal, ch2_signal, mode='full')
    delay = np.argmax(corr) - (len(ch2_signal) - 1)

    x_original = np.arange(len(ch2_signal))
    x_new = x_original - delay / 2
    x_new = np.clip(x_new, x_original[0], x_original[-1])
    ch2_corrected = np.interp(x_new, x_original, ch2_signal)

    time_corrected = offset_corrected.copy()
    time_corrected[1::2] = ch2_corrected

    fft_ch1 = fft(time_corrected[::2])
    fft_ch2 = fft(time_corrected[1::2])
    freq_response_ratio = fft_ch1 / (fft_ch2 + 1e-12)
    freq_response_ratio[np.abs(fft_ch2) < 0.01] = 1

    calibrated_output = time_corrected.copy()
    fft_ch2_corrected = fft_ch2 * freq_response_ratio
    ch2_calibrated = np.real(ifft(fft_ch2_corrected))
    if len(ch2_calibrated) == len(calibrated_output[1::2]):
        calibrated_output[1::2] = ch2_calibrated

    return {
        'fully_corrected': calibrated_output,
        'measured_offset': (offset_ch1, offset_ch2),
        'measured_delay': delay
    }

# 执行校准
calibration_results = calibrate_ti_adc(ti_adc_output)

# 可视化
plt.figure(figsize=(14, 10))

n = len(ti_adc_output)
freq = np.fft.fftfreq(n, Ts)
fft_original = 20 * np.log10(np.abs(fft(nrz_signal) / n) + 1e-12)
fft_tiadc = 20 * np.log10(np.abs(fft(ti_adc_output) / n) + 1e-12)
fft_corrected = 20 * np.log10(np.abs(fft(calibration_results['fully_corrected']) / n) + 1e-12)

plt.subplot(2, 1, 1)
plt.semilogx(freq[:n // 2], fft_original[:n // 2], 'b-', label='Original')
plt.semilogx(freq[:n // 2], fft_tiadc[:n // 2], 'r--', label='With Mismatches')
plt.semilogx(freq[:n // 2], fft_corrected[:n // 2], 'g-.', label='Calibrated')
plt.title('Frequency Domain Analysis')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

# 时域只显示前 100 个点
plt.subplot(2, 1, 2)
plt.plot(t[:100], ti_adc_output[:100], 'r--', label='Before Calibration')
plt.plot(t[:100], calibration_results['fully_corrected'][:100], 'g-', label='After Calibration')
plt.title('Time Domain Comparison (First 100 Points)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 性能评估
original_sndr = calculate_sndr(nrz_signal, fs, int(data_rate / fs * n))
mismatched_sndr = calculate_sndr(ti_adc_output, fs, int(data_rate / fs * n))
corrected_sndr = calculate_sndr(calibration_results['fully_corrected'], fs, int(data_rate / fs * n))

print(f"SNDR Results:")
print(f"Original: {original_sndr:.2f} dB")
print(f"With Mismatches: {mismatched_sndr:.2f} dB")
print(f"After Calibration: {corrected_sndr:.2f} dB")
print(f"\nMeasured Parameters:")
print(f"Offsets: Ch1={calibration_results['measured_offset'][0]:.4f}V, Ch2={calibration_results['measured_offset'][1]:.4f}V")
print(f"Time Delay: {calibration_results['measured_delay'] * Ts * 1e12:.2f} ps")