import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


def setup_simulation_parameters():
    """初始化模拟参数"""
    sample_rate = 500e6  # 采样频率 (Hz)
    input_freq = 200e6  # 输入信号频率 (Hz)
    sample_size = 8192  # 总采样点数
    full_scale_voltage = 1.0  # ADC满量程电压
    return sample_rate, input_freq, sample_size, full_scale_voltage


def generate_time_vector(samples: int, rate: float) -> np.ndarray:
    """生成时间序列"""
    return np.arange(samples) / rate


def produce_input_waveform(time: np.ndarray, frequency: float, amplitude: float = 0.5) -> np.ndarray:
    """生成输入正弦信号"""
    return amplitude * np.sin(2 * np.pi * frequency * time)


def apply_quantization(signal: np.ndarray, bit_depth: int = 2, vfs: float = 1.0) -> np.ndarray:
    """执行理想量化"""
    quantization_levels = 2 ** bit_depth
    step_size = vfs / quantization_levels
    shifted_signal = signal + vfs / 2 - step_size / 2
    quantized_values = np.round(shifted_signal / step_size)
    output_signal = quantized_values * step_size - vfs / 2 + step_size / 2
    return np.clip(output_signal, -vfs / 2, vfs / 2)


def simulate_pipeline_adc_with_offsets(input_data: np.ndarray, offset_values: np.ndarray) -> tuple:
    """模拟带前4级OTA偏移的Pipeline ADC"""
    current_residue = input_data.copy()
    digital_output = np.zeros_like(input_data)
    residue_history = []

    # 前4级带偏移处理
    for stage in range(4):
        quantized = apply_quantization(current_residue)
        digital_output += quantized * (2 ** (13 - (stage + 2)))
        current_residue = current_residue - quantized + offset_values[stage]
        residue_history.append(current_residue.copy())

    # 后2级理想处理
    for stage in range(4, 6):
        quantized = apply_quantization(current_residue)
        digital_output += quantized * (2 ** (13 - (stage + 2)))
        current_residue = current_residue - quantized

    return digital_output, np.stack(residue_history, axis=1)


def simulate_ideal_pipeline_adc(input_data: np.ndarray) -> np.ndarray:
    """模拟理想Pipeline ADC"""
    current_residue = input_data.copy()
    digital_output = np.zeros_like(input_data)

    for stage in range(6):
        quantized = apply_quantization(current_residue)
        digital_output += quantized * (2 ** (13 - (stage + 2)))
        current_residue = current_residue - quantized

    return digital_output


def perform_lms_calibration(reference_signal: np.ndarray, distorted_signal: np.ndarray,
                            residue_inputs: np.ndarray, step_size: float = 1e-100) -> tuple:
    """执行LMS校准算法"""
    total_steps = len(reference_signal)
    num_stages = residue_inputs.shape[1]

    weights = np.zeros((total_steps, num_stages))  # 权重矩阵
    error_signal = np.zeros(total_steps)  # 误差信号
    corrected_output = np.zeros(total_steps)  # 校正后的输出

    for step in range(total_steps):
        current_residue = residue_inputs[step]
        if step > 0:
            corrected_output[step] = distorted_signal[step] - np.dot(weights[step - 1], current_residue)
            error_signal[step] = reference_signal[step] - corrected_output[step]
            weights[step] = weights[step - 1] + step_size * error_signal[step] * current_residue
        else:
            corrected_output[step] = distorted_signal[step]
            weights[step] = weights[step - 1]

    return weights, error_signal, corrected_output


def main():
    """主函数，运行ADC模拟和LMS校准"""
    # 初始化参数
    fs, fin, N, vfs = setup_simulation_parameters()
    time_vector = generate_time_vector(N, fs)
    input_wave = produce_input_waveform(time_vector, fin)

    # 定义偏移
    stage_offsets = np.array([0.1, -0.05, 0.07, -0.02])

    # 获取ADC输出
    adc_with_offset_output, residues = simulate_pipeline_adc_with_offsets(input_wave, stage_offsets)
    ideal_adc_output = simulate_ideal_pipeline_adc(input_wave)

    # 执行LMS校准
    weights, errors, corrected_signal = perform_lms_calibration(
        ideal_adc_output, adc_with_offset_output, ascended )
    weights, errors, corrected_signal = perform_lms_calibration(
        ideal_adc_output, adc_with_offset_output, residues)

    # 这里可以添加绘图或其他分析代码

    if __name__ == "__main__":
        main()