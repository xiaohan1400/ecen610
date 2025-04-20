import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ==============================================
# 数据类型定义 (使用dataclass增强可读性)
# ==============================================
@dataclass
class ADCNonidealityParams:
    """Pipeline ADC非理想参数配置"""
    stage_count: int  # 流水线级数
    gain_errors: np.ndarray  # 各级增益误差（标准差）
    offset_errors: np.ndarray  # 各级偏移误差（标准差）
    cap_mismatch: np.ndarray  # 电容失配误差（标准差）
    comp_offset: np.ndarray  # 比较器偏移误差（标准差）


@dataclass
class SignalGenerationParams:
    """信号生成参数配置"""
    sampling_rate_hz: float  # 采样率 (Hz)
    sample_count: int  # 采样点数
    bandwidth_hz: float  # 信号带宽 (Hz)
    tone_count: int  # 子载波数量


# ==============================================
# 模块1: 多音BPSK信号生成
# ==============================================
def generate_multitone_bpsk(params: SignalGenerationParams) -> np.ndarray:
    """
    生成多音BPSK基带信号
    Args:
        params: 信号参数配置对象
    Returns:
        real_signal: 生成的实基带信号
    """
    time_vector = np.arange(params.sample_count) / params.sampling_rate_hz
    freq_bins = np.linspace(-params.bandwidth_hz / 2, params.bandwidth_hz / 2, params.tone_count)
    bpsk_symbols = np.random.choice([-1, 1], size=params.tone_count)

    # 生成复数信号并取实部
    complex_signal = np.sum(
        [symbol * np.exp(2j * np.pi * freq * time_vector)
         for symbol, freq in zip(bpsk_symbols, freq_bins)],
        axis=0
    )
    return np.real(complex_signal)


# ==============================================
# 模块2: 流水线ADC行为模型
# ==============================================
class PipelineADC:
    """流水线ADC行为模型，包含非理想因素"""

    def __init__(self, params: ADCNonidealityParams):
        self.params = params
        self.QUANTIZATION_LEVELS = 4  # 2-bit量化等效电平数

    def convert(self, analog_input: np.ndarray) -> np.ndarray:
        """
        执行ADC转换过程
        Args:
            analog_input: 模拟输入信号
        Returns:
            digital_output: 数字输出信号
        """
        stage_outputs = []
        residual_signal = analog_input.copy()

        for stage in range(self.params.stage_count):
            # 量化过程
            quantized = np.round(residual_signal * self.QUANTIZATION_LEVELS) / self.QUANTIZATION_LEVELS
            quantized += self.params.comp_offset[stage]

            # MDAC输出计算（包含非理想因素）
            residual_signal = (
                                      (residual_signal - quantized) * (2 + self.params.gain_errors[stage])
                                      + self.params.offset_errors[stage]
                              ) * (1 + self.params.cap_mismatch[stage])

            stage_outputs.append(quantized)

        return np.sum(stage_outputs, axis=0)


# ==============================================
# 模块3: LMS校准算法实现
# ==============================================
class LMS_Calibrator:
    """LMS算法校准器"""

    def __init__(self, stage_count: int, learning_rate: float):
        self.weights = np.zeros(stage_count)
        self.LEARNING_RATE = learning_rate

    def update_weights(self, reference_signal: np.ndarray, adc_output: np.ndarray) -> tuple:
        """
        执行权重更新过程
        Args:
            reference_signal: 参考信号（理想输入）
            adc_output: ADC输出信号
        Returns:
            (error_signal, updated_weights): 误差信号和更新后的权重
        """
        error_history = np.zeros_like(reference_signal)
        weight_history = np.zeros((len(reference_signal), len(self.weights)))

        for n in range(len(reference_signal)):
            # 构建特征向量（各stage量化结果）
            features = np.zeros_like(self.weights)
            residual = reference_signal[n]

            for stage in range(len(self.weights)):
                quantized = np.round(residual * 4) / 4
                features[stage] = quantized
                residual = (residual - quantized) * 2  # 理想残差路径

            # LMS核心算法
            estimated = np.dot(self.weights, features)
            error = reference_signal[n] - estimated
            self.weights += self.LEARNING_RATE * error * features

            # 记录过程数据
            error_history[n] = error
            weight_history[n] = self.weights

        return error_history, weight_history


# ==============================================
# 模块4: 可视化模块
# ==============================================
def plot_performance_metrics(errors: np.ndarray, weights_history: np.ndarray) -> None:
    """绘制校准性能指标"""
    plt.figure(figsize=(14, 6))

    # 误差收敛图
    plt.subplot(1, 2, 1)
    plt.plot(10 * np.log10(errors ** 2 + 1e-12))
    plt.title("LMS Error Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Error (dB)")
    plt.grid(True)

    # 权重收敛图
    plt.subplot(1, 2, 2)
    for stage in range(weights_history.shape[1]):
        plt.plot(weights_history[:, stage], label=f"W{stage + 1}")
    plt.title("LMS Weight")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    # 初始化随机种子保证可重复性
    np.random.seed(0)

    # 参数配置
    adc_params = ADCNonidealityParams(
        stage_count=6,
        gain_errors=np.random.normal(0, 0.01, 6),
        offset_errors=np.random.normal(0, 0.01, 6),
        cap_mismatch=np.random.normal(0, 0.005, 6),
        comp_offset=np.random.normal(0, 0.005, 6)
    )

    signal_params = SignalGenerationParams(
        sampling_rate_hz=500e6,
        sample_count=4096,
        bandwidth_hz=200e6,
        tone_count=128
    )

    # 信号生成
    ideal_signal = generate_multitone_bpsk(signal_params)

    # ADC转换
    adc = PipelineADC(adc_params)
    adc_output = adc.convert(ideal_signal)

    # LMS校准
    calibrator = LMS_Calibrator(
        stage_count=adc_params.stage_count,
        learning_rate=0.01
    )
    calibration_errors, weight_history = calibrator.update_weights(ideal_signal, adc_output)

    # 结果可视化
    plot_performance_metrics(calibration_errors, weight_history)