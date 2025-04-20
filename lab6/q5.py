import numpy as np
import matplotlib.pyplot as plt


def setup_simulation_environment():
    """配置模拟环境并生成基础信号"""
    frequency_sampling = 500e6  # 采样频率，单位Hz
    total_points = 8192  # 总采样点数
    frequency_input = 200e6  # 输入信号频率，单位Hz

    time_array = np.arange(total_points) / frequency_sampling
    base_signal = 0.9 * np.sin(2 * np.pi * frequency_input * time_array)
    return time_array, base_signal


def apply_noise(signal: np.ndarray, snr_db: float = 80) -> np.ndarray:
    """向信号中添加指定SNR的噪声"""
    power_signal = np.mean(signal ** 2)
    power_noise = power_signal / (10 ** (snr_db / 10))
    noise_vector = np.random.normal(scale=np.sqrt(power_noise), size=len(signal))
    return signal + noise_vector


def introduce_nonlinearity(data: np.ndarray, factors: list) -> np.ndarray:
    """应用非线性多项式变换"""
    return sum(coef * data ** (index + 1) for index, coef in enumerate(factors))


def run_lms_adjustment(input_signal: np.ndarray, target_signal: np.ndarray,
                       poly_degree: int = 5, learning_rate: float = 0.01,
                       step_interval: int = 1) -> np.ndarray:
    """执行LMS校准并返回误差历史"""
    signal_length = len(input_signal)
    coefficients = np.zeros(poly_degree)
    error_log = []

    for idx in range(0, min(signal_length, 200 * step_interval), step_interval):
        poly_features = np.array([input_signal[idx] ** (k + 1) for k in range(poly_degree)])
        predicted_value = np.dot(coefficients, poly_features)
        error = target_signal[idx] - predicted_value
        coefficients += learning_rate * error * poly_features
        error_log.append(error)

    return np.array(error_log)


def visualize_order_effects(signal_with_noise: np.ndarray, distorted_signal: np.ndarray,
                            rate: float, fixed_step: int):
    """绘制不同阶数LMS校准的误差曲线"""
    degrees = [1, 2, 3, 4, 5]

    plt.figure(figsize=(10, 5))
    for deg in degrees:
        errors = run_lms_adjustment(signal_with_noise, distorted_signal,
                                    poly_degree=deg, learning_rate=rate, step_interval=fixed_step)
        plt.plot(errors, label=f'Degree {deg}', linewidth=1.2)

    plt.title(f"Error Convergence for Different Orders ")
    plt.xlabel("Step Number")
    plt.ylabel("Error Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_step_effects(signal_with_noise: np.ndarray, distorted_signal: np.ndarray,
                           rate: float, fixed_degree: int):
    """绘制不同步长间隔对LMS误差的影响"""
    step_sizes = [10, 100, 1000, 10000]

    plt.figure(figsize=(10, 5))
    for step in step_sizes:
        errors = run_lms_adjustment(signal_with_noise, distorted_signal,
                                    poly_degree=fixed_degree, learning_rate=rate, step_interval=step)
        plt.plot(errors, label=f'Step Size {step}', linewidth=1.2)

    plt.title(f"Error Convergence for Different Decimation")
    plt.xlabel("Step Number")
    plt.ylabel("Error Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """主函数，执行信号处理和LMS校准"""
    # 初始化信号
    _, original_signal = setup_simulation_environment()
    noisy_signal = apply_noise(original_signal, snr_db=80)

    # 定义非线性参数
    gain_factor = 2
    nonlinearity_terms = [gain_factor, 0.1 * gain_factor, 0.2 * gain_factor,
                          0.15 * gain_factor, 0.1 * gain_factor]
    output_signal = introduce_nonlinearity(noisy_signal, nonlinearity_terms)

    # LMS参数
    adjustment_rate = 0.01

    # 绘制结果
    visualize_order_effects(noisy_signal, output_signal, adjustment_rate, fixed_step=10)
    visualize_step_effects(noisy_signal, output_signal, adjustment_rate, fixed_degree=5)


if __name__ == "__main__":
    main()