
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# 信号定义与工具函数
###############################################################################

def generate_signal_x1(time, frequency1=300e6):
    return np.cos(2 * np.pi * frequency1 * time)


def generate_signal_x2(time, frequency2=800e6):
    return np.cos(2 * np.pi * frequency2 * time)


def sinc_interpolation(dense_time, sample_time, sample_values, sampling_period):
    time_diff = (dense_time[:, None] - sample_time[None, :]) / sampling_period
    kernel = np.sinc(time_diff)
    return kernel.dot(sample_values)


def sample_and_reconstruct(signal_function, sampling_rate, start_time, end_time, time_step):
    sampling_period = 1.0 / sampling_rate
    dense_time = np.arange(start_time, end_time, time_step)
    dense_signal = signal_function(dense_time)

    sample_time = np.arange(start_time, end_time, sampling_period)
    sample_values = signal_function(sample_time)

    reconstructed_signal = sinc_interpolation(dense_time, sample_time, sample_values, sampling_period)
    mse = np.mean((dense_signal - reconstructed_signal) ** 2)
    return dense_time, dense_signal, sample_time, sample_values, reconstructed_signal, mse


###############################################################################
# 1. 使用500MHz采样x1与x2并绘图
# 2. 从采样数据重建信号
###############################################################################

def demonstrate_part_1_2():
    frequency1, frequency2 = 300e6, 800e6
    sampling_rate = 500e6  # 500 MHz采样率
    start_time, end_time, time_step = 0.0, 20e-9, 1e-11

    # 对x1采样和重构
    dense_time, dense_signal_x1, sample_time_x1, sample_values_x1, reconstructed_signal_x1, mse_x1 = sample_and_reconstruct(
        lambda t: generate_signal_x1(t, frequency1), sampling_rate, start_time, end_time, time_step)

    # 对x2采样和重构
    _, dense_signal_x2, sample_time_x2, sample_values_x2, reconstructed_signal_x2, mse_x2 = sample_and_reconstruct(
        lambda t: generate_signal_x2(t, frequency2), sampling_rate, start_time, end_time, time_step)

    # 绘制结果
    plt.figure(figsize=(12, 6))

    # x1(t)
    plt.subplot(2, 1, 1)
    plt.plot(dense_time * 1e9, dense_signal_x1, 'b-', label='Original')
    plt.plot(sample_time_x1 * 1e9, sample_values_x1, 'ro', markersize=4, label='Samples')
    plt.title(f'x1(t) @ Fs={sampling_rate / 1e6}MHz (MSE={mse_x1:.2e})')
    plt.xlabel('Time (ns)'), plt.ylabel('Amplitude')
    plt.legend(), plt.grid()

    # x2(t)
    plt.subplot(2, 1, 2)
    plt.plot(dense_time * 1e9, dense_signal_x2, 'g-', label='Original')
    plt.plot(sample_time_x2 * 1e9, sample_values_x2, 'ro', markersize=4, label='Samples')
    plt.title(f'x2(t) @ Fs={sampling_rate / 1e6}MHz (MSE={mse_x2:.2e})')
    plt.xlabel('Time (ns)'), plt.ylabel('Amplitude')
    plt.legend(), plt.grid()

    plt.tight_layout()
    plt.show()

    # 绘制重构信号对比
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(dense_time * 1e9, dense_signal_x1, 'b-', label='Original')
    plt.plot(dense_time * 1e9, reconstructed_signal_x1, 'r--', label='Reconstructed')
    plt.title(f'x1(t) Reconstruction (MSE={mse_x1:.2e})')
    plt.xlabel('Time (ns)'), plt.ylabel('Amplitude')
    plt.legend(), plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(dense_time * 1e9, dense_signal_x2, 'g-', label='Original')
    plt.plot(dense_time * 1e9, reconstructed_signal_x2, 'm--', label='Reconstructed')
    plt.title(f'x2(t) Reconstruction (MSE={mse_x2:.2e})')
    plt.xlabel('Time (ns)'), plt.ylabel('Amplitude')
    plt.legend(), plt.grid()

    plt.tight_layout()
    plt.show()


###############################################################################
# 3. 零阶保持系统的理想重构公式推导 (理论部分)
###############################################################################
"""
零阶保持采样系统的理想重构方程推导：

假设：
- 采样周期 T，满足Nyquist率（Fs = 1/T ≥ 2B，B为信号带宽）
- 脉冲宽度 w ≤ T，采样点位于脉冲末端
- 理想低通滤波器截止频率 Fc = Fs/2

重构过程：
1. 零阶保持：每个采样值 x[n] 保持 w 时间
2. 通过理想低通滤波器 H(f) = rect(f/(2Fc))

重构公式：
x(t) = ∑_{n=-∞}^∞ x[n] · h(t - nT)

其中 h(t) 是矩形脉冲与理想低通冲激响应的卷积：
h(t) = [rect(t/w) * sinc(2Fc t)] 
     = ∫_{-w/2}^{w/2} sinc(2Fc(t - τ)) dτ

解析解形式：
h(t) = (1/(2πFc)) [Si(2πFc(t + w/2)) - Si(2πFc(t - w/2))]

其中 Si 是正弦积分函数。当 w=T 时简化为：
h(t) = sinc(t/T) * rect(t/T)
"""

if __name__ == "__main__":
    demonstrate_part_1_2()


###############################################################################
# 4.
###############################################################################


# =============================================================================
# Parameter Definition
# =============================================================================
frequency1 = 300e6           # Signal frequency 300 MHz
sampling_rate = 500e6           # Sampling rate 800 MHz
sampling_period = 1 / sampling_rate          # Sampling interval 1.25 ns
total_duration = 10 / frequency1          # Total duration (10 cycles) ≈ 33.333 ns
time_step = 1e-12           # Time resolution 0.001 ns

# =============================================================================
# Generate continuous signal
# =============================================================================
dense_time = np.arange(0, total_duration, time_step)
dense_signal_x1 = np.cos(2 * np.pi * frequency1 * dense_time)

# =============================================================================
# Generate sampling points
# =============================================================================
# Case 1: Aligned sampling (starting at 0)
sample_time1 = np.arange(0, total_duration - sampling_period + 1e-15, sampling_period)
sample_values1 = np.cos(2 * np.pi * frequency1 * sample_time1)

# Case 2: Shifted sampling (starting at Ts/2)
sample_time2 = np.arange(sampling_period/2, total_duration - sampling_period/2 + 1e-15, sampling_period)
sample_values2 = np.cos(2 * np.pi * frequency1 * sample_time2)

# =============================================================================
# Sinc reconstruction function
# =============================================================================
def sinc_interpolation(dense_time, sample_time, sample_values, sampling_period):
    """Shannon reconstruction using sinc interpolation"""
    time_diff = (dense_time[:, None] - sample_time[None, :]) / sampling_period
    return np.sinc(time_diff) @ sample_values

# Reconstruct signals
reconstructed_signal1 = sinc_interpolation(dense_time, sample_time1, sample_values1, sampling_period)
reconstructed_signal2 = sinc_interpolation(dense_time, sample_time2, sample_values2, sampling_period)

# =============================================================================
# Calculate MSE
# =============================================================================
mse1 = np.mean((reconstructed_signal1 - dense_signal_x1)**2)
mse2 = np.mean((reconstructed_signal2 - dense_signal_x1)**2)

print(f"[Aligned Sampling] MSE = {mse1:.4e}")
print(f"[Shifted Sampling] MSE = {mse2:.4e}")

# =============================================================================
# Visualization with English labels
# =============================================================================
plt.figure(figsize=(14, 10))

# Original signal and sampling points -----------------------------------------
plt.subplot(3, 1, 1)
plt.plot(dense_time*1e9, dense_signal_x1, "b-", linewidth=1.5, label="Original Signal")
plt.plot(sample_time1*1e9, sample_values1, "ro", markersize=6, markerfacecolor="none", label="Aligned Samples")
plt.plot(sample_time2*1e9, sample_values2, "g^", markersize=6, markerfacecolor="none", label="Shifted Samples (Ts/2)")
plt.title(f"Signal x1(t)=cos(2π·{frequency1/1e6}MHz·t) Sampling Comparison (Fs={sampling_rate/1e6}MHz)")
plt.xlabel("Time (ns)"), plt.ylabel("Amplitude")
plt.legend(), plt.grid(True, linestyle="--")
plt.xlim(0, total_duration*1e9)

# Reconstruction comparison ---------------------------------------------------
plt.subplot(3, 1, 2)
plt.plot(dense_time*1e9, dense_signal_x1, "b-", linewidth=1.5, label="Original Signal")
plt.plot(dense_time*1e9, reconstructed_signal1, "r--", linewidth=1.2, label="Aligned Reconstruction")
plt.plot(dense_time*1e9, reconstructed_signal2, "g--", linewidth=1.2, label="Shifted Reconstruction")
plt.title(f"Reconstruction Comparison (MSE Aligned: {mse1:.1e}, MSE Shifted: {mse2:.1e})")
plt.xlabel("Time (ns)"), plt.ylabel("Amplitude")
plt.legend(), plt.grid(True, linestyle="--")
plt.xlim(0, total_duration*1e9)

# Error comparison ------------------------------------------------------------
plt.subplot(3, 1, 3)
plt.plot(dense_time*1e9, (reconstructed_signal1 - dense_signal_x1)**2, "r-", label="Aligned Error")
plt.plot(dense_time*1e9, (reconstructed_signal2 - dense_signal_x1)**2, "g-", label="Shifted Error")
plt.title("Squared Error Comparison")
plt.xlabel("Time (ns)"), plt.ylabel("Squared Error")
plt.legend(), plt.grid(True, linestyle="--")
plt.yscale("log")  # Logarithmic scale
plt.xlim(0, total_duration*1e9)

plt.tight_layout()
plt.show()