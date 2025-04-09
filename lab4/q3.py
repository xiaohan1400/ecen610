import numpy as np
import matplotlib.pyplot as plt

# 参数设置
N_bits = 7  # 量化位数
M_values = np.arange(2, 11)  # M取值范围
fs = 1000  # 采样率（示例值，需根据问题2调整）
duration = 1.0  # 信号时长
t = np.arange(0, duration, 1 / fs)
np.random.seed(42)

# %% 1. 生成多音信号（假设问题2.b的生成方式）
num_tones = 5
freqs = np.random.randint(10, 400, num_tones)
x = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

# %% 2. 模拟非理想采样（假设问题2.a的时间常数τ对应的RC滤波器）
tau = 0.001  # 示例值，需替换为问题2.a的结果
alpha = np.exp(-1 / (tau * fs))
sampled = np.zeros_like(x)
for i in range(1, len(t)):
    sampled[i] = alpha * sampled[i - 1] + (1 - alpha) * x[i]

# 添加量化噪声
Q_step = 2 / (2 ** N_bits)  # 假设信号范围归一化到[-1,1]
quantized = np.round(sampled / Q_step) * Q_step
q_noise = quantized - sampled

# %% 3. 计算理想采样信号（无时间常数效应）
ideal_sampled = x.copy()
s_error = ideal_sampled - sampled  # 采样误差

# %% 4. 构建FIR滤波器并补偿
variance_ratios = []

for M in M_values:
    # 构造训练数据
    X = []
    S = []
    for i in range(M - 1, len(t) - 1):
        X.append(quantized[i - M + 1:i])
        S.append(s_error[i])
    X = np.array(X)
    S = np.array(S)

    # 最小二乘估计
    w = np.linalg.lstsq(X, S, rcond=None)[0]

    # 应用滤波器并补偿
    s_est = np.convolve(quantized, w, mode='full')[:len(quantized)]
    corrected = quantized + s_est

    # 计算新误差
    E_new = corrected - ideal_sampled
    var_Enew = np.var(E_new)
    var_q = (Q_step ** 2) / 12
    variance_ratios.append(var_Enew / var_q)

# %% 5. 绘图
plt.figure(figsize=(10, 6))
plt.plot(M_values, variance_ratios, 'bo-', linewidth=2)
plt.xlabel('FIR Filter Order (M)')
plt.ylabel('Variance Ratio (E/Quantization Noise)')
plt.title('Error Variance vs Filter Order')
plt.grid(True)
plt.yscale('log')
plt.show()