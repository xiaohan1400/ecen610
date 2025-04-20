import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. 生成 128-tone BPSK 信号
# ============================

Fs = 500e6  # ADC采样率
N_samples = 4096
BW = 200e6
N_tones = 128

t = np.arange(N_samples) / Fs
frequencies = np.linspace(-BW/2, BW/2, N_tones)
bpsk_symbols = np.random.choice([-1, 1], size=N_tones)

input_signal = np.zeros_like(t, dtype=complex)
for i in range(N_tones):
    input_signal += bpsk_symbols[i] * np.exp(2j * np.pi * frequencies[i] * t)
input_signal = np.real(input_signal)

# ============================
# 2. 定义非线性函数（10%非线性）
# ============================

def nonlinear_gain(x, alpha2=0.1, alpha3=0.1, alpha4=0.1):
    return x * (1 + alpha2 * x + alpha3 * x**2 + alpha4 * x**3)

# ============================
# 3. 模拟 Pipeline ADC（MDAC 含非线性）
# ============================

stages = 6  # 每级2.5-bit
np.random.seed(0)
gain_err = np.random.normal(0, 0.01, stages)
offset_err = np.random.normal(0, 0.01, stages)
cap_mismatch = np.random.normal(0, 0.005, stages)
comp_offset = np.random.normal(0, 0.005, stages)

def pipeline_adc_nonlinear_mdac(x, stages, gain_err, offset_err, cap_mismatch, comp_offset):
    stage_output = []
    residual = x.copy()
    for i in range(stages):
        quant = np.round(residual * 4) / 4  # 等效2-bit量化
        quant += comp_offset[i]
        mdac_input = residual - quant
        mdac_gain = (2 + gain_err[i]) * (1 + cap_mismatch[i])
        mdac_output = nonlinear_gain(mdac_input * mdac_gain) + offset_err[i]
        stage_output.append(quant)
        residual = mdac_output
    return np.sum(stage_output, axis=0)

adc_out = pipeline_adc_nonlinear_mdac(input_signal, stages, gain_err, offset_err, cap_mismatch, comp_offset)

# ============================
# 4. LMS 校准
# ============================

mu = 0.01
weights = np.zeros((N_samples, stages))
errors = np.zeros(N_samples)
w = np.zeros(stages)

for n in range(N_samples):
    features = np.zeros(stages)
    r = adc_out[n]
    for i in range(stages):
        q = np.round(r * 4) / 4
        features[i] = q
        r = (r - q) * 2
    y_hat = np.dot(w, features)
    e = input_signal[n] - y_hat
    w = w + mu * e * features
    weights[n] = w
    errors[n] = e

# ============================
# 5. 绘图：误差与权重收敛
# ============================

plt.figure(figsize=(12, 5))

# 误差收敛图（dB）
plt.subplot(1, 2, 1)
plt.plot(10 * np.log10(errors**2 + 1e-12))
plt.title("LMS Error Convergence (MDAC Nonlinearity 10%)")
plt.xlabel("Iteration")
plt.ylabel("Error Power (dB)")

# 权重收敛图
plt.subplot(1, 2, 2)
for i in range(stages):
    plt.plot(weights[:, i], label=f"w{i+1}")
plt.title("LMS Weights Convergence (MDAC Nonlinearity 10%)")
plt.xlabel("Iteration")
plt.ylabel("Weights")
plt.legend()

plt.tight_layout()
plt.show()