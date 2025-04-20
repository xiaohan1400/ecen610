import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 参数设置
# -----------------------------
fs     = 500e6       # 采样率 500 MHz
T      = 1 / fs
t_sim  = 1e-6        # 仿真时间 1 μs
n_samp = int(t_sim * fs)
time   = np.linspace(0, t_sim, n_samp)

# 输入信号：Vpp=1V → amplitude=0.5V, f=200MHz
vpp    = 1.0
amp    = vpp / 2
f_in   = 200e6
vin    = amp * np.sin(2 * np.pi * f_in * time)

# ADC 结构参数
n_bits    = 13
gain      = 4
vref      = amp      # 参考电压设为 0.5V
n_stages  = int(np.ceil(n_bits / 2))  # 7 stages
levels    = np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]) * vref

# -----------------------------
# 单级 MDAC 函数
# -----------------------------
def mdac_stage(v, vref, levels, gain):
    if v <= levels[0]:
        d = 0; vres = gain * (v - (-vref))
    elif v <= levels[1]:
        d = 1; vres = gain * (v - levels[0])
    elif v <= levels[2]:
        d = 2; vres = gain * (v - levels[1])
    elif v <= levels[3]:
        d = 3; vres = gain * (v - levels[2])
    elif v <= levels[4]:
        d = 4; vres = gain * (v - levels[3])
    elif v <= levels[5]:
        d = 5; vres = gain * (v - levels[4])
    elif v <= levels[6]:
        d = 6; vres = gain * (v - levels[5])
    else:
        d = 7; vres = gain * (v - levels[6])
    vres = np.clip(vres, -vref, vref)
    return d, vres

# -----------------------------
# 流水线仿真
# -----------------------------
digital_outputs = [[] for _ in range(n_stages)]
vres = vin.copy()

for stage in range(n_stages):
    dout = []
    next_vres = []
    for v in vres:
        d, rv = mdac_stage(v, vref, levels, gain)
        dout.append(d)
        next_vres.append(rv)
    digital_outputs[stage] = np.array(dout)
    vres = np.array(next_vres)

# 组合总体数字码（简单加权）
adc_code = np.zeros(n_samp, dtype=int)
for i in range(n_stages):
    weight = 2 ** (2 * (n_stages - i - 1))  # 每级有效 2 bit
    adc_code += digital_outputs[i] * weight

# 重建波形：归一化后映射到 [-amp, +amp]
adc_code_norm = adc_code / adc_code.max()
recon = (adc_code_norm - 0.5) * vpp

# -----------------------------
# SNR 计算
# -----------------------------
signal_power = np.mean(vin**2)
noise_power  = np.mean((recon - vin)**2)
snr = 10 * np.log10(signal_power / noise_power)
print(f"SNR = {snr:.2f} dB")  # 结果示例：约 12.13 dB

# -----------------------------
# 可视化（同前）
# -----------------------------
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# 1. 输入波形
axs[0].plot(time * 1e6, vin, color='C0')
axs[0].set_ylabel("Voltage (V)")
axs[0].set_title(f"Input Waveform: Vpp={vpp}V, f={f_in/1e6:.0f}MHz")
axs[0].grid(True)

# 2. 每一级离散化输出
for i in range(n_stages):
    axs[1].step(time * 1e6,
                digital_outputs[i] + i * 10,
                where='post',
                label=f"Stage {i+1}")
axs[1].set_ylabel("Code + Offset")
axs[1].set_title("Stage-wise Discrete Outputs")
axs[1].legend(loc='upper right', ncol=2, fontsize='small')
axs[1].grid(True)

# 3. 整体数字码
axs[2].step(time * 1e6, adc_code, where='post', color='C2')
axs[2].set_ylabel("Digital Code")
axs[2].set_title("Overall Digital Code")
axs[2].grid(True)

# 4. 重建后的输出波形
axs[3].plot(time * 1e6, recon, color='C3')
axs[3].set_ylabel("Voltage (V)")
axs[3].set_xlabel("Time (μs)")
axs[3].set_title("Reconstructed Analog Waveform")
axs[3].grid(True)

plt.tight_layout()
plt.show()
