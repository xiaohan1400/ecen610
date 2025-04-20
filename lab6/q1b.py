import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 参数设置
# -----------------------------
fs = 500e6  # 采样率 500 MHz
T = 1 / fs
t_sim = 1e-6  # 仿真时间 1 μs
n_samp = int(t_sim * fs)
time = np.linspace(0, t_sim, n_samp)

# 输入信号：Vpp=1V → amplitude=0.5V, f=200MHz
vpp = 1.0
amp = vpp / 2
f_in = 200e6
vin = amp * np.sin(2 * np.pi * f_in * time)

# ADC 结构参数
n_bits = 13
gain = 4
vref = amp  # 参考电压设为 0.5V
n_stages = int(np.ceil(n_bits / 2))  # 7 stages
levels = np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]) * vref


# -----------------------------
# 单级 MDAC 函数
# -----------------------------
def mdac_stage(v, vref, levels, gain):
    if v <= levels[0]:
        d = 0;
        vres = gain * (v - (-vref))
    elif v <= levels[1]:
        d = 1;
        vres = gain * (v - levels[0])
    elif v <= levels[2]:
        d = 2;
        vres = gain * (v - levels[1])
    elif v <= levels[3]:
        d = 3;
        vres = gain * (v - levels[2])
    elif v <= levels[4]:
        d = 4;
        vres = gain * (v - levels[3])
    elif v <= levels[5]:
        d = 5;
        vres = gain * (v - levels[4])
    elif v <= levels[6]:
        d = 6;
        vres = gain * (v - levels[5])
    else:
        d = 7;
        vres = gain * (v - levels[6])
    return d, np.clip(vres, -vref, vref)


# -----------------------------
# 流水线仿真（产生重建波形 recon）
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

# 组合总体数字码
adc_code = np.zeros(n_samp, dtype=int)
for i in range(n_stages):
    weight = 2 ** (2 * (n_stages - i - 1))
    adc_code += digital_outputs[i] * weight

# 重建波形
adc_code_norm = adc_code / adc_code.max()
recon = (adc_code_norm - 0.5) * vpp


# -----------------------------
# 频域分析
# -----------------------------
def plot_spectrum(x, fs, ax, title):
    N = len(x)
    # 可选窗函数
    window = np.hanning(N)
    X = np.fft.fft(x * window)
    X = X[:N // 2]
    freq = np.fft.fftfreq(N, 1 / fs)[:N // 2]
    # 幅度谱(dB)
    X_mag = 20 * np.log10(np.abs(X) / np.max(np.abs(X)))

    ax.plot(freq / 1e6, X_mag, color='C1')
    ax.set_xlim(0, fs / 2 / 1e6)
    ax.set_ylim(-100, 0)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True)


fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 输入信号频谱
plot_spectrum(vin, fs, axs[0], "Input Signal Spectrum")

# 重建输出信号频谱
plot_spectrum(recon, fs, axs[1], "Reconstructed Output Spectrum")

plt.tight_layout()
plt.show()
