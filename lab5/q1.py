import numpy as np
import matplotlib.pyplot as plt

# 参数设置
fs = 100e6  # 采样频率 100 MHz
T = 1 / fs  # 采样周期
t_sim = 1e-6  # 仿真时间 1us
n_samples = int(t_sim * fs)  # 采样点数
time = np.linspace(0, t_sim, n_samples)  # 时间向量

# 输入信号（正弦波，频率为1 MHz）
f_in = 1e6  # 输入信号频率
vin = 1.0 * np.sin(2 * np.pi * f_in * time)  # 输入电压范围 [-1, 1] V

# 2.5-bit MDAC参数（使用6个比较器）
vref = 1.0  # 参考电压
levels = [-0.75 * vref, -0.5 * vref, -0.25 * vref, 0, 0.25 * vref, 0.5 * vref, 0.75 * vref]  # 6个比较器，7个阈值
gain = 4  # 残差放大增益


# MDAC量化函数
def mdac_2_5bit_six_comparators(vin_sample, vref, levels, gain):
    # 量化输入信号（8个区间）
    if vin_sample <= levels[0]:
        d = 0  # -3
        vres = gain * (vin_sample - (-vref))
    elif vin_sample <= levels[1]:
        d = 1  # -2
        vres = gain * (vin_sample - (-0.75 * vref))
    elif vin_sample <= levels[2]:
        d = 2  # -1
        vres = gain * (vin_sample - (-0.5 * vref))
    elif vin_sample <= levels[3]:
        d = 3  # 0
        vres = gain * (vin_sample - (-0.25 * vref))
    elif vin_sample <= levels[4]:
        d = 4  # 1
        vres = gain * (vin_sample - (0.25 * vref))
    elif vin_sample <= levels[5]:
        d = 5  # 2
        vres = gain * (vin_sample - (0.5 * vref))
    elif vin_sample <= levels[6]:
        d = 6  # 3
        vres = gain * (vin_sample - (0.75 * vref))
    else:
        d = 7  # 4
        vres = gain * (vin_sample - vref)

    # 限制残差输出范围（模拟实际电路中的饱和）
    vres = np.clip(vres, -vref, vref)
    return d, vres


# 仿真MDAC
digital_out = []
residual_out = []
for v in vin:
    d, vres = mdac_2_5bit_six_comparators(v, vref, levels, gain)
    digital_out.append(d)
    residual_out.append(vres)

# 转换为numpy数组
digital_out = np.array(digital_out)
residual_out = np.array(residual_out)

# 可视化结果
plt.figure(figsize=(12, 8))

# 输入信号
plt.subplot(3, 1, 1)
plt.plot(time * 1e6, vin, label="Input Signal (1 MHz)")
plt.xlabel("Time (us)")
plt.ylabel("Voltage (V)")
plt.title("Input Signal")
plt.grid(True)
plt.legend()

# 数字输出
plt.subplot(3, 1, 2)
plt.step(time * 1e6, digital_out, label="Digital Output (6 Comparators)")
plt.xlabel("Time (us)")
plt.ylabel("Digital Code")
plt.title("Digital Output")
plt.grid(True)
plt.legend()

# 残差输出
plt.subplot(3, 1, 3)
plt.plot(time * 1e6, residual_out, label="Residual Output (Gain=4)")
plt.xlabel("Time (us)")
plt.ylabel("Voltage (V)")
plt.title("Residual Output")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 计算基本性能指标
print(f"Sampling Frequency: {fs / 1e6} MHz")
print(f"Input Frequency: {f_in / 1e6} MHz")
print(f"Reference Voltage: {vref} V")
print(f"Gain: {gain}")