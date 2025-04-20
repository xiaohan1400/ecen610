import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

# 基本参数设置
fs = 500e6  # 采样率 500 MHz
fin = 200e6  # 输入频率 200 MHz
bits = 13  # ADC 分辨率
N = 8192  # 采样点数
t = np.arange(N) / fs
x_in = 0.9 * np.sin(2 * np.pi * fin * t)  # 输入信号
full_scale = 1.0  # 满量程范围 ±1V


def simulate_pipeline_adc_with_non_idealities(
        x_in,
        ota_gain=1000,  # OTA 有限增益 (A0)
        mdac_offset=0,  # MDAC 输入偏置 (相对于满量程的比例)
        cap_mismatch=0,  # 电容失配 (相对差异，如 0.01 表示 1%)
        comp_offset=0,  # 比较器偏置 (相对于满量程的比例)
        nonlinear_factor=0,  # 非线性开环增益因子 (0 表示理想线性)
        bandwidth_ratio=0.5  # OTA 带宽与奈奎斯特频率的比值 (BW/(fs/2))
):
    """
    模拟带有非理想因素的流水线 ADC

    参数:
    x_in: 输入信号
    ota_gain: OTA 有限增益 (直流增益，A0)
    mdac_offset: MDAC 输入偏置 (相对于满量程的比例)
    cap_mismatch: 电容失配 (相对差异)
    comp_offset: 比较器偏置 (相对于满量程的比例)
    nonlinear_factor: 非线性增益因子 (影响 OTA 开环增益的非线性程度)
    bandwidth_ratio: OTA 带宽与奈奎斯特频率的比值

    返回:
    sndr: 信号噪声失真比 (dB)
    fft_result: FFT 结果
    """
    # 流水线 ADC 配置
    stages = 3
    bits_per_stage = 4
    residue = x_in.copy()
    adc_output = np.zeros_like(residue)

    # 计算 OTA 带宽（基于奈奎斯特频率）
    bw = bandwidth_ratio * (fs / 2)  # 修正后的带宽计算
    tau = 1 / (2 * np.pi * bw) if bw > 0 else 0

    # 处理每个流水线级
    for stage in range(stages):
        # 当前级分辨率
        if stage == stages - 1:
            stage_bits = bits - (stages - 1) * bits_per_stage
        else:
            stage_bits = bits_per_stage

        stage_levels = 2 ** stage_bits
        stage_step = 2 * full_scale / stage_levels

        # 1. 子 ADC 量化 (比较器带偏置)
        stage_decision = np.round((residue + comp_offset * full_scale) / stage_step)
        stage_decision = np.clip(stage_decision, -stage_levels / 2, stage_levels / 2 - 1)

        # 2. DAC 输出 (带 MDAC 偏置)
        if stage == 0:
            offset_factor = 1.0
        else:
            offset_factor = 0.2  # 后级影响较小

        dac_output = stage_decision * stage_step + mdac_offset * full_scale * offset_factor

        # 3. MDAC 电容失配效应
        gain_error = 1.0 + cap_mismatch
        ideal_gain = 2 ** stage_bits
        actual_gain = ideal_gain * gain_error

        if stage < stages - 1:
            # 4. 有限 OTA 增益和带宽效应
            input_signal = residue - dac_output

            # 非线性增益模型
            signal_amp = np.abs(input_signal / full_scale)
            nonlinear_gain = ota_gain / (1 + nonlinear_factor * signal_amp)

            closed_loop_gain = actual_gain / (1 + actual_gain / nonlinear_gain)
            residue = closed_loop_gain * input_signal

            # 5. 有限带宽模拟（修正的滤波器设计）
            if 0 < bandwidth_ratio < 1:  # 确保有效范围
                Wn = bandwidth_ratio  # 直接使用比值
                b, a = signal.butter(1, Wn, 'low')
                residue = signal.lfilter(b, a, residue)

            residue = np.clip(residue, -full_scale, full_scale)

        adc_output += stage_decision * (2 ** ((stages - stage - 1) * bits_per_stage))

    # 计算最终量化输出
    x_adc = adc_output * (2 * full_scale / 2 ** bits)

    # 计算 SNDR
    window = np.hanning(N)
    X = fft(x_adc * window)
    f = fftfreq(N, d=1 / fs)

    X_half = X[:N // 2]
    f_pos = f[:N // 2]

    sig_band = (f_pos >= 199e6) & (f_pos <= 201e6)
    noise_band = (f_pos > 0) & ~sig_band

    signal_power = np.sum(np.abs(X_half[sig_band]) ** 2)
    noise_dist_power = np.sum(np.abs(X_half[noise_band]) ** 2)

    sndr = 10 * np.log10(signal_power / noise_dist_power)

    return sndr, (X_half, f_pos)


# 定义参数范围（修正后的带宽范围）
param_ranges = {
    'ota_gain': np.logspace(0, 3, 10),
    'mdac_offset': np.linspace(0, 0.1, 10),
    'cap_mismatch': np.linspace(0, 0.2, 10),
    'comp_offset': np.linspace(0, 0.2, 10),
    'nonlinear_factor': np.linspace(0, 200, 10),
    'bandwidth_ratio': np.linspace(0.1, 1, 10)  # 修正为0.1-1范围
}


def find_critical_value(param_name, param_range, target_sndr=10):
    default_params = {
        'ota_gain': 1000,
        'mdac_offset': 0,
        'cap_mismatch': 0,
        'comp_offset': 0,
        'nonlinear_factor': 0,
        'bandwidth_ratio': 0.5
    }

    sndr_values = []
    for value in param_range:
        test_params = default_params.copy()
        test_params[param_name] = value
        sndr, _ = simulate_pipeline_adc_with_non_idealities(x_in, **test_params)
        sndr_values.append(sndr)

    try:
        critical_value = np.interp(target_sndr, sndr_values[::-1], param_range[::-1])
        return critical_value, sndr_values
    except:
        return None, sndr_values


# 运行分析（后续绘图代码保持不变）
critical_values = {}
all_sndr_values = {}

for param_name, param_range in param_ranges.items():
    critical_value, sndr_values = find_critical_value(param_name, param_range)
    critical_values[param_name] = critical_value
    all_sndr_values[param_name] = sndr_values
    if critical_value is not None:
        print(f"{param_name} 临界值: {critical_value:.4f}")
    else:
        print(f"{param_name} 未达到临界值")

# 绘制所有非理想因素的影响曲线
plt.figure(figsize=(14, 8))

# 参数显示设置
param_labels = {
    'ota_gain': 'OTA DC Gain (A0)',
    'mdac_offset': 'MDAC Offset (%FS)',
    'cap_mismatch': 'Cap Mismatch (%)',
    'comp_offset': 'Comparator Offset (%FS)',
    'nonlinear_factor': 'Nonlinear Gain Factor',
    'bandwidth_ratio': 'OTA BW/Nyquist Ratio'
}

# 单位转换因子
unit_conversion = {
    'mdac_offset': 100,
    'cap_mismatch': 100,
    'comp_offset': 100
}

line_styles = ['-o', '-s', '-^', '-d', '-x', '-*']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 绘制每个参数的曲线
for idx, (param_name, param_range) in enumerate(param_ranges.items()):
    sndr_values = all_sndr_values[param_name]
    x_vals = param_range

    # 应用单位转换
    if param_name in unit_conversion:
        x_vals = x_vals * unit_conversion[param_name]

    # 特殊处理对数坐标参数
    if param_name in ['ota_gain', 'bandwidth_ratio']:
        plt.semilogx(x_vals, sndr_values, line_styles[idx],
                     color=colors[idx], linewidth=2, markersize=8,
                     label=param_labels[param_name])
    else:
        plt.plot(x_vals, sndr_values, line_styles[idx],
                 color=colors[idx], linewidth=2, markersize=8,
                 label=param_labels[param_name])

    # 标记临界点
    if critical_values[param_name] is not None:
        crit_x = critical_values[param_name]
        if param_name in unit_conversion:
            crit_x *= unit_conversion[param_name]

        plt.plot(crit_x, 10, 'o', markersize=10, color=colors[idx],
                 markeredgecolor='white', markeredgewidth=1.5)

# 添加参考线
plt.axhline(10, color='gray', linestyle='--', linewidth=2, label='SNDR=10dB')

# 图表装饰
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlabel('Parameter Value', fontsize=12)
plt.ylabel('SNDR (dB)', fontsize=12)
plt.title('Pipeline ADC Non-ideality Impact Analysis', fontsize=14, pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
plt.ylim(0, 100)
plt.xlim(left=1e-1)  # 确保对数坐标起始点

# 添加注释
plt.text(0.12, 0.95, 'Critical Operation Point @ SNDR=10dB',
         transform=plt.gcf().transFigure, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()

# 显示总结表格
plt.figure(figsize=(10, 4))
plt.axis('off')
columns = ['Parameter', 'Critical Value', 'Units']
cell_text = []

unit_mapping = {
    'ota_gain': '',
    'mdac_offset': '%FS',
    'cap_mismatch': '%',
    'comp_offset': '%FS',
    'nonlinear_factor': '',
    'bandwidth_ratio': '×Nyquist'
}

for param in param_ranges.keys():
    crit_val = critical_values[param]
    if crit_val is None:
        display_val = '> Test Range'
    else:
        # 格式化显示
        if param in ['ota_gain', 'bandwidth_ratio']:
            display_val = f"{crit_val:.1e}"
        elif param in unit_conversion:
            display_val = f"{crit_val * unit_conversion[param]:.2f}"
        else:
            display_val = f"{crit_val:.2f}"

    row = [
        param_labels[param],
        display_val,
        unit_mapping[param]
    ]
    cell_text.append(row)

table = plt.table(cellText=cell_text,
                  colLabels=columns,
                  loc='center',
                  cellLoc='center',
                  colColours=['#f0f0f0'] * 3)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('Critical Parameter Values @ SNDR=10dB', y=0.8, fontsize=12)

plt.tight_layout()
plt.show()

# 绘制频谱比较图
plt.figure(figsize=(12, 6))

# 理想情况
sndr_ideal, (X_ideal, f_pos) = simulate_pipeline_adc_with_non_idealities(x_in)
X_mag_ideal = 20 * np.log10(np.abs(X_ideal) + 1e-10)
plt.plot(f_pos / 1e6, X_mag_ideal, 'k--', alpha=0.7, label='Ideal ADC')

# 各非理想情况临界频谱
for param_name in param_ranges:
    crit_val = critical_values[param_name]
    if crit_val is None:
        continue

    test_params = {
        'ota_gain': 1000,
        'mdac_offset': 0,
        'cap_mismatch': 0,
        'comp_offset': 0,
        'nonlinear_factor': 0,
        'bandwidth_ratio': 0.5
    }
    test_params[param_name] = crit_val

    sndr, (X_crit, _) = simulate_pipeline_adc_with_non_idealities(x_in, **test_params)
    X_mag = 20 * np.log10(np.abs(X_crit) + 1e-10)

    label = f"{param_labels[param_name]} = {crit_val:.2g}"
    plt.plot(f_pos / 1e6, X_mag, alpha=0.7, label=label)

plt.grid(True)
plt.xlabel('Frequency (MHz)', fontsize=12)
plt.ylabel('Magnitude (dB)', fontsize=12)
plt.title('Output Spectrum Comparison @ Critical Parameters', fontsize=14)
plt.legend(fontsize=10, ncol=2)
plt.xlim(0, 250)
plt.ylim(-120, 10)
plt.tight_layout()
plt.show()