import numpy as np
import matplotlib.pyplot as plt


def two_point_five_bit_stage(signal, vref=1.0):
    """
    实现2.5-bit流水线级（无冗余）的转换
    :param signal: 输入信号（需包含偏移）
    :param vref: 参考电压，默认1.0V
    :return: 量化后的输出
    """
    quant_unit = vref / 6
    quant_level = np.floor(signal / quant_unit)
    quant_level = np.clip(quant_level, 0, 5)
    return 6 * (signal - quant_level * quant_unit)


def compute_pipeline_output(input_signal, vref=1.0, offset=0.0625):
    """
    先对输入信号加上偏移，再经过2.5-bit流水线级量化处理，最后将输出限制在0-1范围内。
    :param input_signal: 未经处理的输入信号
    :param vref: 参考电压
    :param offset: 输入偏移
    :return: 量化后并限幅到[0,1]的输出信号
    """
    processed_signal = two_point_five_bit_stage(input_signal + offset, vref)
    return np.clip(processed_signal, 0, 1)


def generate_input_signal(fs, f_in, num_samples):
    """
    生成输入信号：全幅0.9V正弦波，中心电平0.5V
    :param fs: 采样率
    :param f_in: 信号频率
    :param num_samples: 采样点数
    :return: 时间向量和生成的正弦信号
    """
    t = np.arange(num_samples) / fs
    # 振幅0.5，使得正弦波幅值为0.5，加上直流0.5后信号总幅度为0.9V（相当于0.5±0.5）
    vin = 0.5 + 0.5 * np.sin(2 * np.pi * f_in * t)
    return t, vin


def plot_signals(time_vector, input_signal, output_signal):
    """
    绘制输入信号和经过2.5-bit流水线级转换后的输出信号。
    :param time_vector: 时间向量
    :param input_signal: 原始输入信号
    :param output_signal: 处理后的输出信号
    """
    # 时间单位转换为纳秒方便观察
    time_ns = time_vector * 1e9
    plt.figure(figsize=(10, 6))
    plt.plot(time_ns, input_signal, label='Input Signal (200 MHz sine)',color='blue', linewidth=2)
    plt.plot(time_ns, output_signal, label='2.5-bit No Redundancy Output',color='green',
             linewidth=2, linestyle='--')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.title('Input and Output of 2.5-bit Pipeline Stage (No Redundancy)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 参数设置（注意注释中提到1 GHz，但这里fs值为1000e9，即1000 GHz，根据原代码保持不变）
    fs = 1000e9  # 采样率：原代码注释为1 GHz，但此处数值为1000e9
    f_in = 200e6  # 正弦波频率200 MHz
    N = 10000  # 采样点数，用于可视化

    # 生成输入信号
    t, vin = generate_input_signal(fs, f_in, N)

    # 经过2.5-bit流水线级转换（先加偏移再限幅处理）
    vout = compute_pipeline_output(vin, vref=1.0, offset=0.0625)

    # 绘图展示输入和输出波形
    plot_signals(t, vin, vout)


if __name__ == "__main__":
    main()
