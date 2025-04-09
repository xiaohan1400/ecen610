import numpy as np
import matplotlib.pyplot as plt


def stage_2p5bit_with_redundancy(x, vref=1.0):
    """
    采用冗余设计的 2.5-bit 流水线级量化函数
    :param x: 输入信号（在进入量化前已经加入偏移）
    :param vref: 参考电压，默认1.0V
    :return: 量化后的信号
    """
    q = vref / 8
    thresholds = np.array([0.0, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 1.0])
    d = np.digitize(x, thresholds) - 1
    d = np.clip(d, 0, 6)
    return 4 * (x - d * q)


def generate_input_signal(fs, f_in, N):
    """
    生成输入信号
    输入信号为全幅约 0.9V 的正弦波（中心 0.5V）
    :param fs: 采样率（Hz）
    :param f_in: 正弦波频率（Hz）
    :param N: 采样点数
    :return: 时间向量 t 和 输入信号 vin
    """
    t = np.arange(N) / fs
    vin = 0.5 + 0.5 * np.sin(2 * np.pi * f_in * t)
    return t, vin


def process_signal(vin, vref=1.0, offset=0.0625):
    """
    对输入信号先加入偏移，再调用冗余的 2.5-bit 量化函数，
    最后将输出裁剪到 [0,1]
    :param vin: 原始输入信号
    :param vref: 参考电压，默认1.0V
    :param offset: 量化前的输入偏移，默认0.0625
    :return: 处理后的输出信号 vout
    """
    quantized = stage_2p5bit_with_redundancy(vin + offset, vref)
    return np.clip(quantized, 0, 1)


def plot_signals(t, vin, vout):
    """
    绘制时域信号图
    将时间单位转换为纳秒进行展示
    :param t: 时间向量（单位：秒）
    :param vin: 输入信号
    :param vout: 处理后的输出信号
    """
    t_ns = t * 1e9  # 秒转换为纳秒
    plt.figure(figsize=(10, 6))
    plt.plot(t_ns, vin, label='Input Signal (200 MHz sine)', linewidth=2)
    plt.plot(t_ns, vout, label='2.5-bit With Redundancy Output', color='green', linestyle='--', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.title('Input and Output of 2.5-bit Stage (With Redundancy)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 参数设置
    fs = 1000e9  # 采样率（1 GHz —— 注意，原代码数值与注释可能不一致）
    f_in = 200e6  # 输入信号频率: 200 MHz
    N = 10000  # 采样点数（用于画图）

    # 生成信号
    t, vin = generate_input_signal(fs, f_in, N)

    # 加偏移、量化及裁剪处理
    vout = process_signal(vin, vref=1.0, offset=0.0625)

    # 绘制时域波形
    plot_signals(t, vin, vout)


if __name__ == '__main__':
    main()
