import numpy as np
import matplotlib.pyplot as plt


def transfer_function(f, Gm, cr, ch, a,  Ts, fir):

    fs = 1 / Ts  # 采样频率
    pi = np.pi

    # 避免除以零的情况
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(pi * Ts * f) / (pi * Ts * f)
        sinc_term = np.where(f == 0, 1.0, sinc_term)  # 处理f=0的情况

    fir = 1
    denominator = 1 - a
    magnitude = np.abs((Gm / (cr + ch )) * Ts * sinc_term * fir/denominator )

    return magnitude



if __name__ == "__main__":
    #
    Gm = 0.01
    cr = 0.5e-12
    ch = 15.425e-12
    a =  ch /(ch + cr)
    fclk=2.4e9
    ts = 1/ fclk
    N = 8
    Ts = N * ts
    fir = 1

    # 频率范围
    f = np.linspace(0, 1.2e9, 100000)  # 从0到2 MHz

    # 计算传输函数
    G = transfer_function(f, Gm, cr, ch, a,  Ts, fir)
    GdB = 20 * np.log10(G)
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(f, GdB)
    plt.title('Transfer Function Magnitude |G(f)|')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude(dB)')
    plt.grid(True)
    plt.show()