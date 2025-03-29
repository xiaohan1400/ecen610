import numpy as np
import matplotlib.pyplot as plt

def transfer_function_modified(f, Gm, Cs, Ts):
    fs = 1 / Ts  # 采样频率
    pi = np.pi

    # 1. 原始采样和保持的 sinc 响应
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(pi * Ts * f) / (pi * Ts * f)
        sinc_term = np.where(f == 0, 1.0, sinc_term)  # 处理 f=0 的情况

        magnitude = (Gm / Cs) * Ts * sinc_term

    return np.abs(magnitude)

# 示例使用
if __name__ == "__main__":
    #
    Gm = 1e-3
    Cs = 1e-12
    Cr = 0.5e-12
    Ch = 15.425e-12
    fclk = 2.4e9
    ts = 1 / fclk
    N = 8
    Ts = N * ts

    # 频率范围
    f = np.linspace(0, 2.4e9, 100000)  # 0 到 2.4 GHz

    # 计算原始和改进后的传递函数
    G_original = transfer_function_modified(f, Gm, Cs, Ts)  # 原始情况（无 Cr/Ch）
    G_original_dB = 20 * np.log10(G_original)

    #
    plt.figure(figsize=(10, 6))
    plt.plot(f, G_original, label='Original')
    plt.title('Transfer Function Magnitude Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude(dB)')
    plt.legend()
    plt.grid(True)
    plt.show()