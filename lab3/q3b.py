import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def transfer_function(f, Gm, cr, ch, a, Ts, fir, L):
    fs = 1 / Ts
    pi = np.pi

    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(pi * Ts * f) / (pi * Ts * f)
        sinc_term = np.where(f == 0, 1.0, sinc_term)

    denominator = 1 - a
    magnitude = np.abs((Gm / (cr + ch)) * Ts * sinc_term * fir / denominator)

    # 计算频率响应
    length3a = np.ones(L)
    denominator1 = [1]
    x, y = signal.freqz(length3a, denominator1, worN=f, fs=fs)  # 直接使用 f (Hz)

    magq3a = magnitude * np.abs(y)


    f_integrator = f.copy()
    f_integrator[f_integrator == 0] = np.finfo(float).eps  # Replace 0 with smallest positive float

    xq3b, yq3b = signal.freqz(length3a, denominator1, worN=2 * np.pi * f_integrator / fs, whole=False)

    # 5. Cascade all
    magq3b = magq3a * np.abs(yq3b)

    return magq3b

if __name__ == "__main__":
    Gm = 0.01
    cr = 0.5e-12
    ch = 15.425e-12
    a = ch / (ch + cr)
    fclk = 2.4e9
    ts = 1 / fclk
    N = 8
    Ts = N * ts
    fir = 1
    L = 4

    f = np.linspace(1, 1.2e9, 100000)
    G = transfer_function(f, Gm, cr, ch, a, Ts, fir, L)
    G = np.maximum(G, 1e-12)  # 避免 log10(0)
    GdB = 20 * np.log10(G)

    plt.figure(figsize=(10, 6))
    plt.plot(f, GdB, '-', drawstyle='steps-post')
    plt.title('Transfer Function Magnitude |G(f)|')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.show()