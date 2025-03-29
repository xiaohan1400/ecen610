import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def transfer_function(f, Gm, Cr_list, ch, Ts, fir, L):
    fs = 1 / Ts
    pi = np.pi

    # Calculate a values for each Cr
    a_values = [ch / (ch + cr) for cr in Cr_list]

    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_term = np.sin(pi * Ts * f) / (pi * Ts * f)
        sinc_term = np.where(f == 0, 1.0, sinc_term)

    # Calculate individual transfer functions for each stage
    magnitudes = []
    for i, (cr, a) in enumerate(zip(Cr_list, a_values)):
        # Create numerator and denominator for each stage
        numerator = [0] * i + [1]  # [1], [0,1], [0,0,1], [0,0,0,1]
        denominator = [1, -a]

        # Calculate frequency response for each  stage
        w, h = signal.freqz(numerator, denominator, worN=2 * np.pi * f / fs, fs=fs)

        # Calculate individual magnitude for this stage
        mag = np.abs((Gm / (cr + ch)) * Ts * sinc_term * fir * h)
        magnitudes.append(mag)

    # Sum all individual magnitudes
    sum_magnitudes = np.sum(magnitudes, axis=0)

    # Calculate original magnitude
    w, h_original = signal.freqz(np.ones(L), [1], worN=2 * np.pi * f / fs, fs=fs)
    original_magnitude = np.abs((Gm / (Cr_list[0] + ch)) * Ts * sinc_term * fir * h_original)

    # Combine by multiplying summed magnitudes with original magnitude
    combined_magnitude = sum_magnitudes * original_magnitude

    return combined_magnitude


if __name__ == "__main__":
    # Parameters
    Gm = 0.01
    Cr_list = [0.5e-12, 1e-12, 2e-12, 4e-12]
    ch = 15.425e-12
    fclk = 2.4e9
    ts = 1 / fclk
    N = 8
    Ts = N * ts
    fir = 1
    L = 4

    # Frequency range
    f = np.linspace(1, 1.2e9, 100000)

    # Calculate transfer function
    G = transfer_function(f, Gm, Cr_list, ch, Ts, fir, L)
    G = np.maximum(G, 1e-12)  # Avoid log10(0)
    GdB = 20 * np.log10(G)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, GdB, '-', linewidth=2)
    plt.title('Combined Transfer Function Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="-")
    plt.xlim(1e6, 1.2e9)
    plt.show()