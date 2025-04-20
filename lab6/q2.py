import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Base parameters (from your code)
fs = 500e6  # Sampling rate 500 MHz
fin = 200e6  # Input frequency 200 MHz
bits = 13  # ADC resolution
N = 8192  # Sample points
t = np.arange(N) / fs
x_in = 0.9 * np.sin(2 * np.pi * fin * t)  # Input signal
full_scale = 1.0  # Full scale range ±1V


# Function to calculate SNDR with MDAC offset
def calculate_sndr_with_offset(mdac_offset, stages=3, bits_per_stage=4):
    """
    Calculate SNDR for a pipeline ADC with MDAC offset

    Parameters:
    mdac_offset: Offset error in the MDAC (as a fraction of full scale)
    stages: Number of pipeline stages
    bits_per_stage: Bits per stage

    Returns:
    sndr: Signal-to-Noise-and-Distortion Ratio in dB
    """
    # Initialize residue from first stage
    residue = x_in.copy()
    adc_output = np.zeros_like(residue)

    # Process through pipeline stages
    for stage in range(stages):
        # Stage resolution
        if stage == stages - 1:
            # Last stage can have different resolution
            stage_bits = bits - (stages - 1) * bits_per_stage
        else:
            stage_bits = bits_per_stage

        stage_levels = 2 ** stage_bits
        stage_step = 2 * full_scale / stage_levels

        # Sub-ADC quantization
        stage_decision = np.round(residue / stage_step)
        stage_decision = np.clip(stage_decision, -stage_levels / 2, stage_levels / 2 - 1)

        # DAC output with offset (applied primarily to first stage as it's most critical)
        if stage == 0:
            dac_output = stage_decision * stage_step + mdac_offset * full_scale
        else:
            dac_output = stage_decision * stage_step

        # MDAC amplification and residue generation
        gain = 2 ** (stage_bits)
        if stage < stages - 1:  # No residue from last stage
            residue = gain * (residue - dac_output)
            # Apply clipping to model MDAC saturation
            residue = np.clip(residue, -full_scale, full_scale)

        # Accumulate result for this stage
        adc_output += stage_decision * (2 ** ((stages - stage - 1) * bits_per_stage))

    # Calculate the final quantized output
    x_adc = adc_output * (2 * full_scale / 2 ** bits)

    # Apply window for spectral analysis
    window = np.hanning(N)
    X = fft(x_adc * window)
    f = fftfreq(N, d=1 / fs)

    # Only positive frequencies
    X_half = X[:N // 2]
    f_pos = f[:N // 2]

    # Signal and noise bands
    sig_band = (f_pos >= 199e6) & (f_pos <= 201e6)
    noise_band = (f_pos > 0) & ~sig_band

    # Calculate signal and noise+distortion power
    signal_power = np.sum(np.abs(X_half[sig_band]) ** 2)
    noise_dist_power = np.sum(np.abs(X_half[noise_band]) ** 2)

    # Calculate SNDR
    sndr = 10 * np.log10(signal_power / noise_dist_power)

    return sndr, x_adc


# Test a range of offset values
offset_values = np.linspace(0, 0.1, 20)  # 0% to 10% of full scale
sndr_results = []

for offset in offset_values:
    sndr, _ = calculate_sndr_with_offset(offset)
    sndr_results.append(sndr)
    print(f"Offset: {offset * 100:.2f}% of FS, SNDR: {sndr:.2f} dB")

# Find where SNDR drops to 10 dB
offset_10db = np.interp(10, sndr_results[::-1], offset_values[::-1])
print(f"\nSNDR drops to 10 dB at offset = {offset_10db * 100:.4f}% of full scale")

# Plot SNDR vs offset
plt.figure(figsize=(10, 6))
plt.plot(offset_values * 100, sndr_results, 'o-')
plt.axhline(y=10, color='r', linestyle='--', label='SNDR = 10 dB')
plt.axvline(x=offset_10db * 100, color='g', linestyle='--',
            label=f'Offset = {offset_10db * 100:.4f}% FS')
plt.xlabel('MDAC Offset (% of Full Scale)')
plt.ylabel('SNDR (dB)')
plt.title('SNDR vs MDAC Input Offset')
plt.grid(True)
plt.legend()

# Plot example FFTs for best and worst cases
plt.figure(figsize=(12, 8))

# No offset case
sndr_no_offset, x_adc_no_offset = calculate_sndr_with_offset(0)
window = np.hanning(N)
X_no_offset = fft(x_adc_no_offset * window)
f = fftfreq(N, d=1 / fs)
X_half_no_offset = X_no_offset[:N // 2]
f_pos = f[:N // 2]
X_mag_no_offset = 20 * np.log10(np.abs(X_half_no_offset) + 1e-10)

# Critical offset case (10 dB SNDR)
sndr_critical, x_adc_critical = calculate_sndr_with_offset(offset_10db)
X_critical = fft(x_adc_critical * window)
X_half_critical = X_critical[:N // 2]
X_mag_critical = 20 * np.log10(np.abs(X_half_critical) + 1e-10)

# Plot comparison
plt.subplot(2, 1, 1)
plt.plot(f_pos / 1e6, X_mag_no_offset)
plt.title(f'FFT with No Offset (SNDR = {sndr_no_offset:.2f} dB)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f_pos / 1e6, X_mag_critical)
plt.title(f'FFT with Critical Offset of {offset_10db * 100:.4f}% FS (SNDR = {sndr_critical:.2f} dB)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)

plt.tight_layout()
plt.show()