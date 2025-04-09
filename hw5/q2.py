import numpy as np
import matplotlib.pyplot as plt

def two_bit_stage(signal, v_ref=1.0):
    """
    Compute the output of an ideal 2-bit stage (no redundancy) using a 4-level quantizer.
    """
    quant = v_ref / 4.0
    code = np.floor(signal / quant)
    code = np.clip(code, 0, 3)
    return 4 * (signal - code * quant)

def two_bit_stage_with_offset(signal, v_ref=1.0, offset=0.0625):
    """
    Compute the 2-bit stage output after applying an offset.
    The input is shifted by the offset and the resulting output is clipped between 0 and 1.
    """
    shifted_output = two_bit_stage(signal + offset, v_ref)
    return np.clip(shifted_output, 0, 1)

if __name__ == '__main__':
    # Signal parameters
    sample_rate = 100e9            # Sampling rate (100 GHz as given)
    sine_freq   = 200e6            # 200 MHz sine wave
    num_samples = 10000            # Number of samples
    time_axis   = np.arange(num_samples) / (11 * sample_rate)  # Time axis in seconds

    # Generate a full-scale 1 V sine wave centered at 0.5 V (0.9V peak-to-peak)
    input_signal = 0.5 + 0.5 * np.sin(2 * np.pi * sine_freq * time_axis)

    # Process the signal through the 2-bit stage with offset
    output_signal = two_bit_stage_with_offset(input_signal)

    # Plotting: display input and processed signals
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis * 1e9, input_signal,color='blue', linewidth=2,
             label='Input Signal (200 MHz sine, FS=1V)')
    plt.plot(time_axis * 1e9, output_signal,color='green', linestyle='--', linewidth=2,
             label='2bit with offset')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.title('Input and Output of 2-bit Pipeline Stage (with offset)')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
