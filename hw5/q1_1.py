import numpy as np
import matplotlib.pyplot as plt


def two_bit_quantizer(input_signal, ref_val=1.0):
    """
    Compute the residue of an ideal 2-bit ADC stage (no redundancy) with a 4-level quantizer.
    """
    quant_step = ref_val / 4.0
    quant_code = np.floor(input_signal / quant_step)
    quant_code = np.clip(quant_code, 0, 3)
    residue = 4 * (input_signal - quant_code * quant_step)
    return residue


def two_and_half_bit_no_overlap(input_signal, ref_val=1.0):
    """
    Compute the residue of a 2.5-bit ADC stage (no redundancy) with an ideal 5-level quantizer.
    """
    quant_step = ref_val / 6.0
    quant_code = np.floor(input_signal / quant_step)
    quant_code = np.clip(quant_code, 0, 5)
    residue = 6 * (input_signal - quant_code * quant_step)
    return residue


def two_and_half_bit_with_overlap(input_signal, ref_val=1.0):
    """
    Compute the residue of a 2.5-bit ADC stage (with redundancy) using a 7-level quantizer with overlapping thresholds.
    """
    quant_step = ref_val / 8.0
    thresholds = np.array([0.0, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 1.0])
    quant_code = np.digitize(input_signal, thresholds) - 1
    quant_code = np.clip(quant_code, 0, 6)
    residue = 4 * (input_signal - quant_code * quant_step)
    return residue


if __name__ == "__main__":
    # Define the input voltage range from 0 to 1 with 1000 points.
    vin = np.linspace(0, 1, 1000)

    # Compute residue outputs for each stage.
    res_2bit = two_bit_quantizer(vin)
    res_2p5_no_ov = two_and_half_bit_no_overlap(vin)
    res_2p5_with_ov = two_and_half_bit_with_overlap(vin)

    # Plot the transfer function for the 2-bit ADC stage
    plt.figure(figsize=(8, 5))
    plt.plot(vin, res_2bit, linestyle=':', color='red', linewidth=2, label='2-bit Stage Residue')
    plt.xlabel('Input Voltage [V]')
    plt.ylabel('Residue Output [V]')
    plt.title('Ideal 2-bit Stage Transfer Function')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot the transfer function for the 2.5-bit ADC stage without redundancy
    plt.figure(figsize=(8, 5))
    plt.plot(vin, res_2p5_no_ov, linestyle='-.', marker='.', markersize=3, color='blue',
             label='2.5-bit Stage (No Redundancy)')
    plt.xlabel('Input Voltage [V]')
    plt.ylabel('Residue Output [V]')
    plt.title('2.5-bit Stage Transfer Function without Redundancy')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot the transfer function for the 2.5-bit ADC stage with redundancy
    plt.figure(figsize=(8, 5))
    plt.plot(vin, res_2p5_with_ov, linestyle='-', marker='x', markersize=3, color='green',
             label='2.5-bit Stage (With Redundancy)')
    plt.xlabel('Input Voltage [V]')
    plt.ylabel('Residue Output [V]')
    plt.title('2.5-bit Stage Transfer Function with Redundancy')
    plt.legend(loc='best')
    plt.grid(True, linestyle='-.', alpha=0.7)
    plt.tight_layout()
    plt.show()
