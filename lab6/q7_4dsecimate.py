import numpy as np
import matplotlib.pyplot as plt


# =============================================
# Module 1: Multi-tone Signal Generation
# =============================================
def generate_multitone_waveform():
    sample_rate = 500e6  # 500 MHz sampling
    num_samples = 4096  # Total samples
    bandwidth = 200e6  # 200 MHz bandwidth
    num_tones = 128  # Number of subcarriers

    time_axis = np.arange(num_samples) / sample_rate
    freq_points = np.linspace(-bandwidth / 2, bandwidth / 2, num_tones)
    bpsk_data = np.random.choice([-1 + 0j, 1 + 0j], size=num_tones)

    baseband_waveform = np.zeros(num_samples, dtype=complex)
    for idx, freq in enumerate(freq_points):
        carrier = np.exp(2j * np.pi * freq * time_axis)
        baseband_waveform += bpsk_data[idx] * carrier

    return np.real(baseband_waveform), time_axis


# =============================================
# Module 2: Nonlinear Distortion Model
# =============================================
def apply_nonlinear_distortion(signal_in, coeffs=(0.1, 0.1, 0.1)):
    a2, a3, a4 = coeffs
    return signal_in * (1 + a2 * signal_in + a3 * signal_in ** 2 + a4 * signal_in ** 3)


# =============================================
# Module 3: Imperfect Pipeline ADC Simulation
# =============================================
class PipelineADC:
    def __init__(self, num_stages=6):
        self.stages = num_stages
        np.random.seed(0)
        self.gain_errors = np.random.normal(0, 0.01, num_stages)
        self.offset_errors = np.random.normal(0, 0.01, num_stages)
        self.cap_mismatch = np.random.normal(0, 0.005, num_stages)
        self.comp_offsets = np.random.normal(0, 0.005, num_stages)

    def convert(self, analog_signal):
        residual = analog_signal.copy()
        digital_output = np.zeros_like(analog_signal)

        for stage in range(self.stages):
            quantized = np.round(residual * 4) / 4
            quantized += self.comp_offsets[stage]
            digital_output += quantized

            error_signal = residual - quantized
            scaled_error = error_signal * (2 + self.gain_errors[stage]) * (1 + self.cap_mismatch[stage])
            residual = apply_nonlinear_distortion(scaled_error) + self.offset_errors[stage]

        return digital_output


# =============================================
# Module 4: Adaptive Calibration Engine
# =============================================
class LMS_Calibrator:
    def __init__(self, num_stages, learning_rate=0.01):
        self.weights = np.zeros(num_stages)
        self.mu = learning_rate

    def process(self, ideal_signal, adc_output, downsample_factor):
        calibrated_output = np.zeros_like(ideal_signal)
        error_history = np.zeros(len(ideal_signal))

        for n in range(len(ideal_signal)):
            if n % downsample_factor != 0:
                error_history[n] = error_history[n - 1]
                continue

            stage_features = []
            residual = adc_output[n]

            for _ in range(len(self.weights)):
                quantized = np.round(residual * 4) / 4
                stage_features.append(quantized)
                residual = (residual - quantized) * 2

            estimated = np.dot(self.weights, stage_features)
            error = ideal_signal[n] - estimated
            self.weights += self.mu * error * np.array(stage_features)
            error_history[n] = error

        return error_history


# =============================================
# Module 5: Visualization System
# =============================================
def plot_convergence(results):
    plt.figure(figsize=(10, 5.5))
    colors = ['#2E86C1', '#17A589', '#D4AC0D', '#CB4335']

    for idx, (factor, data) in enumerate(results.items()):
        error_power = 10 * np.log10(data ** 2 + 1e-12)
        plt.plot(error_power, color=colors[idx],
                 label=f'D={factor}', linewidth=1.5)

    plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio() * 0.3)
    plt.title('Adaptive Calibration Performance', fontsize=12)
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel('Error Power (dB)', fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# =============================================
# Main Processing Flow
# =============================================
if __name__ == "__main__":
    # Signal generation
    clean_signal, _ = generate_multitone_waveform()

    # ADC simulation
    adc = PipelineADC(num_stages=6)
    distorted_signal = adc.convert(clean_signal)

    # Calibration experiments
    calibration_factors = [10, 100, 1000, 10000]
    convergence_data = {}

    for factor in calibration_factors:
        calibrator = LMS_Calibrator(num_stages=6, learning_rate=0.01)
        err = calibrator.process(clean_signal, distorted_signal, factor)
        convergence_data[factor] = err

    # Result visualization
    plot_convergence(convergence_data)