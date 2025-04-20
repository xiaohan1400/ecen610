import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional


class ADCSimulator:
    """
    ADC (Analog-to-Digital Converter) simulator with non-ideal effects and compensation.
    This class simulates a continuous-time sigma-delta modulator with various non-idealities
    and provides facilities for non-linear compensation using LMS algorithm.
    """

    def __init__(self,
                 sampling_freq: float = 500e6,  # 500 MHz
                 sample_count: int = 8192,
                 reference_voltage: float = 1.0):
        """
        Initialize the ADC simulator with configuration parameters.

        Args:
            sampling_freq: Sampling frequency in Hz
            sample_count: Number of samples to simulate
            reference_voltage: Reference voltage for the ADC
        """
        self.fs = sampling_freq
        self.N = sample_count
        self.v_ref = reference_voltage
        self.time = np.arange(self.N) / self.fs

        # Default non-ideality parameters
        self.gain = 2.0  # Integrator gain
        self.ota_gain = 25.0  # Finite OTA gain
        self.cap_mismatch = 0.03  # Capacitor mismatch factor
        self.ota_offset = 0.02  # OTA input offset voltage
        self.comp_offset = 0.01  # Comparator offset voltage

        # OTA bandwidth limitation (80% of Nyquist)
        self.bandwidth_factor = 0.8
        self._update_bandwidth_alpha()

        # Non-linear distortion coefficients
        self._update_nonlinear_coeffs()

    def _update_bandwidth_alpha(self):
        """Update the alpha coefficient based on bandwidth settings"""
        bandwidth = self.bandwidth_factor * self.fs
        self.alpha = np.exp(-1 / (self.fs * (1 / (2 * np.pi * bandwidth))))

    def _update_nonlinear_coeffs(self):
        """Update the polynomial distortion coefficients"""
        G = self.gain
        self.nonlinear_coeffs = {
            1: G / (1 + G / self.ota_gain),  # Linear gain with finite OTA effect
            2: 0.1 * G,  # 2nd order non-linearity
            3: 0.2 * G,  # 3rd order non-linearity
            4: 0.15 * G,  # 4th order non-linearity
            5: 0.1 * G  # 5th order non-linearity
        }

    def generate_input_signal(self, frequency: float = 200e6, amplitude: float = 0.9) -> np.ndarray:
        """
        Generate a sinusoidal input signal.

        Args:
            frequency: Input signal frequency in Hz
            amplitude: Signal amplitude (0 to 1)

        Returns:
            Input signal array
        """
        return amplitude * np.sin(2 * np.pi * frequency * self.time)

    def simulate_adc_output(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Simulate ADC output with all non-ideal effects.

        Args:
            input_signal: Input signal to the ADC

        Returns:
            ADC output signal with non-idealities
        """
        N = len(input_signal)
        output = np.zeros(N)

        # Comparator stage with offset
        digital_out = np.sign(input_signal + self.comp_offset)

        # First stage feedforward signal
        first_stage_in = input_signal - digital_out * self.v_ref

        # Calculate effective linear gain including mismatch and finite gain
        effective_gain = (1 + self.cap_mismatch) * (self.gain / (1 + self.gain / self.ota_gain))

        # Linear part with offset
        linear_response = effective_gain * first_stage_in + self.ota_offset

        # Apply bandwidth limitations and non-linearities
        for n in range(1, N):
            # Bandwidth-limited response (first-order IIR filter)
            bandwidth_limited = self.alpha * output[n - 1] + (1 - self.alpha) * linear_response[n]

            # Add non-linear distortion terms
            nonlinear_terms = sum(self.nonlinear_coeffs[k] * input_signal[n] ** k
                                  for k in range(2, 6))

            # Complete output
            output[n] = bandwidth_limited + nonlinear_terms

        return output


class NonlinearCompensator:
    """
    Non-linear compensator using polynomial LMS algorithm.
    This class implements polynomial-based least mean square algorithm
    for non-linear system identification and compensation.
    """

    def __init__(self, max_order: int = 5):
        """
        Initialize the non-linear compensator.

        Args:
            max_order: Maximum polynomial order
        """
        self.max_order = max_order
        self.weights = np.zeros(max_order)
        self.weight_history = []
        self.error_history = []

    @staticmethod
    def _create_feature_matrix(x: np.ndarray, order: int) -> np.ndarray:
        """
        Create polynomial feature matrix from input signal.

        Args:
            x: Input signal
            order: Polynomial order

        Returns:
            Feature matrix where each column is x^(i+1)
        """
        return np.vstack([x ** (i + 1) for i in range(order)]).T

    def train(self, input_signal: np.ndarray,
              reference_signal: np.ndarray,
              learning_rate: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the compensator using LMS algorithm.

        Args:
            input_signal: Input signal
            reference_signal: Target/reference signal
            learning_rate: LMS algorithm learning rate

        Returns:
            Tuple of (weight_history, error_history)
        """
        N = len(input_signal)
        self.weights = np.zeros(self.max_order)
        self.weight_history = []
        self.error_history = []

        # Create feature matrix with polynomial terms
        feature_matrix = self._create_feature_matrix(input_signal, self.max_order)

        # Run LMS algorithm
        for n in range(N):
            # Current features
            features = feature_matrix[n]

            # Predicted output
            prediction = np.dot(self.weights, features)

            # Calculate error
            error = reference_signal[n] - prediction

            # Update weights
            self.weights += learning_rate * error * features

            # Store history
            self.weight_history.append(self.weights.copy())
            self.error_history.append(error)

        return np.array(self.weight_history), np.array(self.error_history)

    def predict(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Apply the compensator to new input data.

        Args:
            input_signal: Input signal to compensate

        Returns:
            Compensated signal
        """
        features = self._create_feature_matrix(input_signal, self.max_order)
        return np.sum(features * self.weights, axis=1)


class ResultVisualizer:
    """
    Visualization utilities for ADC simulation results.
    """

    @staticmethod
    def plot_learning_curves(error_history: np.ndarray,
                             weight_history: np.ndarray,
                             ideal_weights: Optional[Dict[int, float]] = None,
                             figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot LMS learning curves and weight convergence.

        Args:
            error_history: Error history from LMS algorithm
            weight_history: Weight history from LMS algorithm
            ideal_weights: Dictionary of ideal weight values (if known)
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        # # Error plot
        # plt.subplot(1, 2, 1)
        # plt.plot(error_history)
        # plt.title("Error vs Iteration (LMS)")
        # plt.xlabel("Iteration")
        # plt.ylabel("Error")
        # plt.grid(True, alpha=0.3)

        # Weight convergence plot
        plt.subplot(1, 2, 2)
        order = weight_history.shape[1]

        for i in range(order):
            plt.plot(weight_history[:, i], label=f'w{i + 1}')

            # Add reference line for ideal weight if available
            if ideal_weights and (i + 1) in ideal_weights:
                plt.axhline(y=ideal_weights[i + 1], linestyle='--',
                            color='gray', alpha=0.7, label=f'ideal w{i + 1}')

        # plt.title("Weight Convergence")
        # plt.xlabel("Iteration")
        # plt.ylabel("Weight")
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        #
        # plt.tight_layout()
        # plt.show()

    @staticmethod
    def compare_weights(estimated_weights: np.ndarray,
                        ideal_weights: Dict[int, float],
                        order: int) -> None:
        """
        Print comparison between estimated and ideal weights.

        Args:
            estimated_weights: Final estimated weights
            ideal_weights: Dictionary of ideal weights
            order: Maximum polynomial order
        """
        print("\nLMS Weight Comparison:")
        print("=" * 50)
        print(f"{'Order':<5} {'Estimated':<15} {'Ideal':<15} {'Error (%)':<10}")
        print("-" * 50)

        for i in range(order):
            ideal = ideal_weights.get(i + 1, 0)
            estimated = estimated_weights[i]
            error_pct = abs((estimated - ideal) / (ideal + 1e-10)) * 100 if ideal != 0 else float('inf')

            print(f"{i + 1:<5} {estimated:15.5f} {ideal:15.5f} {error_pct:10.2f}")

        print("=" * 50)


def main():
    """Main function to run the ADC simulation and compensation."""
    # Initialize the ADC simulator
    adc = ADCSimulator(sampling_freq=500e6, sample_count=8192)

    # Generate input signal (200 MHz sine wave)
    input_signal = adc.generate_input_signal(frequency=200e6, amplitude=0.9)

    # Simulate ADC output with non-idealities
    adc_output = adc.simulate_adc_output(input_signal)

    # Initialize the non-linear compensator
    compensator = NonlinearCompensator(max_order=5)

    # Train the compensator with LMS
    weight_history, error_history = compensator.train(
        input_signal=input_signal,
        reference_signal=adc_output,
        learning_rate=0.002  # Adjusted for better convergence
    )

    # Get final weights
    final_weights = weight_history[-1]

    # Visualize results
    visualizer = ResultVisualizer()
    visualizer.plot_learning_curves(
        error_history=error_history,
        weight_history=weight_history,
        ideal_weights=adc.nonlinear_coeffs
    )

    # Compare estimated weights with ideal values
    visualizer.compare_weights(
        estimated_weights=final_weights,
        ideal_weights=adc.nonlinear_coeffs,
        order=compensator.max_order
    )

    # Optional: Plot compensated signal vs original
    compensated_output = compensator.predict(input_signal)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(adc.time[:100] * 1e9, input_signal[:100])
    plt.title('Input Signal (First 100 samples)')
    plt.xlabel('Time (ns)')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(adc.time[:100] * 1e9, adc_output[:100])
    plt.title('ADC Output with Non-idealities (First 100 samples)')
    plt.xlabel('Time (ns)')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(adc.time[:100] * 1e9, compensated_output[:100])
    plt.title('Compensated Output (First 100 samples)')
    plt.xlabel('Time (ns)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()