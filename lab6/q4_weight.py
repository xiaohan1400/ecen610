import numpy as np
import matplotlib.pyplot as plt

def initialize_simulation():
    np.random.seed(42)
    step_size = 0.5
    stabilizer = 1e-5
    total_steps = 2000
    target_gain = 2
    ota_gain = 20
    return step_size, stabilizer, total_steps, target_gain, ota_gain

def generate_signals(steps: int, true_gain: float, ota_gain: float) -> tuple:
    input_signal = np.random.uniform(-0.9, 0.9, size=steps)
    ideal_signal = true_gain * input_signal
    nonideal_signal = true_gain * input_signal / (1 + true_gain / ota_gain)
    return input_signal, ideal_signal, nonideal_signal

def run_nlms_algorithm(input_data: np.ndarray, reference: np.ndarray,
                       mu: float, epsilon: float) -> tuple:
    steps = len(input_data)
    current_weight = 1.0
    weight_track = []
    error_track = []

    for step in range(steps):
        current_input = input_data[step]
        estimated_output = current_weight * current_input
        error = reference[step] - estimated_output
        normalization = current_input ** 2 + epsilon
        current_weight += mu * error * current_input / normalization

        weight_track.append(current_weight)
        error_track.append(error)

    return np.array(weight_track), np.array(error_track), current_weight

def visualize_results(weights: np.ndarray, errors: np.ndarray, correction_target: float):
    # 截取前40次迭代
    weights = weights[:40]
    errors = errors[:40]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(errors, color='blue', marker='o', markersize=4)
    ax1.set_title('Error Convergence (First 40 Steps)')
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Error Value')
    ax1.grid(True)

    ax2.plot(weights, label='Estimated Weight', color='green', marker='s', markersize=4)
    ax2.axhline(y=correction_target, color='red', linestyle='--', label='Target Correction')
    ax2.set_title('Weight Adaptation (First 40 Steps)')
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Weight Value')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    mu, eps, iterations, true_gain, ota_gain = initialize_simulation()
    input_signal, ideal_output, nonideal_output = generate_signals(iterations, true_gain, ota_gain)
    weight_history, error_history, final_weight = run_nlms_algorithm(
        nonideal_output, ideal_output, mu, eps
    )
    true_correction = 1 / (1 + true_gain / ota_gain)
    visualize_results(weight_history, error_history, true_correction)
    print(f"第40次迭代权重值: {weight_history[39]:.5f}")

if __name__ == "__main__":
    main()