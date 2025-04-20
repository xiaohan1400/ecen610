import numpy as np
import matplotlib.pyplot as plt

# Known DNL values (in LSB) for 8 codes (3-bit ADC)
DNL = np.array([0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0])

# The ideal step is 1 LSB
ideal_step = 1

# Offset and full-scale error (in LSB)
offset_error = 0.5
full_scale_error = 0.5  # Ideal maximum code is 7, so actual T(7)=7+0.5=7.5

# Calculate the uncalibrated conversion thresholds T(i)
# T(0) = offset_error, then T(i) = T(i-1) + [ideal_step + DNL(i)]
T = np.zeros(8)
T[0] = offset_error
for i in range(1, 8):
    T[i] = T[i-1] + (ideal_step + DNL[i])

print("Uncalibrated conversion thresholds T(i):")
for i, t in enumerate(T):
    print(f"Code {i}: {t:.2f} LSB")

# Endpoint calibration: map T(0) to 0, and T(7) to 7
Y = T - T[0]  # Simple shift (here T(7)=7.5, T(0)=0.5, so Y ranges exactly from 0 to 7)

print("\nEndpoint calibrated conversion values Y(i):")
for i, y in enumerate(Y):
    print(f"Code {i}: {y:.2f} LSB")

# Calculate INL: defined as the deviation of the calibrated conversion values from the ideal values
# Ideal conversion values: ideally, Y_ideal(i) = i, for i=0,...,7
ideal_codes = np.arange(8)
INL = Y - ideal_codes

print("\nINL for each code (in LSB):")
for i, inl in enumerate(INL):
    print(f"Code {i}: {inl:.2f} LSB")

# Plot the transfer curves
plt.figure(figsize=(8, 5))
# Plot the calibrated ADC transfer curve
plt.plot(ideal_codes, Y, 'o-', label="ADC Transfer Curve")
# Plot the ideal straight line
plt.plot(ideal_codes, ideal_codes, 'k--', label="Ideal Line (Y = i)")
plt.xlabel("Ideal Code")
plt.ylabel("Output (LSB)")
plt.title("Comparison of 3-bit ADC Transfer Characteristics")
plt.legend()
plt.grid(True)
plt.show()
