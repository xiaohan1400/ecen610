import numpy as np
import matplotlib.pyplot as plt

# Parameters
f_in = 1e9  # Input signal frequency: 1 GHz
f_s = 10e9  # Sampling frequency: 10 GHz
tau = 10e-12  # Time constant: 10 ps
R = 1000  # Resistance: 1 kOhm
C = tau / R  # Capacitance: 10 fF

# Time parameters
T_s = 1 / f_s  # Sampling period: 1 / 10 GHz = 100 ps
T_sim = 2e-9  # Total simulation time: 2 ns
dt = 1e-13  # Time step for simulation (small enough for accuracy)
t = np.arange(0, T_sim, dt)  # Time array

# Input signal: Sinusoidal at 1 GHz
Vin = np.sin(2 * np.pi * f_in * t)

# Sampling signal: Square wave at 10 GHz (50% duty cycle)
Vsw = np.zeros_like(t)
for i in range(len(t)):
    Vsw[i] = 1 if (t[i] % T_s) < (T_s / 2) else 0  # ON for half the sampling period

# Simulate the output voltage
Vout = np.zeros_like(t)
Vout[0] = 0  # Initial condition: Vout(0) = 0

# Numerical integration using Euler's method
for i in range(1, len(t)):
    if Vsw[i] == 1:  # Switch ON: RC circuit behavior
        dVout_dt = (1 / tau) * (Vin[i] - Vout[i-1])
        Vout[i] = Vout[i-1] + dVout_dt * dt
    else:  # Switch OFF: Hold the previous value
        Vout[i] = Vout[i-1]

# Plot the results
plt.figure(figsize=(12, 6))

# Plot input signal
plt.subplot(3, 1, 1)
plt.plot(t * 1e9, Vin, label='Input Voltage (Vin)', color='blue')
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.title('Input Signal (1 GHz Sinusoid)')
plt.grid(True)
plt.legend()

# Plot sampling signal
plt.subplot(3, 1, 2)
plt.plot(t * 1e9, Vsw, label='Sampling Signal (Vsw)', color='green')
plt.xlabel('Time (ns)')
plt.ylabel('Switch State')
plt.title('Sampling Signal (10 GHz)')
plt.grid(True)
plt.legend()

# Plot output signal
plt.subplot(3, 1, 3)
plt.plot(t * 1e9, Vout, label='Output Voltage (Vout)', color='red')
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.title('Output of ZOH Sampling Circuit')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()