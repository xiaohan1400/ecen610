import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk

# Given histogram data for 16 ADC codes
H = np.array([43, 115, 85, 101, 122, 170, 75, 146, 125, 60, 95, 95, 115, 40, 120, 242])

# Calculate cumulative histogram
C = np.cumsum(H)

# Endpoint correction: lowest and highest cumulative values
C_min = C[0]          # 43
C_max = C[-1]         # 1749
delta_C = C_max - C_min  # 1706

# For a 4-bit ADC, the ideal output codes range from 0 to 15
N_levels = 15

# Calculate the corrected digital values D for each code
D = (C - C_min) / delta_C * N_levels

# Calculate DNL for k=1,...,15 (difference between successive codes minus the ideal step of 1 LSB)
DNL = np.diff(D) - 1

# Calculate INL: deviation of each code from the ideal integer code
ideal_codes = np.arange(0, 16)
INL = D - ideal_codes

# Check for monotonicity: the corrected codes should be strictly increasing
is_monotonic = np.all(np.diff(D) > 0)

# Find the peak (maximum absolute) DNL and INL
peak_DNL = np.max(np.abs(DNL))
peak_INL = np.max(np.abs(INL))

# Create DataFrames for display
df_codes = pd.DataFrame({
    'Code': ideal_codes,
    'Cumulative': C,
    'Corrected D': np.round(D, 3),
    'INL (LSB)': np.round(INL, 3)
})

df_dnl = pd.DataFrame({
    'Transition (k)': np.arange(1, 16),
    'DNL (LSB)': np.round(DNL, 3)
})

# --- Tkinter GUI to display tables and summary information ---

# Create the main window
root = tk.Tk()
root.title("ADC Analysis Results")

# Create a Notebook widget with two tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both', padx=5, pady=5)

# Tab 1: Corrected Digital Values & INL
frame1 = ttk.Frame(notebook)
notebook.add(frame1, text="Corrected Digital Values & INL")

# Treeview for the df_codes DataFrame
tree1 = ttk.Treeview(frame1, columns=list(df_codes.columns), show='headings')
tree1.pack(expand=True, fill='both', padx=5, pady=5)
for col in df_codes.columns:
    tree1.heading(col, text=col)
    tree1.column(col, width=100, anchor='center')
for _, row in df_codes.iterrows():
    tree1.insert("", "end", values=list(row))

# Tab 2: Step-by-Step DNL
frame2 = ttk.Frame(notebook)
notebook.add(frame2, text="Step-by-Step DNL")

# Treeview for the df_dnl DataFrame
tree2 = ttk.Treeview(frame2, columns=list(df_dnl.columns), show='headings')
tree2.pack(expand=True, fill='both', padx=5, pady=5)
for col in df_dnl.columns:
    tree2.heading(col, text=col)
    tree2.column(col, width=100, anchor='center')
for _, row in df_dnl.iterrows():
    tree2.insert("", "end", values=list(row))

# Display summary information at the bottom of the main window
summary_text = (
    f"Peak DNL: {peak_DNL:.3f} LSB\n"
    f"Peak INL: {peak_INL:.3f} LSB\n"
    f"Is ADC Monotonic: {'Yes' if is_monotonic else 'No'}"
)
label_summary = tk.Label(root, text=summary_text, font=("Arial", 12), justify="left")
label_summary.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
