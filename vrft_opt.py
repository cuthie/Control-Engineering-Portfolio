import numpy as np
from scipy.signal import lfilter
from numpy.linalg import lstsq

# -----------------------------
# STEP 1: Define reference model M(q)
# -----------------------------
# Example: M(q) = b0 / (1 - a1 q^-1)  (simple first-order system)
# We'll define numerator and denominator
M_num = [0.5]         # numerator (b0)
M_den = [1, -0.5]     # denominator (1 - a1 q^-1)

# -----------------------------
# STEP 2: Collect data (u, y)
# -----------------------------
# Here we simulate or assume you already have process data
N = 200
u = np.random.randn(N)           # excitation input
# Example: true plant y = 0.7 y[k-1] + 0.2 u[k-1] + noise
y = np.zeros(N)
for k in range(1, N):
    y[k] = 0.7*y[k-1] + 0.2*u[k-1] + 0.05*np.random.randn()

# -----------------------------
# STEP 3: Construct virtual reference and error
# -----------------------------
# r_v = M^-1(q) y
# First invert M (apply filter defined by denominator/ numerator swapped)
# That is, r_v = lfilter(M_den, M_num, y)
r_v = lfilter(M_den, M_num, y)

# Virtual error
e_v = r_v - y

# -----------------------------
# STEP 4: Define controller structure C(q, θ)
# -----------------------------
# Let's assume a simple PID-like structure:
# C(q,θ) u = θ1 * e[k] + θ2 * e[k-1]
# So regressor φ[k] = [e[k], e[k-1]]
phi = np.column_stack([e_v[1:], e_v[:-1]])

# -----------------------------
# STEP 5: Solve least-squares problem
# -----------------------------
# We want C(q,θ) e_v ≈ u (the input that would produce desired closed-loop)
theta, _, _, _ = lstsq(phi, u[1:], rcond=None)

# -----------------------------
# STEP 6: Tuned controller
# -----------------------------
print("Tuned controller parameters (θ):", theta)

import matplotlib.pyplot as plt

# Assume theta from VRFT:
theta1, theta2 = theta

# -----------------------------
# STEP 7: Apply the tuned controller C(q, θ)
# -----------------------------
# Define closed-loop with plant: y[k] = 0.7 y[k-1] + 0.2 u[k-1]
# Controller: u[k] = θ1*e[k] + θ2*e[k-1], where e[k] = r[k] - y[k]

N_sim = 100
r = np.ones(N_sim)   # step reference
y_cl = np.zeros(N_sim)
u_cl = np.zeros(N_sim)
e_cl = np.zeros(N_sim)

for k in range(1, N_sim):
    e_cl[k] = r[k] - y_cl[k]
    u_cl[k] = theta1*e_cl[k] + theta2*e_cl[k-1]
    y_cl[k] = 0.7*y_cl[k-1] + 0.2*u_cl[k-1]   # true plant dynamics

# -----------------------------
# STEP 8: Compare with reference model response
# -----------------------------
# Simulate desired closed-loop reference model M(q)
y_ref = lfilter(M_num, M_den, r)

# -----------------------------
# STEP 9: Plot results
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(y_ref, 'g--', label="Reference model (desired)")
plt.plot(y_cl, 'b', label="VRFT closed-loop output")
plt.plot(r, 'k:', label="Reference input")
plt.xlabel("Time step")
plt.ylabel("Output")
plt.legend()
plt.title("VRFT Controller Performance vs Reference Model")
plt.grid(True)
plt.show()