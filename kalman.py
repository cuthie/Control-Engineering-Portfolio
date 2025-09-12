import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Define system model
# -----------------------------
# State: [position, velocity]
# Dynamics: x_{k+1} = A x_k + w_k
# Measurement: y_k = C x_k + v_k

A = np.array([[1, 1],
              [0, 1]])
C = np.array([[1, 0]])

# Covariances
Q = np.array([[1e-4, 0],
              [0, 1e-4]])  # process noise
R = np.array([[1e-2]])  # measurement noise

# Simulation settings
T = 50
true_x = np.zeros((2, T))
measurements = np.zeros(T)

# Initial true state
true_x[:, 0] = [0, 1]

# Generate true states + noisy measurements
for k in range(1, T):
    w = np.random.multivariate_normal(mean=[0, 0], cov=Q)
    v = np.random.normal(0, np.sqrt(R))
    true_x[:, k] = A @ true_x[:, k - 1] + w
    measurements[k] = C @ true_x[:, k] + v


# -----------------------------
# 2. Kalman Filter Implementation
# -----------------------------
def kalman_filter(A, C, Q, R, y_seq, x0, P0):
    n = A.shape[0]
    N = len(y_seq)

    x_est = np.zeros((n, N))
    P = P0
    x_est[:, 0] = x0

    for k in range(1, N):
        # Prediction
        x_pred = A @ x_est[:, k - 1]
        P_pred = A @ P @ A.T + Q

        # Kalman Gain
        K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)

        # Update
        x_est[:, k] = x_pred + K @ (y_seq[k] - C @ x_pred)
        P = (np.eye(n) - K @ C) @ P_pred

    return x_est


# -----------------------------
# 3. Run the Kalman Filter
# -----------------------------
init_x = np.array([0, 0])  # initial guess
init_P = np.eye(2)

x_est = kalman_filter(A, C, Q, R, measurements, init_x, init_P)

# -----------------------------
# 4. Plot results
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(true_x[0, :], label='True Position')
plt.plot(measurements, 'r.', label='Measurements (noisy)')
plt.plot(x_est[0, :], 'g-', label='KF Estimate')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.title('Kalman Filter State Estimation')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(true_x[1, :], label='True Velocity')
plt.plot(x_est[1, :], 'g-', label='KF Estimate')
plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.legend()
plt.title('Velocity Estimation with KF (not directly measured)')
plt.grid(True)
plt.show()