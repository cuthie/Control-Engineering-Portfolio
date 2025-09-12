import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

def dlqr(A,B,Q,R):
    P = solve_discrete_are(A,B,Q,R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, P

def kalman_filter_gain(A,C,Qn,Rn):
    # Solve steady-state Riccati for estimator
    P = solve_discrete_are(A.T, C.T, Qn, Rn)
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + Rn)
    return L, P

if __name__ == "__main__":
    # Discrete plant
    Ts = 0.02
    A = np.array([[1.0, Ts],
                  [-0.02, 0.98]])
    B = np.array([[0.0],[0.02]])
    C = np.array([[1.0, 0.0]])
    # LQR
    Q = np.diag([10.0, 1.0]); R = np.array([[0.1]])
    K,_ = dlqr(A,B,Q,R)
    # Kalman (process/measurement noise)
    Qn = np.diag([1e-3, 1e-3]); Rn = np.array([[2e-3]])
    L,_ = kalman_filter_gain(A,C,Qn,Rn)
    # Simulation
    x = np.zeros((2,1)); xhat = np.zeros((2,1))
    r = 1.0
    y_hist = []; u_hist = []; ymeas_hist = []
    for k in range(600):
        # reference prefilter Nbar
        M = np.block([[np.eye(2)-A, B],[C, np.zeros((1,1))]])
        rhs = np.vstack((np.zeros((2,1)), np.array([[1.0]])))
        sol = np.linalg.pinv(M) @ rhs
        Nbar = float(sol[2:])
        # control using estimated state
        u = -K @ xhat + Nbar*r
        # true plant + process noise
        w = np.random.multivariate_normal([0,0], Qn).reshape(2,1)
        v = np.random.normal(0, np.sqrt(Rn[0,0]), size=(1,1))
        x = A @ x + B @ u + w
        y = C @ x
        ymeas = y + v
        # estimator update
        xhat = A @ xhat + B @ u + L @ (ymeas - C @ (A @ xhat + B @ u))
        y_hist.append(float(y)); u_hist.append(float(u)); ymeas_hist.append(float(ymeas))
    t = np.arange(len(y_hist))*Ts
    plt.figure()
    plt.plot(t, y_hist, label="y (true)")
    plt.plot(t, ymeas_hist, label="y (meas)")
    plt.plot(t, u_hist, label="u")
    plt.legend(); plt.title("LQG (LQR + Kalman)"); plt.xlabel("Time [s]")
    plt.show()