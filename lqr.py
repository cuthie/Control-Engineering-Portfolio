import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

def dlqr(A,B,Q,R):
    P = solve_discrete_are(A,B,Q,R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, P

def simulate(A,B,C, K, r=1.0, steps=300):
    # reference prefilter for SISO: compute Nbar via dc gain
    M = np.block([[np.eye(A.shape[0]) - A, B],
                  [C, np.zeros((C.shape[0], B.shape[1]))]])
    rhs = np.vstack((np.zeros((A.shape[0],1)), np.array([[1.0]])))
    sol = np.linalg.pinv(M) @ rhs
    Nbar = float(sol[A.shape[0]:])
    x = np.zeros((A.shape[0],1)); y_hist=[]; u_hist=[]
    for k in range(steps):
        u = -K @ x + Nbar*r
        x = A @ x + B @ u
        y = C @ x
        y_hist.append(float(y)); u_hist.append(float(u))
    return np.arange(steps), np.array(y_hist), np.array(u_hist)

if __name__ == "__main__":
    # Discrete mass–spring–damper example
    A = np.array([[1.0, 0.02],
                  [-0.02, 0.98]])
    B = np.array([[0.0],[0.02]])
    C = np.array([[1.0, 0.0]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    K,_ = dlqr(A,B,Q,R)
    t,y,u = simulate(A,B,C,K)
    plt.figure(); plt.plot(t*0.02, y, label="y"); plt.plot(t*0.02, u, label="u")
    plt.legend(); plt.title("Discrete LQR Tracking"); plt.xlabel("Time [s]")
    plt.show()