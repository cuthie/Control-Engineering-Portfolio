import numpy as np
import matplotlib.pyplot as plt

# Simple discrete-time PID with anti-windup (clamping)
def pid_sim(Kp=2.0, Ki=1.0, Kd=0.1, Ts=0.02, N=20.0, umax=2.0, umin=-2.0, T=8.0):
    n = int(T / Ts)
    r = 1.0  # step reference
    y = 0.0
    e_prev = 0.0
    I = 0.0
    D = 0.0
    u_hist, y_hist, t = [], [], []
    # First-order plant: y[k+1] = a*y[k] + b*u[k]
    a = 0.9
    b = 0.2
    for k in range(n):
        e = r - y
        # derivative with filtered derivative (Tustin w/ filter N)
        D = (N*Kd*(e - e_prev) + D) / (1 + N*Ts)
        u_unsat = Kp*e + I + D
        u = np.clip(u_unsat, umin, umax)
        # anti-windup via clamping
        if (u != u_unsat):
            pass
        else:
            I += Ki*Ts*e
        # plant update
        y = a*y + b*u
        e_prev = e
        u_hist.append(u); y_hist.append(y); t.append(k*Ts)
    return np.array(t), np.array(y_hist), np.array(u_hist)

if __name__ == "__main__":
    t, y, u = pid_sim()
    plt.figure()
    plt.plot(t, y, label="y")
    plt.plot(t, u, label="u")
    plt.legend()
    plt.title("Discrete PID on 1st-order Plant")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()