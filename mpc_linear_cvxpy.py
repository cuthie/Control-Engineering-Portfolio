import numpy as np
import matplotlib.pyplot as plt

# Note: requires cvxpy installed. We do not import CVXPY at module import to avoid errors.
def mpc_siso():
    import cvxpy as cp
    Ts = 0.1
    A = np.array([[1.0, Ts],
                  [0.0, 1.0]])
    B = np.array([[0.0],[Ts]])
    C = np.array([[1.0, 0.0]])
    nx, nu = A.shape[0], B.shape[1]
    N = 20
    Q = np.diag([10.0, 1.0])
    R = 0.1*np.eye(nu)
    x0 = np.array([[0.0],[0.0]])
    r = 1.0
    # Variables
    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))
    cost = 0
    constr = [x[:,0] == x0.flatten()]
    for k in range(N):
        cost += cp.quad_form(x[:,k] - np.array([r,0.0]), Q) + cp.quad_form(u[:,k], R)
        constr += [x[:,k+1] == A@x[:,k] + B@u[:,k],
                   cp.norm_inf(u[:,k]) <= 1.5]
    cost += cp.quad_form(x[:,N] - np.array([r,0.0]), Q)
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True)
    return x.value, u.value, Ts

def mpc_mimo_2x2():
    import cvxpy as cp
    Ts = 0.1
    # 2x2 lightly coupled double integrators
    A = np.block([[np.eye(2), Ts*np.eye(2)],
                  [np.zeros((2,2)), np.eye(2)]])
    B = np.block([[np.zeros((2,2))],
                  [Ts*np.eye(2)]])
    C = np.block([np.eye(2), np.zeros((2,2))])
    nx, nu = A.shape[0], B.shape[1]
    N = 15
    Q = np.diag([10,10,1,1])
    R = 0.1*np.eye(nu)
    x0 = np.zeros((nx,1))
    r = np.array([1.0, -0.5])
    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))
    cost = 0; constr = [x[:,0] == x0.flatten()]
    for k in range(N):
        yk = C @ x[:,k]
        cost += cp.quad_form(yk - r, np.diag([10,10])) + cp.quad_form(u[:,k], R)
        constr += [x[:,k+1] == A@x[:,k] + B@u[:,k],
                   cp.norm_inf(u[:,k]) <= 1.2]
    cost += cp.quad_form(C @ x[:,N] - r, np.diag([10,10]))
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True)
    return x.value, u.value, Ts

if __name__ == "__main__":
    xs, us, Ts = mpc_siso()
    t = np.arange(xs.shape[1])*Ts
    plt.figure()
    plt.plot(t, xs[0,:], label="y")
    plt.step(t[:-1], us[0,:], where="post", label="u")
    plt.legend(); plt.title("Linear MPC (SISO)"); plt.xlabel("Time [s]"); plt.ylabel("Amplitude")
    plt.show()

    xm, um, Ts = mpc_mimo_2x2()
    t = np.arange(xm.shape[1])*Ts
    plt.figure()
    plt.plot(t, xm[0,:], label="y1")
    plt.plot(t, xm[1,:], label="y2")
    plt.step(t[:-1], um[0,:], where="post", label="u1")
    plt.step(t[:-1], um[1,:], where="post", label="u2")
    plt.legend(); plt.title("Linear MPC (2x2 MIMO)"); plt.xlabel("Time [s]")
    plt.show()