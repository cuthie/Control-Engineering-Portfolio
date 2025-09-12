import numpy as np
import matplotlib.pyplot as plt

# Nonlinear pendulum: theta_ddot = -(g/L) sin(theta) - d*theta_dot + (1/(mL^2)) u
# States: x = [theta, theta_dot]; control u = torque

def f(x,u,params):
    g,L,m,d = params["g"], params["L"], params["m"], params["d"]
    theta, omega = x
    theta_dot = omega
    omega_dot = -(g/L)*np.sin(theta) - d*omega + (1.0/(m*L*L))*u
    return np.array([theta_dot, omega_dot])

def rk4_step(x,u,dt,params):
    k1 = f(x,u,params)
    k2 = f(x+0.5*dt*k1,u,params)
    k3 = f(x+0.5*dt*k2,u,params)
    k4 = f(x+dt*k3,u,params)
    return x + (dt/6.0)*(k1+2*k2+2*k3+k4)

def jacobian_f(x,u,params):
    g,L,m,d = params["g"], params["L"], params["m"], params["d"]
    theta, omega = x
    A = np.array([[0.0, 1.0],
                  [-(g/L)*np.cos(theta), -d]])
    B = np.array([[0.0],
                  [1.0/(m*L*L)]])
    return A, B

def ekf_predict(xhat, Phat, u, Q, dt, params):
    # Discretize via RK4 step for state; linearize for covariance
    A, B = jacobian_f(xhat, u, params)
    xhat_pred = rk4_step(xhat, u, dt, params)
    Ad = np.eye(2) + A*dt  # simple Euler discretization for covariance
    Bd = B*dt
    Phat_pred = Ad @ Phat @ Ad.T + Q
    return xhat_pred, Phat_pred

def ekf_update(xhat_pred, Phat_pred, y, R, C):
    S = C @ Phat_pred @ C.T + R
    K = Phat_pred @ C.T @ np.linalg.inv(S)
    xhat_upd = xhat_pred + K @ (y - C @ xhat_pred)
    Phat_upd = (np.eye(len(xhat_pred)) - K @ C) @ Phat_pred
    return xhat_upd, Phat_upd

def sl_mpc_step(xlin, N, dt, params, r, umax=3.0):
    """Successive linearization MPC around xlin. Solves a QP using cvxpy."""
    import cvxpy as cp
    A, B = jacobian_f(xlin, 0.0, params)
    # Discretize (Euler); small dt assumed
    Ad = np.eye(2) + A*dt
    Bd = B*dt
    nx, nu = 2, 1
    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))
    Q = np.diag([25.0, 2.0])
    R = np.array([[0.1]])
    constr = [x[:,0] == xlin]
    cost = 0
    for k in range(N):
        yk = x[0,k]  # theta
        cost += cp.quad_form(cp.hstack([yk - r, x[1,k]]), Q) + cp.quad_form(u[:,k], R)
        constr += [x[:,k+1] == Ad @ x[:,k] + Bd @ u[:,k],
                   cp.norm_inf(u[:,k]) <= umax]
    cost += cp.quad_form(cp.hstack([x[0,N]-r, x[1,N]]), Q)
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True)
    if u.value is None:
        return 0.0  # fallback
    return float(u.value[0,0])

if __name__ == "__main__":
    params = dict(g=9.81, L=0.7, m=1.0, d=0.15)
    dt = 0.02
    T = 6.0
    steps = int(T/dt)
    # True state and measurements
    x = np.array([0.6, 0.0])  # initial angle rad
    C = np.array([[1.0, 0.0]])  # measure theta
    R = np.array([[0.01]])
    Q = np.diag([1e-4, 1e-4])
    # EKF init
    xhat = np.array([0.0, 0.0])
    Phat = np.eye(2)*0.1
    # MPC settings
    N = 15
    r = 0.0  # regulate to upright (theta=0)
    umax = 3.0
    th_hist=[]; thhat_hist=[]; u_hist=[]; t_hist=[]
    for k in range(steps):
        # Measurement
        y = C @ x + np.random.normal(0, np.sqrt(R[0,0]), size=(1,))
        # EKF predict/update
        xhat_pred, Phat_pred = ekf_predict(xhat, Phat, 0.0, Q, dt, params)
        xhat, Phat = ekf_update(xhat_pred, Phat_pred, y, R, C)
        # SL-MPC around current estimate
        u = sl_mpc_step(xhat.copy(), N, dt, params, r, umax=umax)
        # Plant integrate
        x = rk4_step(x, u, dt, params)
        # Logs
        th_hist.append(x[0]); thhat_hist.append(xhat[0]); u_hist.append(u); t_hist.append(k*dt)
    t = np.array(t_hist)
    plt.figure()
    plt.plot(t, th_hist, label="theta (true)")
    plt.plot(t, thhat_hist, label="theta_hat (EKF)")
    plt.step(t, u_hist, where="post", label="u")
    plt.legend(); plt.title("Nonlinear Pendulum: EKF + Successive-Linearization MPC")
    plt.xlabel("Time [s]"); plt.ylabel("Angle [rad] / Torque")
    plt.show()