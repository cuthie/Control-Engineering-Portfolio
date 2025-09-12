
import numpy as np
from scipy.signal import cont2discrete

def msd_canonical(m=1.0, c=0.4, k=1.0):
    """
    Mass–spring–damper: x_dot = [0 1; -k/m -c/m] x + [0; 1/m] u, y = [1 0] x
    Returns (A,B,C,D) continuous-time.
    """
    A = np.array([[0.0, 1.0],
                  [-k/m, -c/m]])
    B = np.array([[0.0],
                  [1.0/m]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    return A, B, C, D

def c2d(A,B,C,D, Ts, method="zoh"):
    """Discretize continuous-time state-space with sample time Ts."""
    Ad, Bd, Cd, Dd, _ = cont2discrete((A,B,C,D), Ts, method=method)
    return Ad, Bd, Cd, Dd