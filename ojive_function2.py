import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def I(P):
    return np.piecewise(P,[P < -1.5, (P >= -1.5) & (P < 0), (P >= 0) & (P < 1.5), P >= 1.5],[-2,-1,0.8,1])


def odes(x,t):

    H, W = x

    dHdt = 0.2394 - 0.63*H + I(W)
    dWdt = 0.4472 - 0.86*W + I(H)

    return [dHdt, dWdt]

x0 = [-1,-1]
t = np.linspace(0,50,1000)
sol = odeint(odes, x0, t)

H = sol[:,0]
W = sol[:,1]

plt.plot(t,H,color ="#9e68ad", linestyle = "--")
plt.plot(t,W,"#ffc1ff")
plt.legend(["Husband","Wife"])
plt.title("Figure 2 Graph")
plt.show()