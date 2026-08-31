import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def I(P): 
    return np.piecewise(P, [P < -1.5, (-1.5 <= P) & (P < -0.5), (-0.5 <= P) & (P < 0), (0 <= P) & (P < 0.5), (0.5 <= P) & (P < 1.5), P > 1.5],[0, lambda P: (1.5 * -1.5)/(-0.5 + 1.5) + (-1.5/(-0.5+1.5))*P, lambda P: (-1.5/(-0.5))*P, lambda P: (0.7/0.5)*P, lambda P: (-1.5 * 0.7)/(0.5-1.5) + (0.7/(0.5-1.5))*P, 0])

def odes(x, t):
    
    H, W = x

    dHdt = 0.2394 - 0.63*H + I(W)
    dWdt = 0.4472 - 0.86*W + I(H)

    return [dHdt, dWdt]


x0 = [0,0]
t = np.linspace(0,100,5000)

sol = odeint(odes, x0, t)

H = sol[:,0]
W = sol[:,1]

x = np.linspace(-4,4,1000)

nullH = (-0.2394 - I(x))/(-0.63)
nullW = (-0.4472 - I(x))/(-0.86)

plt.plot(nullH,x,color ="#9e68ad", linestyle = '--')
plt.plot(x,nullW,color="#ffc1ff")
plt.xlabel("H(t)")
plt.ylabel("W(t)")
plt.legend(["H nullcline","W nullcline"])
plt.title("Figure 7 nullclines")
plt.show()