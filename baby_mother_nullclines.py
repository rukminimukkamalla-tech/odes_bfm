import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

def IMB(P):
    return np.where(P < 0, 0.40*P, 0.25*P)

def IBM(P):
    return np.where(P < 0, 0.50*P, 0.20*P)

def odes(x,t):

    M, B = x

    dMdt = 0.4472 - 0.86*M + IBM(B)
    dBdt = 0.2222 - 0.74*B + IMB(M)

    return [dMdt, dBdt]

x0 = [-1,-1]
t = np.linspace(0,50,1000)
sol = odeint(odes, x0, t)

M = sol[:,0]
B = sol[:,1]

x = np.linspace(-10,10,1000)

nullM = (-0.4472 - IBM(x))/(-0.86)
nullB = (-0.2222 - IMB(x))/(-0.74)


plt.plot(nullM,x,color ="#9e68ad", linestyle = '--')
plt.plot(x,nullB,color="#ffc1ff")
plt.xlabel("M(t)")
plt.ylabel("B(t)")
plt.legend(["M nullcline","B nullcline"])
plt.title("Figure 7 nullclines")
plt.show()