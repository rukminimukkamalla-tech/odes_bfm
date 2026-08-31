import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def IMB(P):
    return np.where(P < 0, 0.40*P, 0.25*P) #mother on baby

def IBM(P):
    return np.where(P < 0, 0.50*P, 0.20*P) #baby on mother

def IFB(P):
    return np.where(P < 0, 0.20*P, 0.40*P) #father on baby

def IBF(P):
    return np.where(P < 0, 0.40*P, 0.30*P) #baby on father

def IMF(P):
    return np.where(P < 0, 0.28*P, 0.15*P) #mother on father

def IFM(P):
    return np.where(P < 0, 0.31*P, 0.21*P) #father on mother

def odes(x,t):

    rf, rm, rb, af, am, ab = -0.63, -0.86, -0.74, 0.2394, 0.4472, 0.2222

    F, M, B = x

    dFdt = af + rf*F + IMF(M) + IBF(B) 
    dMdt = am + rm*M + IBM(B) + IFM(F)
    dBdt = ab + rb*B + IMB(M) + IFB(F)

    return [dFdt, dMdt, dBdt]

x0 = [-1,-1,-1]
t = np.linspace(0,50,1000)
sol = odeint(odes, x0, t)

rf, rm, rb, af, am, ab = -0.63, -0.86, -0.74, 0.2394, 0.4472, 0.2222


F = sol[:,0]
M = sol[:,1]
B = sol[:,2]

x = np.linspace(-10,10,1000)

#nullF = (-af - IMF(x) - IBF(x))/(rf)
#nullM = (-am - IBM(x) - IFM(x))/(-rm)
#nullB = (-ab - IMB(x) - IFB(x))/(-rb)

A = np.array([
    [rf,   0.15, 0.30],   # dF/dt=0 : rf*F + IMF+*M + IBF+*B
    [0.21, rm,   0.20],   # dM/dt=0 : IFM+*F + rm*M + IBM+*B
    [0.40, 0.25, rb  ],   # dB/dt=0 : IFB+*F + IMB+*M + rb*B
])
b = np.array([-af, -am, -ab])
eq = np.linalg.solve(A, b)
print(eq)   #1.34, 1.18, 1.42
