import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

alpha = 0.10
beta = 0.01
gamma = 0.20
delta = 0.02

def odes(x,t):

     prey = x[0]
     predator = x[1]

     dxdt = alpha*prey - beta*prey*predator
     dydt = -gamma*predator + delta*prey*predator

     return[dxdt,dydt]

x0 = [20,10]
t = np.linspace(0,200,10000)

arr = odeint(odes,x0,t)

prey = arr[:,0]
predator = arr[:,1]

plt.plot(t,prey,'b--',t,predator,'r')
plt.xlabel("time")
plt.ylabel("population")
plt.legend(["prey","predator"])
plt.grid('on','both')
plt.show()
