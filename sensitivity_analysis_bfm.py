import numpy as np
from scipy.integrate import odeint
from SALib.sample import saltelli
from SALib.analyze import sobol

def model_output(p):
    aF, aM, aB, mMBp, mFBp, mBMn, mBFn = p
    def f(x, t):
        F, M, B = x
        IMF = np.where(M<0, 0.28*M, 0.15*M)
        IFM = np.where(F<0, 0.31*F, 0.21*F)
        IMB = np.where(M<0, 0.40*M, mMBp*M)   # mother->baby (+ slope varied)
        IFB = np.where(F<0, 0.20*F, mFBp*F)   # father->baby (+ slope varied)
        IBM = np.where(B<0, mBMn*B, 0.20*B)   # baby->mother (- slope varied)
        IBF = np.where(B<0, mBFn*B, 0.30*B)   # baby->father (- slope varied)
        return [aF - 0.63*F + IMF + IBF,
                aM - 0.86*M + IBM + IFM,
                aB - 0.74*B + IMB + IFB]
    final = odeint(f, [-1,-1,-1], np.linspace(0,100,400))[-1]
    return min(final)        # worst-off member

problem = {
    'num_vars': 7,
    'names': ['aF','aM','aB','mMB+','mFB+','mBM-','mBF-'],
    'bounds': [[0.19,0.29],[0.36,0.54],[0.18,0.27],   # baselines ±20%
               [0.20,0.30],[0.32,0.48],               # parent->baby +slopes ±20%
               [0.40,0.60],[0.32,0.48]],              # baby->parent -slopes ±20%
}

X = saltelli.sample(problem, 512)
Y = np.array([model_output(x) for x in X])
Si = sobol.analyze(problem, Y)

for name, s1, st in zip(problem['names'], Si['S1'], Si['ST']):
    print(f"{name:6s}  S1={s1: .3f}  ST={st: .3f}")