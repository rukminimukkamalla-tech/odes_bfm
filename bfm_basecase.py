import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import itertools

rf, rm, rb = -0.63, -0.86, -0.74        # inertias
af, am, ab = 0.2394, 0.4472, 0.2222     # baselines
r = {'F': rf, 'M': rm, 'B': rb}

def IMB(P): return np.where(P < 0, 0.40*P, 0.25*P)   # mother on baby
def IBM(P): return np.where(P < 0, 0.50*P, 0.20*P)   # baby on mother
def IFB(P): return np.where(P < 0, 0.20*P, 0.40*P)   # father on baby
def IBF(P): return np.where(P < 0, 0.40*P, 0.30*P)   # baby on father
def IMF(P): return np.where(P < 0, 0.28*P, 0.15*P)   # mother on father
def IFM(P): return np.where(P < 0, 0.31*P, 0.21*P)   # father on mother

# slopes
base_slopes = {
    'MF': (0.28, 0.15), 'FM': (0.31, 0.21),
    'MB': (0.40, 0.25), 'FB': (0.20, 0.40),
    'BM': (0.50, 0.20), 'BF': (0.40, 0.30),
}

def odes(x, t):
    F, M, B = x
    dFdt = af + rf*F + IMF(M) + IBF(B)
    dMdt = am + rm*M + IBM(B) + IFM(F)
    dBdt = ab + rb*B + IMB(M) + IFB(F)
    return [dFdt, dMdt, dBdt]

def analyze(a, slopes):
    s = lambda key, sign: slopes[key][0] if sign < 0 else slopes[key][1]
    out = []
    for sF, sM, sB in itertools.product([-1, 1], repeat=3):
        A = np.array([
            [r['F'],      s('MF', sM), s('BF', sB)],
            [s('FM', sF), r['M'],      s('BM', sB)],
            [s('FB', sF), s('MB', sM), r['B']],
        ])
        eq = np.linalg.solve(A, -np.array([a['F'], a['M'], a['B']]))
        signs = [sF, sM, sB]
        if all(np.sign(eq[i]) == signs[i] for i in range(3)):     
            stable = bool(np.all(np.linalg.eigvals(A).real < 0))
            out.append((np.round(eq, 3), stable))
    return out

cases = {
  'both engaged':  ({'F':0.2394,'M':0.4472,'B':0.2222}, base_slopes),
  'father wanted': ({'F':0.2394,'M':0.20,'B':0.2222},
                    {**base_slopes, 'MB':(0.40,0.10), 'BM':(0.70,0.20)}),
  'mother wanted': ({'F':0.05,'M':0.4472,'B':0.2222},
                    {**base_slopes, 'FB':(0.20,0.15), 'BF':(0.65,0.30)}),
  'neither':       ({'F':0.05,'M':0.20,'B':0.2222},
                    {**base_slopes, 'MB':(0.40,0.10), 'BM':(0.70,0.20),
                                    'FB':(0.20,0.15), 'BF':(0.65,0.30)}),
  'mother PPD':    ({'F':0.2394,'M':-0.10,'B':0.2222}, base_slopes),
}

print("=== Equilibria (F, M, B) and stability ===")
for name, (a, sl) in cases.items():
    results = analyze(a, sl)
    if not results:
        print(f"{name:15s}  (no consistent equilibrium found)")
    for eq, stable in results:
        print(f"{name:15s}  eq = {eq}   stable = {stable}")

# ---- baseline time-series ----
t = np.linspace(0, 50, 1000)
sol = odeint(odes, [-1, -1, -1], t)
F, M, B = sol[:, 0], sol[:, 1], sol[:, 2]

plt.plot(t, M, color="#ff1d96", linestyle="--")
plt.plot(t, B, "#bb7cbb")
plt.plot(t, F, "#00c1de", linestyle="-.")
plt.legend(["Mother", "Infant", "Father"])
plt.title("Mother - Infant - Father interaction")
plt.show()