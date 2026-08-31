import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import itertools

# ---- parameters ----
rf, rm, rb = -0.63, -0.86, -0.74
af, am, ab = 0.2394, 0.4472, 0.2222
r = {'F': rf, 'M': rm, 'B': rb}

# ---- bilinear influence functions (m-, m+) ----
def IMB(P): return np.where(P < 0, 0.40*P, 0.25*P)   # mother -> baby
def IBM(P): return np.where(P < 0, 0.50*P, 0.20*P)   # baby   -> mother
def IFB(P): return np.where(P < 0, 0.20*P, 0.40*P)   # father -> baby
def IBF(P): return np.where(P < 0, 0.40*P, 0.30*P)   # baby   -> father
def IMF(P): return np.where(P < 0, 0.28*P, 0.15*P)   # mother -> father
def IFM(P): return np.where(P < 0, 0.31*P, 0.21*P)   # father -> mother

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

# ---- equilibria + stability for one case ----
def analyze(a, slopes, rng=5.0):
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
        if all(np.sign(eq[i]) == signs[i] for i in range(3)) \
           and np.max(np.abs(eq)) < rng:                 # keep meaningful ones
            stable = bool(np.all(np.linalg.eigvals(A).real < 0))
            out.append((np.round(eq, 3), stable))
    return out

# ---- cases: PPD = lowered baseline (slopes unchanged) ----
cases = {
  'both healthy':      ({'F':0.2394,'M':0.4472,'B':0.2222}, base_slopes),
  'mother PPD mild':   ({'F':0.2394,'M':-0.10, 'B':0.2222}, base_slopes),
  'mother PPD severe': ({'F':0.2394,'M':-0.70, 'B':0.2222}, base_slopes),
  'father PPD':        ({'F':-0.30, 'M':0.4472,'B':0.2222}, base_slopes),
  'both PPD':          ({'F':-0.30, 'M':-0.70, 'B':0.2222}, base_slopes),
}

print("=== Equilibria (F, M, B) and stability ===")
for name, (a, sl) in cases.items():
    results = analyze(a, sl)
    if not results:
        print(f"{name:18s} (no meaningful equilibrium)")
    for eq, stable in results:
        print(f"{name:18s} eq = {eq}   stable = {stable}")



# ---- time-dependent influence: boosted early, smoothly eases (sigmoid) ----
t_switch = 3.0   # months: center of the transition (kept for the axvline)

def factor(t, boost=0.3, a=2.0, t0=t_switch):   # was boost=0.6
    sig = 1.0 / (1.0 + np.exp(a*(t - t0)))   # ~1 before 3 mo, ~0 after
    return 1.0 + boost*sig                    # ~1+boost early, -> 1 later

def odes_td(x, t, a):
    F, M, B = x
    g = factor(t)                              # smooth time-decaying multiplier
    IMB_t = g*np.where(M < 0, 0.40*M, 0.25*M)  # mother -> baby
    IFB_t = g*np.where(F < 0, 0.20*F, 0.40*F)  # father -> baby
    IBM_t = g*np.where(B < 0, 0.50*B, 0.20*B)  # baby   -> mother
    IBF_t = g*np.where(B < 0, 0.40*B, 0.30*B)  # baby   -> father
    dFdt = a['F'] + rf*F + IMF(M) + IBF_t
    dMdt = a['M'] + rm*M + IBM_t + IFM(F)
    dBdt = a['B'] + rb*B + IMB_t + IFB_t
    return [dFdt, dMdt, dBdt]

t = np.linspace(0, 12, 2000)   # months, first year
a_healthy = {'F':0.2394,'M':0.4472,'B':0.2222}
a_severe  = {'F':0.2394,'M':-0.70, 'B':0.2222}

fig2, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, a, title in [(axes[0], a_healthy, 'Healthy'),
                     (axes[1], a_severe,  'Severe maternal PPD')]:
    sol = odeint(odes_td, [-1,-1,-1], t, args=(a,))
    ax.plot(t, sol[:,0], "#00c1de", linestyle="-.", label="Father")
    ax.plot(t, sol[:,1], "#ff1d96", linestyle="--", label="Mother")
    ax.plot(t, sol[:,2], "#bb7cbb", label="Infant")
    ax.axvline(t_switch, color="gray", lw=0.8, linestyle=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(title); ax.set_xlabel("time scale"); ax.legend(fontsize=8)
axes[0].set_ylabel("emotional state")
fig2.tight_layout()
plt.show()