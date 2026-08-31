import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# ----------------------------------------------------------------------
# Bilinear influence function with negative/positive slopes
# ----------------------------------------------------------------------
def bilinear(x, m_neg, m_pos):
    return np.where(x < 0, m_neg * x, m_pos * x)
 
# ----------------------------------------------------------------------
# Parameter sets
# ----------------------------------------------------------------------
# Inertias (negative; |r| small => high emotional inertia / lability).
# r_H, r_W from Gottman Table 10.1 (validating couple) as used in the source papers.
# r_B chosen smaller in magnitude: infants self-regulate slowly.
def make_params(scenario="supported"):
    p = dict(
        a_H=0.2394, r_H=-0.63,   # father  uninfluenced eq H_s = 0.380
        a_W=0.4472, r_W=-0.86,   # mother  uninfluenced eq W_s = 0.520
        a_B=0.1500, r_B=-0.50,   # baby    uninfluenced eq B_s = 0.300
        # couple <-> couple slopes (negative larger than positive: validating couple)
        m_WH_neg=0.28, m_WH_pos=0.15,   # wife -> husband
        m_HW_neg=0.31, m_HW_pos=0.21,   # husband -> wife
        # baby -> parent slopes (negative >> positive: infant distress demands more
        # than infant contentment soothes; bidirectional NA->anxiety literature)
        m_BH_neg=0.45, m_BH_pos=0.15,   # baby -> father
        m_BW_neg=0.55, m_BW_pos=0.18,   # baby -> mother (primary-caregiver load)
        # parent -> baby slopes (coregulation / synchrony / emotion socialization)
        m_HB_neg=0.30, m_HB_pos=0.22,   # father -> baby
        m_WB_neg=0.40, m_WB_pos=0.28,   # mother -> baby
        sigma=np.array([0.15, 0.15, 0.20]),  # white-noise amplitudes
    )
    if scenario == "overload":
        # sleep deprivation lowers the parents' uninfluenced set-points, and the
        # infant's negative pull on the mother strengthens.
        p["a_H"] = -0.10
        p["a_W"] = -0.05
        p["m_BW_neg"] = 0.85
        p["m_BH_neg"] = 0.70
    return p
 
# ----------------------------------------------------------------------
# Vector field
# ----------------------------------------------------------------------
def field(state, p):
    H, W, B = state
    dH = p["a_H"] + p["r_H"]*H + bilinear(W, p["m_WH_neg"], p["m_WH_pos"]) \
                               + bilinear(B, p["m_BH_neg"], p["m_BH_pos"])
    dW = p["a_W"] + p["r_W"]*W + bilinear(H, p["m_HW_neg"], p["m_HW_pos"]) \
                               + bilinear(B, p["m_BW_neg"], p["m_BW_pos"])
    dB = p["a_B"] + p["r_B"]*B + bilinear(H, p["m_HB_neg"], p["m_HB_pos"]) \
                               + bilinear(W, p["m_WB_neg"], p["m_WB_pos"])
    return np.array([dH, dW, dB])
 
# ----------------------------------------------------------------------
# Integrators
# ----------------------------------------------------------------------
def rk4(state0, p, T=80.0, dt=0.01):
    n = int(T/dt); xs = np.empty((n+1, 3)); ts = np.linspace(0, T, n+1)
    xs[0] = state0
    for k in range(n):
        x = xs[k]
        k1 = field(x, p); k2 = field(x+0.5*dt*k1, p)
        k3 = field(x+0.5*dt*k2, p); k4 = field(x+dt*k3, p)
        xs[k+1] = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
    return ts, xs
 
def euler_maruyama(state0, p, T=80.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n = int(T/dt); xs = np.empty((n+1, 3)); ts = np.linspace(0, T, n+1)
    xs[0] = state0; sq = np.sqrt(dt)
    for k in range(n):
        x = xs[k]
        xs[k+1] = x + dt*field(x, p) + p["sigma"]*sq*rng.standard_normal(3)
    return ts, xs
 
# ----------------------------------------------------------------------
# Equilibrium in the all-positive octant + stability
# ----------------------------------------------------------------------
def positive_equilibrium(p):
    # In the octant H,W,B > 0 all influence functions use their positive slope,
    # so the system is linear:  A x = -a
    A = np.array([
        [p["r_H"],       p["m_WH_pos"],  p["m_BH_pos"]],
        [p["m_HW_pos"],  p["r_W"],       p["m_BW_pos"]],
        [p["m_HB_pos"],  p["m_WB_pos"],  p["r_B"]],
    ])
    a = np.array([p["a_H"], p["a_W"], p["a_B"]])
    x = np.linalg.solve(A, -a)
    eig = np.linalg.eigvals(A)        # Jacobian == A for bilinear (within octant)
    return x, A, eig
 
def routh_hurwitz_3(A):
    # char poly: l^3 + c2 l^2 + c1 l + c0
    c2 = -np.trace(A)
    # sum of principal 2x2 minors
    c1 = (A[0,0]*A[1,1]-A[0,1]*A[1,0]
          + A[0,0]*A[2,2]-A[0,2]*A[2,0]
          + A[1,1]*A[2,2]-A[1,2]*A[2,1])
    c0 = -np.linalg.det(A)
    stable = (c2 > 0) and (c0 > 0) and (c2*c1 > c0)
    return (c2, c1, c0), stable
 
# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    for name in ["supported", "overload"]:
        p = make_params(name)
        x, A, eig = positive_equilibrium(p)
        coeffs, stable = routh_hurwitz_3(A)
        print(f"\n=== scenario: {name} ===")
        print("positive-octant equilibrium (H*,W*,B*):", np.round(x, 4),
              "valid(all>0):", bool(np.all(x > 0)))
        print("Jacobian eigenvalues:", np.round(eig, 4))
        print("Routh-Hurwitz (c2,c1,c0):", np.round(coeffs, 4), "-> stable:", stable)
 
    # Figure: supported vs overload, deterministic + one stochastic realization
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    labels = ["H father", "W mother", "B baby"]
    colors = ["#1f5fa8", "#c0392b", "#27ae60"]
    for ax, name in zip(axes, ["supported", "overload"]):
        p = make_params(name)
        ts, xd = rk4(np.array([0.1, 0.1, 0.1]), p)
        _, xs = euler_maruyama(np.array([0.1, 0.1, 0.1]), p, seed=3)
        for j in range(3):
            ax.plot(ts, xd[:, j], color=colors[j], lw=2.0, label=labels[j])
            ax.plot(ts, xs[:, j], color=colors[j], lw=0.8, alpha=0.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("time t"); ax.set_title(name)
        ax.set_ylabel("emotional state")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Family emotional dynamics: deterministic (bold) and a stochastic realization (faint)")
    fig.tight_layout()
    fig.savefig("family_dynamics.png", dpi=150)
    print("\nsaved family_dynamics.png")