import numpy as np
import matplotlib.pyplot as plt

params = ['aF','aM','aB','mMB+','mFB+','mBM-','mBF-']
S1 = [0.083, 0.554, 0.043, 0.076, 0.241, 0.000, 0.000]
ST = [0.083, 0.557, 0.043, 0.076, 0.245, 0.000, 0.000]

x = np.arange(len(params)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, S1, w, label='S1 (first-order)', color="#00c1de")
ax.bar(x + w/2, ST, w, label='ST (total)',       color="#ff1d96")
ax.set_xticks(x); ax.set_xticklabels(params)
ax.set_ylabel('Sobol index'); ax.set_title('Global sensitivity of the family equilibrium')
ax.legend(); plt.tight_layout(); plt.show()