import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,500)
y = np.where(x < 0, 0.31*x, 0.21*x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="#1d1ddf")          # the light purple line

ax.spines["left"].set_position("zero")   # y-axis through x=0
ax.spines["bottom"].set_position("zero") # x-axis through y=0
ax.spines["right"].set_visible(False)    # hide the box edges
ax.spines["top"].set_visible(False)

plt.show()