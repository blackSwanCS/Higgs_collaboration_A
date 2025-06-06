import matplotlib.pyplot as plt
import numpy as np

# Données de la première courbe
x1 = np.linspace(0, 50, 11)
y1 = [2.06, 5.55, 5.56, 5.74, 5.73, 5.76, 5.33, 5.38, 5.25, 5.21, 5.09]

# Données de la deuxième courbe
x2 = np.linspace(0, 50, 11)
y2 = [0.5, 0.873, 0.873, 0.877, 0.873, 0.875, 0.870, 0.869, 0.868, 0.867, 0.864]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Premier axe Y
color1 = 'royalblue'
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Significance', color=color1, fontsize=12)
ax1.plot(x1, y1, color=color1, marker='o', linestyle='None', linewidth=2, label='Y1', markersize=6)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.7)

# Deuxième axe Y (côté droit)
ax2 = ax1.twinx()
color2 = 'darkorange'
ax2.set_ylabel('AUC', color=color2, fontsize=12)
ax2.plot(x2, y2, color=color2, marker='s', linestyle='None', linewidth=2, label='Y2', markersize=3)
ax2.tick_params(axis='y', labelcolor=color2)

# Titre et ajustements
plt.title("Significance and AUC in function of Epochs", fontsize=16)
fig.tight_layout()
plt.show()