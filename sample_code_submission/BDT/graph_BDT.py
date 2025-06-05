import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('data_simulations.txt', sep='\t')

# S'assurer que les colonnes sont bien nommées
# Colonnes attendues : 'n_estimators', 'max_depth', 'eta', 'subsample', 'significance'

# Courbes 3D pour différentes valeurs de eta
etas = df['eta'].unique()
fig_eta = plt.figure(figsize=(10, 7))
ax_eta = fig_eta.add_subplot(111, projection='3d')
for eta in etas:
    subset = df[df['eta'] == eta]
    ax_eta.plot(
        subset['n_estimators'],
        subset['max_depth'],
        subset['significance'],
        label=f'eta={eta}'
    )
ax_eta.set_xlabel('n_estimators')
ax_eta.set_ylabel('max_depth')
ax_eta.set_zlabel('significance')
ax_eta.set_title('Significance en fonction de n_estimators et max_depth (par eta)')
ax_eta.legend()

# Courbes 3D pour différentes valeurs de subsample
subsamples = df['subsample'].unique()
fig_sub = plt.figure(figsize=(10, 7))
ax_sub = fig_sub.add_subplot(111, projection='3d')
for subsample in subsamples:
    subset = df[df['subsample'] == subsample]
    ax_sub.plot(
        subset['n_estimators'],
        subset['max_depth'],
        subset['significance'],
        label=f'subsample={subsample}'
    )
ax_sub.set_xlabel('n_estimators')
ax_sub.set_ylabel('max_depth')
ax_sub.set_zlabel('significance')
ax_sub.set_title('Significance en fonction de n_estimators et max_depth (par subsample)')
ax_sub.legend()

plt.show()