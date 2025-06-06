import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Charger les données
data = np.loadtxt('C:/Users/julie/Documents/CS/Cours/Black Swans/EI/Higgs_collaboration_A/sample_code_submission/BDT/data_simulations.txt', delimiter=',')
# Colonnes : 0=n_estimators, 1=max_depth, 2=eta, 3=subsample, ..., -1=significance
n_estimators = data[:, 0]
max_depth = data[:, 1]
eta = data[:, 2]
subsample = data[:, 3]
significance = data[:, -1]


def courbes_nest_et_maxdepth ():
    # Trouver toutes les combinaisons uniques de (eta, subsample)
    unique_eta_subsample = np.unique(np.column_stack((eta, subsample)), axis=0)

    # Tracer une figure pour chaque combinaison unique de (eta, subsample)
    for eta_val, subsample_val in unique_eta_subsample:
        # Créer une nouvelle figure pour chaque combinaison
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Subgraphe 3D

        # Filtrer les données pour la combinaison (eta, subsample)
        mask = (eta == eta_val) & (subsample == subsample_val)
        x = n_estimators[mask]
        y = max_depth[mask]
        z = significance[mask]

        # Pour avoir des courbes propres, trier selon n_estimators puis max_depth
        sort_idx = np.lexsort((y, x))
        x, y, z = x[sort_idx], y[sort_idx], z[sort_idx]

        # Tracer la surface 3D ou les points
        ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

        # Ajouter des titres et labels
        ax.set_title(f'eta={eta_val}, subsample={subsample_val}')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('max_depth')
        ax.set_zlabel('significance')

        # Afficher chaque figure séparément
        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def courbes_eta_et_subsample():
    # Trouver toutes les combinaisons uniques de (n_estimators, max_depth)
    unique_nest_maxdepth = np.unique(np.column_stack((n_estimators, max_depth)), axis=0)

    # Tracer une figure pour chaque combinaison unique de (n_estimators, max_depth)
    for nest_val, depth_val in unique_nest_maxdepth:
        # Créer une nouvelle figure pour chaque combinaison (n_estimators, max_depth)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Subgraphe 3D

        # Filtrer les données pour la combinaison (n_estimators, max_depth)
        mask = (n_estimators == nest_val) & (max_depth == depth_val)
        x = eta[mask]  # Valeurs de eta
        y = subsample[mask]  # Valeurs de subsample
        z = significance[mask]  # Valeurs de significance

        # Créer la surface 3D
        ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

        # Ajouter des titres et labels
        ax.set_title(f'Significance vs eta and subsample\n(n_estimators={nest_val}, max_depth={depth_val})')
        ax.set_xlabel('eta')
        ax.set_ylabel('subsample')
        ax.set_zlabel('significance')

        # Afficher chaque figure séparément
        plt.tight_layout()
        plt.show()
    


def courbe_significance():
    """trace la moyenne de la significance en fonction de n_estimators. (la moyenne sur les valeurs 
    de paramètres eta, max_depth et subsample)  """
    # Charger les données
    data = np.loadtxt('C:/Users/julie/Documents/CS/Cours/Black Swans/EI/Higgs_collaboration_A/sample_code_submission/BDT/data_simulations.txt', delimiter=',')
    
    # Extraire les colonnes : 
    # 0 = n_estimators, 1 = max_depth, 2 = eta, 3 = subsample, ..., -1 = significance
    n_estimators = data[:, 0] 
    print(n_estimators)
    max_depth = data[:, 1]
    eta = data[:, 2]
    subsample = data[:, 3]
    significance = data[:, -1]
    
    # Créer une liste unique de valeurs pour n_estimators
    unique_n_estimators = np.unique(n_estimators)
    unique_n_estimators = unique_n_estimators[:-1] #on eclu la dernière valeur de n_estimators (pas assez de simulations faites pour obtenir une moyenne significative)
    # Initialiser un tableau pour stocker les moyennes de significance
    mean_significance = []

    # Calculer la moyenne de la significance pour chaque n_estimators
    for n in unique_n_estimators:
        # Filtrer les données pour cette valeur de n_estimators
        mask = (n_estimators == n)
        # Calculer la moyenne de la significance pour cette valeur de n_estimators
        mean_significance.append(np.mean(significance[mask]))
    
    # Convertir la liste en array pour une utilisation facile avec matplotlib
    mean_significance = np.array(mean_significance)

    # Tracer la moyenne de la significance en fonction de n_estimators
    plt.figure(figsize=(10, 6))
    plt.plot(unique_n_estimators, mean_significance, marker='o', linestyle='-', color='b', label='Mean Significance')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean Significance')
    plt.title('Mean Significance vs n_estimators')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Appel de la fonction
courbe_significance()
#courbes_nest_et_maxdepth()
#courbes_eta_et_subsample()
