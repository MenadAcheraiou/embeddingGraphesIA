import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib.lines import Line2D



def creer_graphe_twitter(fichier_liens):

    print(f"Chargement des liens depuis {fichier_liens}...")
    df_edges = pd.read_csv(fichier_liens)

    G = nx.from_pandas_edgelist(
        df_edges,
        source='user_id',
        target='original_author',
        edge_attr='nb_retweeted',
        create_using=nx.DiGraph()
    )

    print("-" * 30)
    print(f"Nombre de nœuds : {G.number_of_nodes()}")
    print(f"Nombre d'arêtes (retweets) : {G.number_of_edges()}")
    print("-" * 30)

    return G



def plot_umap_projections(projections_dict, figsize=(15, 12)):
    """
    Affiche les projections UMAP pour tous les fichiers (train et test)
    
    Paramètres:
    -----------
    projections_dict : dict
        Dictionnaire contenant les projections UMAP (format généré par la cellule précédente)
    figsize : tuple
        Taille de la figure (largeur, hauteur)
    """
    
    n_files = len(projections_dict)
    
    # Création d'une figure avec 2 colonnes (train/test) et n_files lignes
    fig, axes = plt.subplots(n_files, 2, figsize=(figsize[0], figsize[1] * n_files / 3))
    
    # Si un seul fichier, axes n'est pas un tableau 2D
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (fichier, data) in enumerate(projections_dict.items()):
        # Extraction des données
        train_coords = data['train']
        train_labels = data['y_train']
        test_coords = data['test']
        test_labels = data['y_test']
        
        # --- Plot TRAIN (colonne gauche) ---
        ax_train = axes[idx, 0]
        
        # Séparer les points par label
        train_label_0 = train_coords[np.array(train_labels) == 0]
        train_label_4 = train_coords[np.array(train_labels) == 4]
        
        ax_train.scatter(train_label_0[:, 0], train_label_0[:, 1], 
                        c='blue', label='Pro-climat (0)', alpha=0.6, s=10)
        ax_train.scatter(train_label_4[:, 0], train_label_4[:, 1], 
                        c='red', label='Sceptique (4)', alpha=0.6, s=10)
        
        ax_train.set_title(f'{fichier} - TRAIN ({len(train_labels)} points)')
        ax_train.legend()
        ax_train.grid(True, alpha=0.3)
        
        # --- Plot TEST (colonne droite) ---
        ax_test = axes[idx, 1]
        
        # Séparer les points par label
        test_label_0 = test_coords[np.array(test_labels) == 0]
        test_label_4 = test_coords[np.array(test_labels) == 4]
        
        ax_test.scatter(test_label_0[:, 0], test_label_0[:, 1], 
                       c='blue', label='Pro-climat (0)', alpha=0.6, s=10)
        ax_test.scatter(test_label_4[:, 0], test_label_4[:, 1], 
                       c='red', label='Sceptique (4)', alpha=0.6, s=10)
        
        ax_test.set_title(f'{fichier} - TEST ({len(test_labels)} points)')
        ax_test.legend()
        ax_test.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
