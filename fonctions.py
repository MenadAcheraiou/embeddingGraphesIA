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




def load_node2vec_embeddings(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().split()
        n_nodes, n_dim = int(header[0]), int(header[1])
        print(f"Chargement de {n_nodes} nœuds en {n_dim} dimensions...")

        node_ids = []
        vectors = []
        for line in f:
            parts = line.strip().split()
            node_ids.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])

    return np.array(vectors), node_ids



def bonjour():
    print("bonjour")
    




def plot_umap_results(embedding, labels=None):
    """
    Affiche le scatter plot UMAP.
    Argument: 
        - embedding: le résultat de fit_transform
        - labels: (optionnel) le tableau des opinions (0, 4, -1)
    """
    if labels is None:
        raise ValueError("La fonction a besoin des labels pour colorer le graph. "
                         "Passez 'train_opinions_np' en deuxième argument.")

    plt.figure(figsize=(12, 8))
    
    categories = {
        0: {'color': '#e74c3c', 'label': 'Sceptique'},
        4: {'color': '#3498db', 'label': 'Pro-climat'},
        -1: {'color': '#bdc3c7', 'label': 'Inconnu'}
    }
    
    for val, info in categories.items():
        mask = (labels == val)
        if np.any(mask):
            plt.scatter(
                embedding[mask, 0], 
                embedding[mask, 1], 
                c=info['color'], 
                label=f"{info['label']} ({val})",
                s=8, 
                alpha=0.7
            )

    plt.title("Répartition des opinions (UMAP)", fontsize=16)
    plt.legend(title="Opinions", markerscale=2)
    plt.show()


def barycenter_precision(train_embedding, train_opinions_np):

    X = np.array(train_embedding)
    y = np.array(train_opinions_np)

    # Clustering
    clusters = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X)

    # Barycentres
    b0 = X[clusters == 0].mean(axis=0)
    b1 = X[clusters == 1].mean(axis=0)

    # Transformation affine (sans rotation)
    center = (b0 + b1) / 2
    X_centered = X - center

    dist = np.linalg.norm(b0 - b1)
    X_scaled = X_centered * (4 * np.sqrt(2) / dist)

    # Classification par y = -x
    pred_cluster = (X_scaled[:, 1] > -X_scaled[:, 0]).astype(int)

    # Sécurité : vérifier que les deux classes existent
    if len(np.unique(pred_cluster)) < 2:
        print(" Tous les points sont du même côté de y = -x")
        print("Précision forcée à 0.0")
        return 0.0

    # Association cluster a label
    def majority_label(c):
        return Counter(y[pred_cluster == c]).most_common(1)[0][0]

    label_0 = majority_label(0)
    label_1 = majority_label(1)

    y_pred = np.where(pred_cluster == 0, label_0, label_1)

    # Précision
    accuracy = np.mean(y_pred == y)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="coolwarm", s=12)

    x = np.linspace(-6, 6, 200)
    plt.plot(x, -x, 'k--', label="y = -x")

    plt.scatter(2, 2, c='green', s=140, marker='*', label='Barycentre +')
    plt.scatter(-2, -2, c='red', s=140, marker='*', label='Barycentre -')

    # Légende personnalisée pour les labels
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Pro-climat',
               markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Sceptique',
               markerfacecolor='red', markersize=8)
    ]

    plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0])

    plt.title(f"Transformation barycentres → ±(2,2) | précision = {accuracy:.3f}")
    plt.axis("equal")
    plt.show()

    return accuracy