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



def barycenter_precision2(train_embedding, train_opinions_np):

    emb = np.array(train_embedding)
    opinions = np.array(train_opinions_np)

    # Masques basés sur les vrais labels (pas KMeans)
    mask_4 = (opinions == 4)
    mask_0 = (opinions == 0)

    # Barycentres des deux groupes
    b4 = emb[mask_4].mean(axis=0)
    b0 = emb[mask_0].mean(axis=0)

    # Un seul changement de coordonnées pour tout le monde :
    # On veut que b4 -> (2,2) et b0 -> (-2,-2)
    # On cherche : emb_new = A @ emb + t
    #
    # Midpoint des barycentres -> doit aller en (0,0) : translation t = -midpoint
    # Puis on scale/rotate pour que la distance soit bonne
    #
    # Solution simple : translation pour centrer, puis transformation affine

    # 1) Translation : le milieu des deux barycentres va en (0,0)
    midpoint = (b4 + b0) / 2
    emb_centered = emb - midpoint

    # Barycentres après translation
    b4_c = b4 - midpoint  # = (b4 - b0) / 2
    b0_c = b0 - midpoint  # = (b0 - b4) / 2

    # 2) Transformation linéaire : b4_c -> (2,2) et b0_c -> (-2,-2)
    # On note v = b4_c, on veut A @ v = (2,2)
    # b0_c = -v donc A @ b0_c = (-2,-2) automatiquement (cohérent)
    #
    # A est une matrice 2x2, on résout A @ v = [2,2]
    # avec v = b4_c : deux équations, quatre inconnues -> on cherche A symétrique
    # ou plus simple : A = outer([2,2], v) / (v @ v)
    # c'est la projection qui envoie v sur [2,2] mais déforme le reste

    # Meilleure approche : rotation + scaling uniforme
    # On veut que b4_c soit à distance 2*sqrt(2) dans la direction (1,1)/sqrt(2)
    target = np.array([2.0, 2.0])

    # Matrice qui envoie b4_c -> target
    # A = target @ b4_c^T / (b4_c @ b4_c)  (rang 1, trop restrictif)
    # -> On utilise une vraie transformation : rotation + scale uniforme
    # scale = |target| / |b4_c|
    # rotation : angle entre b4_c et target

    norm_b4c = np.linalg.norm(b4_c)
    norm_target = np.linalg.norm(target)

    scale = norm_target / norm_b4c

    # Angle de rotation
    angle_b4c = np.arctan2(b4_c[1], b4_c[0])
    angle_target = np.arctan2(target[1], target[0])
    theta = angle_target - angle_b4c

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])

    A = scale * R  # transformation unique pour tout le monde

    emb_transformed = (A @ emb_centered.T).T

    # Score par rapport à la droite y = -x
    scores = emb_transformed[:, 0] + emb_transformed[:, 1]

    pred_a = np.where(scores > 0, 4, 0)
    pred_b = np.where(scores > 0, 0, 4)

    acc_a = np.mean(pred_a == opinions)
    acc_b = np.mean(pred_b == opinions)

    if acc_a >= acc_b:
        predicted_opinions = pred_a
        accuracy = acc_a
    else:
        predicted_opinions = pred_b
        accuracy = acc_b

    # Plot
    plt.figure(figsize=(6, 6))

    plt.scatter(
        emb_transformed[:, 0],
        emb_transformed[:, 1],
        c=predicted_opinions,
        cmap="coolwarm",
        s=12,
        alpha=0.7
    )

    x = np.linspace(-6, 6, 200)
    plt.plot(x, -x, 'k--', label="y = -x")

    # Barycentres après transformation
    b4_t = emb_transformed[mask_4].mean(axis=0)
    b0_t = emb_transformed[mask_0].mean(axis=0)

    plt.scatter(b4_t[0], b4_t[1], c='red',  s=140, marker='*', label='Barycentre pro-climat (4)')
    plt.scatter(b0_t[0], b0_t[1], c='blue', s=140, marker='*', label='Barycentre sceptique (0)')
    
    legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Pro-climat (4)',
           markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Sceptique (0)',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], linestyle='--', color='black', label='y = -x')
]
    plt.legend(handles=legend_elements)
    plt.title(f"Transformation unique | précision = {accuracy:.3f}")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.show()

    return accuracy




def compute_barycenter_transform(train_embedding, train_opinions_np):
    """Calcule et retourne les paramètres de transformation depuis le train."""
    emb = np.array(train_embedding)
    opinions = np.array(train_opinions_np)

    # On exclut les inconnus
    mask_known = opinions != -1
    emb_known = emb[mask_known]
    opinions_known = opinions[mask_known]

    mask_4 = (opinions_known == 4)
    mask_0 = (opinions_known == 0)

    b4 = emb_known[mask_4].mean(axis=0)
    b0 = emb_known[mask_0].mean(axis=0)

    midpoint = (b4 + b0) / 2
    b4_c = b4 - midpoint

    target = np.array([2.0, 2.0])
    scale = np.linalg.norm(target) / np.linalg.norm(b4_c)

    angle_b4c    = np.arctan2(b4_c[1], b4_c[0])
    angle_target = np.arctan2(target[1], target[0])
    theta = angle_target - angle_b4c

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    A = scale * R

    # On retourne les paramètres, pas les données transformées
    return {"midpoint": midpoint, "A": A}

def apply_barycenter_transform(embedding, transform_params, opinions, plot=True, title=""):
    emb = np.array(embedding)
    opinions = np.array(opinions)

    emb_transformed = (transform_params["A"] @ (emb - transform_params["midpoint"]).T).T

    scores = emb_transformed[:, 0] + emb_transformed[:, 1]
    pred_a = np.where(scores > 0, 4, 0)
    pred_b = np.where(scores > 0, 0, 4)

    mask_known = opinions != -1
    acc_a = np.mean(pred_a[mask_known] == opinions[mask_known])
    acc_b = np.mean(pred_b[mask_known] == opinions[mask_known])

    if acc_a >= acc_b:
        predicted_opinions = pred_a
        accuracy = acc_a
    else:
        predicted_opinions = pred_b
        accuracy = acc_b

    if plot:
        # Calcul des barycentres dans l'espace transformé
        mask_4 = (opinions == 4)
        mask_0 = (opinions == 0)
        b4_t = emb_transformed[mask_4].mean(axis=0)
        b0_t = emb_transformed[mask_0].mean(axis=0)

        plt.figure(figsize=(6, 6))
        plt.scatter(
            emb_transformed[:, 0], emb_transformed[:, 1],
            c=predicted_opinions, cmap="coolwarm", s=12, alpha=0.7
        )
        x = np.linspace(-6, 6, 200)
        plt.plot(x, -x, 'k--')

        plt.scatter(b4_t[0], b4_t[1], c='red',  s=140, marker='*', zorder=5)
        plt.scatter(b0_t[0], b0_t[1], c='blue', s=140, marker='*', zorder=5)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Pro-climat (4)',
                   markerfacecolor='red', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Sceptique (0)',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], linestyle='--', color='black', label='y = -x')
        ]
        plt.legend(handles=legend_elements)
        plt.title(f"{title} | précision = {accuracy:.3f}")
        plt.axis("equal")
        plt.grid(alpha=0.2)
        plt.show()

    return accuracy