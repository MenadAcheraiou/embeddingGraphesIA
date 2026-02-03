import pandas as pd
import numpy as np
import networkx as nx
import umap
import random
from sklearn.model_selection import train_test_split

# ===============================
# 1. Chargement des données
# ===============================

df_nodes = pd.read_csv("Nodes.csv")
df_edges = pd.read_csv("graphe_complet_final.csv")

# Graphe
G = nx.from_pandas_edgelist(
    df_edges,
    source="user_id",
    target="original_author"
)


opinion_dict = (
    df_nodes
    .set_index("Id")["modularity_class"]
    .map({4: 1, 0: 0})
    .dropna()
    .to_dict()
)

# ===============================
# 2. Sélection d’un sous-graphe
# ===============================


N_SUB = 4000

# On force la présence de nœuds labellisés
nodes_labeled = list(opinion_dict.keys())
nodes_unlabeled = list(set(G.nodes()) - set(nodes_labeled))

n_labeled_sub = min(2000, len(nodes_labeled))
n_unlabeled_sub = N_SUB - n_labeled_sub

nodes_subset = (
    random.sample(nodes_labeled, n_labeled_sub)
    + random.sample(nodes_unlabeled, n_unlabeled_sub)
)

G_sub = G.subgraph(nodes_subset)

# ===============================
# 3. Matrice d’adjacence 
# ===============================

adj = nx.adjacency_matrix(G_sub)
adj_dense = adj.toarray()

# ===============================
# 4. Labels ALIGNÉS avec le sous-graphe
# ===============================

labels_sub = [
    opinion_dict.get(node, -1)
    for node in G_sub.nodes()
]

# ===============================
# 5. UMAP supervisé
# ===============================

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=10,
 
)

embedding = reducer.fit_transform(adj_dense, y=labels_sub)

# ===============================
# 6. Résultat
# ===============================

print("Embedding shape :", embedding.shape)
print("Nb labels connus :", sum(l != -1 for l in labels_sub))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. Sélection des nœuds labellisés
# ===============================

labels_sub = np.array(labels_sub)
embedding = np.array(embedding)

mask_labeled = labels_sub != -1

X = embedding[mask_labeled]
y = labels_sub[mask_labeled]

# ===============================
# 2. Train / Test
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# 3. k-NN
# ===============================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# ===============================
# 4. Matrice de confusion
# ===============================

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")
plt.title("Matrice de confusion - kNN sur embedding UMAP")
plt.show()
