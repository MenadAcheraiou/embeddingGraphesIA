import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


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