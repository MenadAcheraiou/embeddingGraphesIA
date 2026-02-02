import pandas as pd 
import glob
import os

if not  os.path.exists("embedings.csv"):

    fichiers_csv=glob.glob(os.path.join('.', "graph_*.csv"))

    print(f"Trouvé {len(fichiers_csv)} fichiers à fusionner.")
    liste_df=[]

    for fichier in fichiers_csv:
        print(f"Lecture du fichier {fichier}...")
        df=pd.read_csv(fichier)
        liste_df.append(df)

    df_global = pd.concat(liste_df, ignore_index=True)
    df_unique =df_global.drop_duplicates(subset=['user_id','original_author'])

    #Creation du graphe :
    import networkx as nx
    G=nx.from_pandas_edgelist(df_unique, source='user_id', target='original_author')

    print(f"Le graphe final contient {G.number_of_nodes()} utilisateurs.")
    from node2vec import Node2Vec
    #Exploration :
    node2vec=Node2Vec(G, dimensions=32, walk_length=10, num_walks=10, workers=2)

    model=node2vec.fit(window=10, min_count=1, batch_words=4)

    nodes =list(G.nodes)
    embeddings=[model.wv[str(node)] for node in nodes]

    pd_save=pd.DataFrame(embeddings, index=nodes)

    pd_save.to_csv("embedings.csv")
    embeddings = {str(nodes[i]): embeddings[i] for i in range(len(nodes))}
else:

    df_charge = pd.read_csv('embedings.csv', index_col=0)
    embeddings = {str(idx): row.values for idx, row in df_charge.iterrows()}


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


file_labels = 'NODES_climatoscope_graph_2022-07-01_2022-10-30_th=3.0_flc=0_world_2185-pro-anti_Louvaindic_testtop_tableusers.csv'
df_labels = pd.read_csv(file_labels)
df_labels['Id'] = df_labels['Id'].astype(str)

X = []
y = []

for index,row in df_labels.iterrows():
    node_id = str(row['Id'])
    if node_id in embeddings:
        X.append(embeddings[node_id])
        # Dummy label assignment for demonstration purposes
        classe=row['modularity_class']
        opinion = 'Pro-Climat' if classe == 4 else ('Sceptique' if classe == 0 else 'Neutre/Autre')
        y.append(opinion)

X = np.array(X)
y = np.array(y) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Entraînement sur : {len(X_train)} personnes")
print(f"Test sur : {len(X_test)} personnes")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. Initialisation et Entraînement ---
# On choisit K=5 (un classique)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


# --- 2. Prédiction ---
y_pred = knn.predict(X_test_scaled) 
print(classification_report(y_test, y_pred))