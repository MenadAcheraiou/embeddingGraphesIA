from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA


df_embeddings=pd.read_csv('embedings.csv', index_col=0)
file_labels = 'NODES_climatoscope_graph_2022-07-01_2022-10-30_th=3.0_flc=0_world_2185-pro-anti_Louvaindic_testtop_tableusers.csv'
df_labels = pd.read_csv(file_labels)
df_labels['Id'] = df_labels['Id'].astype(str)
df_embeddings.index = df_embeddings.index.astype(str)

commonIds=df_labels[df_labels['Id'].isin(df_embeddings.index)]
X=df_embeddings.loc[commonIds['Id']].values
y_true = commonIds['modularity_class'].map({4: 'Pro-Climat', 0: 'Sceptique'}).fillna('Neutre/Autre')


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

# Graphique A : Les groupes trouv�s par le K-means (L'IA)
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Groupes trouv�s par K-means (IA)')
plt.colorbar(label='Cluster ID')

# Graphique B : Les vraies opinions (La Modularit�)
plt.subplot(1, 2, 2)
# On convertit les noms en chiffres pour la couleur
y_colors = pd.factorize(y_true)[0]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_colors, cmap='coolwarm', alpha=0.6)
plt.title('Vraies opinions (Modularity Class)')

plt.tight_layout()
plt.show()

