Pour le Node2vec et UMAP, il y a trois notebooks selon le decoupage temporel des donnees :

- Umap_03_parts.ipynb : pipeline sur 3 periodes (debut / milieu / fin)
- Umap_10_parts.ipynb : pipeline sur 10 parties (part_01 a part_10)
- Umap_2Sem_parts.ipynb : pipeline sur des fenetres glissantes de 2 semaines (environ 59 graphes, janvier a decembre 2022)

Les trois notebooks ont exactement le meme pipeline, seule la liste des fichiers traites change.

Tous les fichiers (notebooks, fichiers CSV des graphes, nodes.csv, et fichiers embeddings) doivent etre places dans le meme dossier.
Les fichiers Python fournis fonctions.py, embeddingsFunctions.py et BaryCentreClassifier.py doivent aussi etre presents dans ce meme dossier.

Le fichier nodes.csv doit avoir le separateur ; et contenir les colonnes Id et modularity_class.
Les fichiers CSV des graphes doivent contenir les colonnes user_id, original_author et nb_retweeted.

La generation des embeddings Node2Vec (cellule 4) est tres longue. Tous les fichiers embeddings32_*.csv sont deja pre-calcules et disponibles sur le Drive :

https://drive.google.com/file/d/12kWeLeBPykfxrYCIQJ2xzItx4sg2iTLb/view?usp=sharing

Il suffit de les telecharger et de les mettre dans le meme dossier, la cellule 4 peut alors etre ignoree.

Pour lancer un notebook, ouvrir Jupyter, placer tous les fichiers dans le meme dossier, et executer toutes les cellules dans l'ordre (la cellule 3 peut etre ignorée aussi).

Librairies a installer :
pip install pandas numpy matplotlib seaborn scikit-learn networkx umap-learn node2vec

Attention : le paquet s'appelle umap-learn sur PyPI mais s'importe avec "import umap" dans le code.