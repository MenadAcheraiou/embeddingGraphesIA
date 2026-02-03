import pandas as pd
import numpy as np


def regression(fichier):
    # 1. Chargement des données
    print("Chargement des fichiers...")
    # On charge les embeddings sans présumer du format de l'index pour l'instant
    df_embeddings = pd.read_csv('embedings.csv', index_col=0)

    # On charge les labels
    df_labels = pd.read_csv(
        'NODES_climatoscope_graph_2022-07-01_2022-10-30_th=3.0_flc=0_world_2185-pro-anti_Louvaindic_testtop_tableusers.csv')

    # 2. NETTOYAGE DES IDs (L'étape cruciale)
    def clean_id(val):
        """Transforme n'importe quel ID (123, 123.0, '123') en string propre '123'"""
        try:
            # Convertir en float d'abord pour gérer les "123.0", puis en int, puis en str
            return str(int(float(val)))
        except:
            return str(val)

    # On applique ce nettoyage aux deux listes
    df_embeddings.index = df_embeddings.index.map(clean_id)
    df_labels['Id'] = df_labels['Id'].map(clean_id)

    print(f"Exemple ID Embedding nettoyé : {df_embeddings.index[0]}")
    print(f"Exemple ID Label nettoyé     : {df_labels['Id'].iloc[0]}")

    # 3. FUSION AUTOMATIQUE (Plus rapide et sûr qu'une boucle for)
    # On ne garde que les IDs qui sont dans les deux fichiers
    common_ids = df_labels[df_labels['Id'].isin(df_embeddings.index)].copy()

    print(f"\n--- RÉSULTAT ---")
    print(f"Utilisateurs communs trouvés : {len(common_ids)}")

    if len(common_ids) > 0:
        # 4. PRÉPARATION FINALE
        # On filtre les opinions neutres (on garde que 0 et 4)
        common_ids = common_ids[common_ids['modularity_class'].isin([0, 4])]

        # On aligne les données : pour chaque Label, on va chercher son Embedding
        # .loc[] est magique : il va chercher la ligne correspondante dans le désordre
        X = df_embeddings.loc[common_ids['Id']].values

        # On crée la cible (1 pour Pro-Climat, 0 pour Sceptique)
        y_reg = np.where(common_ids['modularity_class'] == 4, 1, 0)
        y_cls = np.where(common_ids['modularity_class'] == 4, 'Pro-Climat', 'Sceptique')

        print(f"Données prêtes pour la régression : {len(X)} lignes.")

        # 5. TEST RAPIDE
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report

        X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        print("\nScore de test :")
        print(classification_report(y_test, model.predict(X_test)))

    else:
        print("Aperçu de l'index embeddings :", df_embeddings.index[:5].tolist())
