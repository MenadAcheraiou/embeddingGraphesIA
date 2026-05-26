import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class BarycenterClassifier:
    def __init__(self, label_positive=4, label_negative=0, target=(2, 2)):
        self.label_positive = label_positive
        self.label_negative = label_negative
        self.target = np.array(target, dtype=float)

        self.b_pos = None
        self.b_neg = None
        self.midpoint = None
        self.A = None
        self.is_fitted = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.b_pos = X[y == self.label_positive].mean(axis=0)
        self.b_neg = X[y == self.label_negative].mean(axis=0)
        self.midpoint = (self.b_pos + self.b_neg) / 2

        v = self.b_pos - self.midpoint
        target = np.array([2.0, 2.0])

        scale = np.linalg.norm(target) / np.linalg.norm(v)

        angle_v = np.arctan2(v[1], v[0])
        angle_target = np.arctan2(target[1], target[0])
        theta = angle_target - angle_v

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        self.A = scale * R
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Tu dois appeler fit() avant transform().")
        X = np.array(X)
        return (self.A @ (X - self.midpoint).T).T

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        X_t = self.transform(X)
        scores = X_t[:, 0] + X_t[:, 1]
        return np.where(scores > 0, self.label_positive, self.label_negative)

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    @staticmethod
    def _confidence_interval(p, n, z=1.96):
        """Intervalle de confiance à 95% via erreur standard."""
        if n == 0:
            return 0.0
        return z * np.sqrt(p * (1 - p) / n)

    def plot(self, X, y, title=""):
        X = np.array(X)
        y = np.array(y)

        X_t = self.transform(X)
        acc = self.score(X, y)

        plt.figure(figsize=(6, 6))

        mask_pos = y == self.label_positive
        plt.scatter(X_t[mask_pos, 0], X_t[mask_pos, 1], s=12, alpha=0.7, label=f"Classe {self.label_positive}")

        mask_neg = y == self.label_negative
        plt.scatter(X_t[mask_neg, 0], X_t[mask_neg, 1], s=12, alpha=0.7, label=f"Classe {self.label_negative}")

        x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
        x_line = np.linspace(x_min, x_max, 200)
        plt.plot(x_line, -x_line, "k--", label="y = -x")

        b_pos_t = self.transform([self.b_pos])[0]
        b_neg_t = self.transform([self.b_neg])[0]

        plt.scatter(b_pos_t[0], b_pos_t[1], s=180, marker="*", label=f"Barycentre {self.label_positive}")
        plt.scatter(b_neg_t[0], b_neg_t[1], s=180, marker="*", label=f"Barycentre {self.label_negative}")

        plt.title(f"{title} | précision = {acc:.3f}")
        plt.axis("equal")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.show()

    def confusion_matrix(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred, labels=[self.label_negative, self.label_positive])

    def plot_confusion_matrix(self, X, y, title="Matrice de confusion - Barycentre"):
        y = np.array(y)
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred, labels=[self.label_negative, self.label_positive])

        # --- Calcul des pourcentages et intervalles de confiance par case ---
        # cm = [[TN, FP], [FN, TP]]
        TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        n_neg = TN + FP   # total vrais négatifs
        n_pos = FN + TP   # total vrais positifs
        n_total = len(y)

        # Pourcentages par case (sur le total de la ligne = classe réelle)
        cases = {
            (0,0): (TN, n_neg),   # TN : % parmi classe négative
            (0,1): (FP, n_neg),   # FP : % parmi classe négative
            (1,0): (FN, n_pos),   # FN : % parmi classe positive
            (1,1): (TP, n_pos),   # TP : % parmi classe positive
        }

        # --- Calcul précision, recall + IC ---
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy  = (TP + TN) / n_total

        ic_precision = self._confidence_interval(precision, TP + FP)
        ic_recall    = self._confidence_interval(recall,    TP + FN)
        ic_accuracy  = self._confidence_interval(accuracy,  n_total)

        # --- Plot manuel de la matrice ---
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, interpolation='nearest', cmap='Blues')

        labels = [f"Classe {self.label_negative}", f"Classe {self.label_positive}"]

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")

        for (i, j), (count, n_row) in cases.items():
            p = count / n_row if n_row > 0 else 0.0
            ic = self._confidence_interval(p, n_row)
            text = f"{p*100:.1f}%\n±{ic*100:.1f}%\n(n={count})"
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)

        # --- Titre avec accuracy ± IC ---
        full_title = (
            f"{title}\n"
            f"Accuracy : {accuracy*100:.1f}% ± {ic_accuracy*100:.1f}%  |  "
            f"Précision : {precision*100:.1f}% ± {ic_precision*100:.1f}%  |  "
            f"Recall : {recall*100:.1f}% ± {ic_recall*100:.1f}%"
        )
        ax.set_title(full_title, fontsize=10)
        plt.tight_layout()
        plt.show()