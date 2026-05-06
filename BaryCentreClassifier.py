import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

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

        # Barycentres des deux classes
        self.b_pos = X[y == self.label_positive].mean(axis=0)
        self.b_neg = X[y == self.label_negative].mean(axis=0)

        # Milieu entre les deux barycentres
        self.midpoint = (self.b_pos + self.b_neg) / 2

        # Vecteur source : du centre vers le barycentre positif
        v = self.b_pos - self.midpoint

        # Scale
        scale = np.linalg.norm(self.target) / np.linalg.norm(v)

        # Rotation
        angle_v = np.arctan2(v[1], v[0])
        angle_target = np.arctan2(self.target[1], self.target[0])
        theta = angle_target - angle_v

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Matrice finale
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

        # Frontière : y = -x, donc x + y = 0
        scores = X_t[:, 0] + X_t[:, 1]

        return np.where(scores > 0, self.label_positive, self.label_negative)

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)

    def plot(self, X, y, title=""):
        X = np.array(X)
        y = np.array(y)

        X_t = self.transform(X)
        acc = self.score(X, y)

        plt.figure(figsize=(6, 6))

        # Points label 4
        mask_pos = y == self.label_positive
        plt.scatter(
            X_t[mask_pos, 0],
            X_t[mask_pos, 1],
            s=12,
            alpha=0.7,
            label=f"Classe {self.label_positive}"
        )

        # Points label 0
        mask_neg = y == self.label_negative
        plt.scatter(
            X_t[mask_neg, 0],
            X_t[mask_neg, 1],
            s=12,
            alpha=0.7,
            label=f"Classe {self.label_negative}"
        )

        # Droite y = -x
        x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
        x_line = np.linspace(x_min, x_max, 200)
        plt.plot(x_line, -x_line, "k--", label="y = -x")

        # Barycentres transformés
        b_pos_t = self.transform([self.b_pos])[0]
        b_neg_t = self.transform([self.b_neg])[0]

        plt.scatter(
            b_pos_t[0],
            b_pos_t[1],
            s=180,
            marker="*",
            label=f"Barycentre {self.label_positive}"
        )

        plt.scatter(
            b_neg_t[0],
            b_neg_t[1],
            s=180,
            marker="*",
            label=f"Barycentre {self.label_negative}"
        )

        plt.title(f"{title} | précision = {acc:.3f}")
        plt.axis("equal")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.show()


    def confusion_matrix(self, X, y):
        """
        Retourne la matrice de confusion du modèle barycentre.
        """
        y = np.array(y)
        y_pred = self.predict(X)

        return confusion_matrix(
            y,
            y_pred,
            labels=[self.label_negative, self.label_positive]
        )

    def plot_confusion_matrix(self, X, y, title="Matrice de confusion - Barycentre"):
        """
        Affiche la matrice de confusion du modèle barycentre.
        """
        cm = self.confusion_matrix(X, y)

        display_labels = [
            f"Classe {self.label_negative}",
            f"Classe {self.label_positive}"
        ]

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels
        )

        disp.plot(cmap="Blues", colorbar=False)
        plt.title(title)
        plt.grid(False)
        plt.show()