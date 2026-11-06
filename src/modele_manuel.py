import time
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# 1. Charger les memes donnees et utiliser la meme separation
df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)


# 2. Entrainer le modele manuel
print("Entrainement de la Random Forest en cours...")

t0 = time.time()

modele = RandomForestClassifier(
    n_estimators=500,
    max_depth=16,
    random_state=42
)

modele.fit(Xtr, ytr)

duree = round(time.time() - t0, 1)
f1 = round(f1_score(yte, modele.predict(Xte)), 4)

print("Modele manuel : Random Forest")
print("F1 sur le jeu de test :", f1)
print("Duree d'entrainement :", duree, "secondes")


# 3. Sauvegarder une copie locale pour mesurer sa taille
Path("models").mkdir(exist_ok=True)
joblib.dump(modele, "models/model_manuel.pkl")


# 4. Enregistrer les resultats et le modele dans MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp5-automl-deploiement")

with mlflow.start_run(run_name="manuel-rf-500-16"):
    mlflow.log_params({
        "outil": "scikit-learn",
        "n_estimators": 500,
        "max_depth": 16
    })

    mlflow.log_metrics({
        "f1": f1,
        "duree_entrainement_s": duree
    })

    info = mlflow.sklearn.log_model(
        modele,
        name="model",
        input_example=Xte.head(3),
        registered_model_name="fraude-manuel",
        serialization_format="skops",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print(
    "Enregistre : fraude-manuel, version",
    info.registered_model_version
)