import sys

import mlflow
from mlflow import MlflowClient


# Nom du modele donne dans la commande
nom = sys.argv[1]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Trouver la version la plus recente
versions = client.search_model_versions(f"name='{nom}'")
derniere = max(int(v.version) for v in versions)

# Attribuer l'alias production a cette version
client.set_registered_model_alias(
    nom,
    "production",
    str(derniere)
)

print(
    f"{nom} : version {derniere} promue "
    "(alias production)"
)