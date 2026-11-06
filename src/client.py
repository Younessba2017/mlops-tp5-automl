import sys
import time
import statistics

import requests


port = sys.argv[1] if len(sys.argv) > 1 else "8100"
URL = f"http://127.0.0.1:{port}/invocations"

# Transaction a soumettre au modele
CORPS = {
    "dataframe_records": [
        {
            "montant": 8500.0,
            "anciennete": 3,
            "freq_24h": 9,
            "nuit": 1
        }
    ]
}


def envoyer():
    reponse = requests.post(URL, json=CORPS, timeout=10)

    if not reponse.ok:
        print("Erreur du service :", reponse.text)

    reponse.raise_for_status()
    return reponse


# 1. Verifier une premiere prediction
r = envoyer()
print("Reponse :", r.status_code, r.json())

# 2. Effectuer 10 appels d'echauffement non mesures
print("Echauffement : 10 appels...")

for _ in range(10):
    envoyer()

# 3. Chronometrer 200 appels
print("Mesure : 200 appels...")
latences = []

for _ in range(200):
    debut = time.perf_counter()
    envoyer()
    latences.append((time.perf_counter() - debut) * 1000)

mediane = statistics.median(latences)
p95 = statistics.quantiles(latences, n=20)[18]

print(
    f"Port {port} | mediane : {mediane:.1f} ms"
    f" | p95 : {p95:.1f} ms"
)