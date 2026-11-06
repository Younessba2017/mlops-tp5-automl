# Modèle manuel ou modèle AutoML ? — TP05

Les deux modèles utilisent les mêmes données et la même séparation :
75 % pour l'entraînement, 25 % pour le test, avec random_state=42.

## Résultats

| Critère | Manuel — Random Forest | AutoML — FLAML |
|---|---|---|
| Famille de modèle | Random Forest | XGBoost |
| F1 sur le jeu de test | 0,8669 | 0,8561 |
| Recherche automatique | Aucune dans ce TP | 90,2 s, entraînements inclus |
| Entraînement du modèle manuel | 1,8 s | Inclus dans la recherche |
| Temps humain | Réglage au TP2, non chronométré | Préparation et lancement, non chronométrés |
| Latence médiane — mesure 1 | 49,9 ms | 10,1 ms |
| Latence médiane — mesure 2 | 48,2 ms | 10,4 ms |
| Latence p95 — mesure 1 | 76,1 ms | 32,9 ms |
| Latence p95 — mesure 2 | 71,0 ms | 32,3 ms |
| Taille de la copie locale .pkl | 13,37 Mo | 0,28 Mo |
| Explicabilité | Importance des variables ; 500 arbres, profondeur maximale 16 | Importance des variables ; 322 arbres, au plus 4 feuilles par arbre |

Les temps de recherche AutoML et d'entraînement manuel correspondent
à des opérations différentes : ils ne mesurent pas le même travail.
Les tailles sont celles des copies locales joblib, pas des packages
MLflow complets.

## Vérification des services

Les deux services répondent avec le code HTTP 200 et prédisent la
classe 1 pour la transaction testée.

Une requête contenant uniquement montant est rejetée par le service
AutoML. Le message indique les entrées manquantes : anciennete,
freq_24h et nuit. La signature du modèle contrôle donc les colonnes
requises.

## Décision

Je retiens XGBoost comme candidat au déploiement pour sa rapidité et sa faible taille.
Sa latence médiane est proche de 10 ms, contre environ 49 ms pour la Random Forest.
Sa copie locale occupe 0,28 Mo, contre 13,37 Mo pour le modèle manuel.
Son F1 est toutefois inférieur de 0,0108, soit 1,08 point sur une échelle en pourcentage.
Avant une mise en production réelle, il faut vérifier que cette perte de qualité est acceptable au regard du coût des fraudes manquées.