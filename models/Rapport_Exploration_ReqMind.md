# Rapport court — Exploration PROMISE_exp

- Nombre d'exigences : **969**.
- Nombre de colonnes initiales : **3** (les colonnes calculées sont exclues ; vérifier en cas de changement de schéma).
- Valeurs manquantes (texte) : **0**.
- Textes dupliqués : **0**.
- Fonctionnelles : **444**.
- Non fonctionnelles : **525**.
- Longueur médiane : **15 mots**.
- Classe la moins fréquente : **PO (12 exemples)**.

## Analyse
Corpus annoté utile pour la classification fonctionnelle / non fonctionnelle.
Les classes sont déséquilibrées : utiliser macro-F1 et un découpage par projet si possible.
Ce corpus n'annotant pas directement l'ambiguïté ou la contradiction, ces tâches demandent des données annotées séparément.

## Prétraitement
Nettoyage prudent des espaces, conservation des négations, nombres et textes originaux.

## Suite
Préparer une séparation entraînement/validation/test sans fuite et comparer les baselines TF-IDF et embeddings.

**Aucun modèle n'a été entraîné dans cette exploration.**
