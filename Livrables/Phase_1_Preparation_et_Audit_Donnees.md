# Phase 1 — Préparation, Audit des Données et Stratégie de Partitionnement

> **Projet :** ReqMind AI — Assistant Intelligent d'Ingénierie des Exigences Logicielle  
> **Auteurs :** Zakariae Chelle & Yasmine Isakrouhen  
> **Statut :** Phase 1 Complète & Validée (100% Tests Passés)  
> **Date de référence :** Octobre 2026  

---

## 1. Contexte & Objectifs de la Phase 1

L'objectif fondamental de la **Phase 1** est d'établir le socle de données scientifique, reproductible et sans fuite d'information (*data leakage*) pour l'ensemble du système **ReqMind AI**.

En ingénierie des exigences logicielles, les erreurs d'interprétation et les ambiguïtés représentent jusqu'à **40% à 50% des défauts** découverts ultérieurement lors des phases de développement et de recette. Pour entraîner et évaluer des modèles d'intelligence artificielle fiables, il est impératif :
1. De disposer d'un corpus de référence étiqueté, vérifié et documenté.
2. D'établir une politique de prétraitement conservatrice qui ne détruit pas le sens technique des phrases.
3. De figer des partitions d'entraînement, de validation et de test strictement hermétiques.
4. De définir des règles claires d'annotation pour la classification (FR vs NFR), la détection d'ambiguïté et la détection de quasi-doublons.

---

## 2. Audit du Jeu de Données de Référence (PROMISE)

### 2.1 Provenance et Caractéristiques Globales
Le corpus sélectionné est le jeu de données **PROMISE Software Engineering Repository** (*Cleland-Huang et al., 2007*), qui constitue la référence académique pour la classification d'exigences logicielles.

* **Fichier source :** `Data/PROMISE_exp.csv`
* **Nombre total d'exigences :** **969 exigences**
* **Nombre de projets logiciels distincts :** **47 projets** (systèmes d'information hospitaliers, contrôle de feux de circulation, ordonnancement de réunions, etc.)

### 2.2 Distribution des Classes

#### Répartition Binaire (Tâche Principale) :
* **Exigences Fonctionnelles (FR) :** **444 exigences (45.82%)**  
  *Exemple :* *"The system shall allow users to search available meeting slots by date and attendee."*
* **Exigences Non-Fonctionnelles (NFR) :** **525 exigences (54.18%)**  
  *Exemple :* *"The database response time for search queries shall not exceed 500 milliseconds."*

#### Répartition des 11 Sous-Classes Non-Fonctionnelles (Qualité & Contraintes) :
| Code | Catégorie Non-Fonctionnelle | Nombre d'Exigences | Pourcentage du Total |
| :---: | :--- | :---: | :---: |
| **SE** | Security (Sécurité / Confidentialité) | 125 | 12.90% |
| **US** | Usability (Facilité d'utilisation / Ergonomie) | 85 | 8.77% |
| **O** | Operational (Exploitation & Environnement) | 77 | 7.95% |
| **PE** | Performance (Temps de réponse, débit) | 67 | 6.91% |
| **LF** | Look and Feel (Apparence visuelle, IHM) | 49 | 5.06% |
| **A** | Availability (Disponibilité / Tolérance aux pannes) | 31 | 3.20% |
| **MN** | Maintainability (Maintenabilité du code) | 24 | 2.48% |
| **SC** | Scalability (Montée en charge) | 22 | 2.27% |
| **FT** | Fault Tolerance (Robustesse / Reprise sur erreur) | 18 | 1.86% |
| **L** | Legal & Compliance (Réglementations et normes) | 15 | 1.55% |
| **PO** | Portability (Portabilité matérielle et OS) | 12 | 1.24% |

---

## 3. Prétraitement Conservateur des Textes

Contrairement aux tâches classiques de NLP (comme l'analyse de sentiment sur Twitter), les exigences logicielles possèdent une grammaire normative très précise. Une politique stricte de **normalisation conservatrice** a été formalisée :

1. **Préservation absolue des négations :**  
   Les mots `not`, `never`, `no`, `without` sont impérativement conservés. Une exigence de sécurité telle que *"The system shall not store plain text passwords"* devient l'exact opposé si la négation est supprimée.
2. **Préservation des verbes modaux :**  
   Les verbes `shall`, `must`, `should`, `may` indiquent le niveau normatif de l'exigence (obligation légale vs recommandation).
3. **Préservation des valeurs numériques et unités :**  
   Des métriques comme `99.95%`, `250ms`, `500 Mo` constituent le cœur même d'une exigence de performance ou de disponibilité.
4. **Préservation des décalages de caractères (*Character Offsets*) :**  
   Chaque terme analysé doit pouvoir être surligné dans l'interface utilisateur avec ses coordonnées précises `(source_start, source_end)`.

---

## 4. Stratégie de Partitionnement (Train / Validation / Test)

Pour éviter tout biais d'évaluation, le jeu de données a été découpé et sauvegardé dans des fichiers CSV figés à l'aide d'un générateur pseudo-aléatoire déterministe (`seed=42`).

### 4.1 Répartition des Fentes (*Splits*)
* **Entraînement (*Train*, 70%) :** `Data/splits/train.csv` (678 exigences)
* **Validation (*Val*, 15%) :** `Data/splits/val.csv` (145 exigences)
* **Test Tenu à l'Écart (*Held-Out Test*, 15%) :** `Data/splits/test.csv` (146 exigences)

### 4.2 Tableau d'Audit de Stratification
La stratification a été appliquée sur les sous-classes fines afin de garantir qu'aucune sous-classe rare (ex: `PO` avec 12 exemples) ne soit absente du jeu de test :

| Fente | Total Exigences | FR (Fonctionnelles) | NFR (Non-Fonctionnelles) | Ratio FR | Ratio NFR | Fichier |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | **678** | 311 | 367 | 45.87% | 54.13% | [train.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/train.csv) |
| **Validation** | **145** | 66 | 79 | 45.52% | 54.48% | [val.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/val.csv) |
| **Test** | **146** | 67 | 79 | 45.89% | 54.11% | [test.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/test.csv) |
| **Total Global** | **969** | **444** | **525** | **45.82%** | **54.18%** | Stratification parfaite |

### 4.3 Vérification Automatisée de l'Absence de Fuite (*Zero Data Leakage*)
Un algorithme de contrôle automatisé vérifie qu'aucune phrase (en comparaison exacte insensible à la casse et après normalisation des espaces) n'est partagée entre les fentes :
$$\text{Train} \cap \text{Val} = \emptyset \quad (\text{0 doublons})$$
$$\text{Train} \cap \text{Test} = \emptyset \quad (\text{0 doublons})$$
$$\text{Val} \cap \text{Test} = \emptyset \quad (\text{0 doublons})$$

---

## 5. Guides d'Annotation & Vérités Terrains

Pour évaluer les modules NLP ultérieurs, des rubriques précises ont été établies dans [docs/annotation_guidelines.md](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/docs/annotation_guidelines.md) :

1. **Classification FR vs NFR :**
   * *FR :* Action ou comportement déclenché par un utilisateur ou un système (*"QUOI faire"*).
   * *NFR :* Critère de qualité, contrainte d'environnement, niveau de service (*"COMMENT le faire avec quelle qualité"*).
2. **Ambiguïtés Lexicales :**
   * Fichier étalon : `Data/annotations/ambiguity_gold.json` (85 cas étiquetés).
   * Catégories : Termes vagues (*user-friendly*, *fast*, *flexible*), quantifieurs flous (*various*, *several*, *sufficient*), passifs sans acteur identifié.
3. **Quasi-Doublons & Similarité Sémantique :**
   * Fichier étalon : `Data/annotations/duplicate_pairs_gold.json` (55 paires étiquetées).
   * Inclut des quasi-doublons réels, des doublons partiels et des *hard negatives* (ex: même action mais actant différent ou négation inversée).

---

## 6. Architecture Technique & Validation des Tests

Le module [src/reqmind/data_loader.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/src/reqmind/data_loader.py) encapsule toute la logique de chargement et de vérification.

### Résultat de la suite de tests automatisés :
```text
tests/test_data_splits.py::test_raw_promise_loading PASSED
tests/test_data_splits.py::test_split_proportions_and_stratification PASSED
tests/test_data_splits.py::test_zero_data_leakage PASSED
tests/test_data_splits.py::test_saved_splits_integrity PASSED
tests/test_data_splits.py::test_ambiguity_gold_annotations_validity PASSED
tests/test_data_splits.py::test_duplicate_pairs_gold_annotations_validity PASSED
============================== 6 passed in 0.42s ===============================
```

### Documents et Fichiers Produits en Phase 1 :
* **Splits de données :** [train.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/train.csv), [val.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/val.csv), [test.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/test.csv)
* **Dataset Card :** [docs/data_card.md](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/docs/data_card.md)
* **Guide d'Annotation :** [docs/annotation_guidelines.md](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/docs/annotation_guidelines.md)
* **Notebook d'Exploration :** [models/ReqMind_AI_Data_Exploration.ipynb](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/models/ReqMind_AI_Data_Exploration.ipynb)
* **Rapport Formel de Phase :** [reports/phases/phase_1.md](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/phases/phase_1.md)
