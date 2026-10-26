# Phase 3 — Entraînement, Fine-Tuning & Benchmark Comparatif des Modèles

> **Projet :** ReqMind AI — Assistant Intelligent d'Ingénierie des Exigences Logicielle  
> **Auteurs :** Zakariae Chelle & Yasmine Isakrouhen  
> **Statut :** Phase 3 Complète & Validée (100% Tests Passés)  
> **Date de référence :** Octobre 2026  

---

## 1. Contexte & Protocole Expérimental Strict

L'objectif de la **Phase 3** est de concevoir, entraîner, comparer et intégrer des modèles d'intelligence artificielle pour la classification binaire des exigences logicielles en :
* **FR (Exigence Fonctionnelle)** : Fonctionnalités, comportements métier, entrées/sorties.
* **NFR (Exigence Non-Fonctionnelle)** : Critères de qualité, performances, sécurité, ergonomie.

### Le Protocole Académique en 4 Étapes :
1. **Entraînement strict sur `train.csv` (678 exemples) :** Aucun modèle ne voit les données de validation ou de test pendant son entraînement.
2. **Optimisation des hyperparamètres sur `val.csv` (145 exemples) :** Recherche sur grille de l'hyperparamètre de régularisation $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ guidée par le **Macro-F1 score**.
3. **Évaluation unique sur `test.csv` (146 exemples) :** Calcul impartial des performances finales sur un jeu de données tenu à l'écart.
4. **Test de résistance inter-projets (*Cross-Project Generalization*) :** Validation croisée à 5 plis (`GroupKFold` sur les 47 projets distincts) afin de simuler l'arrivée d'un projet logiciel 100% inédit en entreprise.

---

## 2. Les Modèles Évalués & Progression Méthodologique

Pour démontrer la valeur ajoutée de chaque avancée, nous avons évalué 5 architectures complémentaires :

1. **Majority Baseline :**  
   Prédit systématiquement la classe majoritaire (`NFR`). Sert de seuil minimal d'utilité (*Floor performance*).
2. **TF-IDF + Régression Logistique :**  
   Modèle linéaire classique basé sur les fréquences de termes (n-grams 1 et 2), pondération TF sous-linéaire, sans suppression de stopwords. Hyperparamètre optimal : $C=10.0$.
3. **TF-IDF + Linear SVM (Support Vector Machine) :**  
   Séparateur à marge maximale utilisant la représentation vectorielle lexicale. Hyperparamètre optimal : $C=1.0$.
4. **Embeddings Figés (MiniLM) + Régression Logistique :**  
   Extraction d'embeddings denses de 384 dimensions via le modèle pré-entraîné `sentence-transformers/all-MiniLM-L6-v2`, suivis d'un classifieur linéaire. Les poids du Transformer restent figés.
5. **SetFit Fine-Tuné (Sentence Transformer Fine-Tuning) — CHAMPION 🚀 :**  
   Fine-tuning contrastif direct de `all-MiniLM-L6-v2` sur nos 678 exigences avec perte Cosine Similarity, puis entraînement d'une tête de classification logistique différentiable.

---

## 3. Résultats Comparatifs Détaillés

### 3.1 Tableau de Synthèse des Performances

| Modèle | Accuracy | Macro-F1 | F1 (FR) | F1 (NFR) | Inter-Projets F1 (GroupKFold) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 54.11% | 35.11% | 0.00% | 70.22% | N/A |
| **Embeddings Figés (MiniLM)** | 85.62% | 85.58% | 84.89% | 86.27% | 78.79% |
| **TF-IDF + LogReg** | 91.78% | 91.68% | 90.77% | 92.59% | 70.16% |
| **TF-IDF + Linear SVM** | 91.78% | 91.68% | 90.77% | 92.59% | 70.26% |
| **SetFit Fine-Tuné (MiniLM) 🚀** | **92.47%** | **92.42%** | **91.85%** | **92.99%** | **80.68%** |

---

## 4. Les Matrices de Confusion (Jeu de Test, $N=146$)

Le jeu de test comprend **67 exigences FR** et **79 exigences NFR**.

### 1. Majority Baseline :
* Prédit 146 fois NFR.
* **Vrais FR détectés : 0 / 67** | Faux NFR : 67 | Faux FR : 0 | Vrais NFR : 79

### 2. Embeddings Figés MiniLM :
* Vrais FR détectés : **59** | Faux NFR : 8
* Faux FR : 13 | Vrais NFR détectés : **66**
* *Constat :* Sans fine-tuning, le modèle générique confond certaines contraintes fonctionnelles et non-fonctionnelles.

### 3. TF-IDF + LogReg & TF-IDF + Linear SVM :
* Vrais FR détectés : **59** | Faux NFR : 8
* Faux FR : 4 | Vrais NFR détectés : **75**
* *Constat :* Très forte précision sur NFR (75/79), mais 8 erreurs sur les FR contenant des verbes passifs ou modaux.

### 4. SetFit Fine-Tuné 🚀 :
* Vrais FR détectés : **62 (+3)** | Faux NFR : **5 (-3)**
* Faux FR : **6** | Vrais NFR détectés : **73**
* **Total des erreurs : seulement 11 sur 146 (Taux d'erreur de 7.53%).**
* *Constat :* Meilleur équilibre global et meilleure détection des exigences fonctionnelles subtiles.

---

## 5. La Découverte Scientifique Majeure : Le Piège du TF-IDF

En comparant le score sur le Test Split classique et le score en validation croisée inter-projets (`GroupKFold`), un phénomène fondamental a été mis en évidence :

```text
Score Macro-F1 (%)
   100 ┌─────────────────────────────────────────────────────────────┐
    90 │  TF-IDF (Test Split) : 91.7%     SetFit (Test Split) : 92.4%│
    80 │                                  SetFit (Inter-Projets) : 80.7% (Gagnant)
    70 │  TF-IDF (Inter-Projets) : 70.2% (Effondrement de 21.5 pts)   │
    60 │                                                             │
       └─────────────────────────────────────────────────────────────┘
```

### Explication Technique :
1. **Surapprentissage Lexical du TF-IDF :**  
   Sur le jeu de test tenu à l'écart, le TF-IDF obtient 91.7% car certains mots-clés spécifiques aux projets (ex: *"patient"*, *"traffic"*, *"grid"*) apparaissent à la fois dans l'entraînement et dans le test. Dès que l'on teste le TF-IDF sur un **projet logiciel inédit** (`GroupKFold`), son vocabulaire de surface disparaît et son score chute dramatiquement à **70.16%** (-21.5 points).
2. **Généralisation Sémantique de SetFit :**  
   SetFit ne mémorise pas des mots isolés. Grâce à son entraînement contrastif de phrases, il comprend l'intention sémantique sous-jacente. Sur des projets 100% inconnus, il conserve un score remarquable de **80.68%** (**+10.52 points au-dessus du TF-IDF**).

### Distribution des Folds Inter-Projets (Scores en Macro-F1 %) :
| Plis Inter-Projets | TF-IDF + LogReg | TF-IDF + SVM | Embeddings Figés | SetFit Fine-Tuné 🚀 |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 69.38% | 69.09% | 66.11% | **73.46%** |
| **Fold 2** | 72.71% | 71.73% | 86.86% | **89.67%** |
| **Fold 3** | 74.13% | 72.06% | 83.54% | **86.96%** |
| **Fold 4** | 66.64% | 68.95% | 80.33% | **72.87%** |
| **Fold 5** | 67.96% | 69.49% | 77.09% | **80.45%** |
| **Moyenne $\pm$ Écart-Type** | **70.16% $\pm$ 2.8%** | **70.26% $\pm$ 1.3%** | **78.79% $\pm$ 7.1%** | **80.68% $\pm$ 6.8%** |

---

## 6. Analyse Qualitative des Erreurs (*Error Analysis*)

Sur les 146 exigences de test, SetFit ne commet que **11 erreurs**. L'inspection de `reports/results/phase3_setfit_test_errors.csv` révèle deux causes principales :

1. **Exigences Hybrides / Mixtes (Action métier + Contrainte de sécurité) :**  
   *Exemple :* *"The system shall authenticate users and lock the session after 3 failed attempts."*  
   Cette exigence décrit à la fois une fonction applicative (authentifier) et une règle de sécurité stricte (`SE`). L'étiquette de référence PROMISE l'étiquette comme NFR alors que le modèle perçoit l'action fonctionnelle.
2. **Formulations Passives trompeuses :**  
   *Exemple :* *"All information must be viewable on the website."*  
   L'accumulation de la modalité `must` et de la forme passive induit en erreur le modèle qui l'interprète comme une contrainte d'exploitation (`O`) au lieu d'une fonction d'affichage (`F`).

---

## 7. Déploiement & Intégration Logicielle

Le modèle SetFit a été intégré de manière pérenne et sécurisée dans l'architecture logicielle :

1. **Classe Dédiée `SetFitClassifier` :**  
   Implémentée dans [src/reqmind/classification.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/src/reqmind/classification.py). Elle fournit des inférences batch, des scores de confiance probabilistes et s'interface directement avec `RequirementRecord`.
2. **Stratégie de Fallback Automatique :**  
   Si les poids du modèle SetFit ne sont pas disponibles sur un serveur léger, la fonction `load_default_classifier()` bascule automatiquement vers le modèle TF-IDF, puis vers l'heuristique sans interrompre l'application.
3. **Application Web Streamlit :**  
   Intégration directe dans [app/main.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/app/main.py). L'utilisateur peut coller un cahier des charges et obtenir la classification instantanée avec le badge de confiance et le nom du modèle actif.
4. **Validation Pytest Complète :**  
   Tous les tests unitaires passent à 100% :
   ```text
   tests/test_classifier.py::test_trained_classifier_predicts_valid_labels_and_scores PASSED
   tests/test_classifier.py::test_trained_classifier_is_deterministic PASSED
   tests/test_classifier.py::test_trained_classifier_empty_input PASSED
   tests/test_classifier.py::test_pipeline_records_trained_model_version PASSED
   tests/test_classifier.py::test_fallback_to_heuristic_when_artifact_missing PASSED
   tests/test_classifier.py::test_setfit_classifier_if_available PASSED
   ============================== 6 passed in 1.82s ===============================
   ```

---

## 8. Documents et Fichiers Produits en Phase 3

* **Notebook Interactif & Rendu Complet :** [models/ReqMind_AI_Model_Benchmark_and_Evaluation.ipynb](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/models/ReqMind_AI_Model_Benchmark_and_Evaluation.ipynb)
* **Scripts d'Entraînement & Évaluation :** [experiments/train_classifier.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/experiments/train_classifier.py), [experiments/train_setfit.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/experiments/train_setfit.py), [experiments/evaluate_classifier.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/experiments/evaluate_classifier.py)
* **Rapport Formel de Phase :** [reports/phases/phase_3.md](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/phases/phase_3.md)
* **Logs des Résultats :** [reports/results/phase3_test.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/results/phase3_test.json), [reports/results/phase3_cross_project.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/results/phase3_cross_project.json), [reports/results/phase3_setfit_results.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/results/phase3_setfit_results.json)
* **Document PDF de Synthèse :** [Livrables/ReqMind_AI_Synthese_Modeles_Phase1_a_Phase3.pdf](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Livrables/ReqMind_AI_Synthese_Modeles_Phase1_a_Phase3.pdf)
