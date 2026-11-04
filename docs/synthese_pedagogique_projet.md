# ReqMind AI — Synthèse Pédagogique Intégrale du Projet

**Module :** Traitement Automatique du Langage Naturel (TALN)  
**Étudiant :** Zakariae CHELLE  
**Groupe :** 5IASDT G1  
**Fichiers Générés :**
- [ReqMind_AI_Synthese_Projet.docx](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/ReqMind_AI_Synthese_Projet.docx) (Document Word conforme au style pédagogique sobre)
- [ReqMind_AI_Synthese_Projet.pdf](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/ReqMind_AI_Synthese_Projet.pdf) (Rapport PDF 5 pages haute fidélité avec numérotation dynamique)

---

## 1. Contexte Industriel & Problématique

L'Ingénierie des Exigences (Requirements Engineering - RE) est la phase déterminante de tout cycle de vie logiciel. Selon l'IEEE, plus de 70% des défaillances logicielles critiques proviennent de spécifications d'exigences (SRS) incomplètes, ambiguës ou mal comprises.

### 1.1. Exigences Fonctionnelles (FR) vs Non-Fonctionnelles (NFR)
- **Exigences Fonctionnelles (FR) :** Décrivent les services, fonctions et calculs que le système doit exécuter (ex. : *« Le système doit permettre aux clients d'ajouter un article au panier »*).
- **Exigences Non-Fonctionnelles (NFR) :** Définissent les qualités, contraintes opérationnelles, seuils de performance, sécurité et conformité réglementaire (ex. : *« Le temps de réponse de l'API de paiement ne doit pas dépasser 500 ms à pleine charge »*).

### 1.2. Les trois fléaux des spécifications textuelles
1. **Mauvaise classification taxonomique :** Traiter des contraintes de sécurité comme de simples fonctionnalités conduit à des erreurs d'architecture.
2. **Ambiguïté lexicale et syntaxique (IEEE 830) :** Termes vagues (*« rapidement »*, *« convivial »*) et modaux faibles (*« devrait »*, *« pourrait »*) impossibles à valider par des tests d'acceptation automatiques.
3. **Redondance et quasi-doublons contradictoires :** Exigences reformulées avec des valeurs de SLA divergentes (ex. : 2 heures vs 6 heures).

---

## 2. Architecture Technique & Pipeline Non-Destructif

```text
+---------------------------------------------------------------------------------+
|  Document Source : Texte brut, Fichiers .txt, .md, .csv ou .xlsx                |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  1. Ingestion & Validation UTF-8 (src/reqmind/ingestion.py)                     |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  2. Segmentation & Défense Décimale (src/reqmind/segmentation.py)               |
|     - Offsets exacts start_char et end_char préservés                           |
+---------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+-------------------+           +-------------------+           +-------------------+
| 3. Classification |           | 4. Ambiguïtés     |           | 5. Similarité     |
|    FR vs NFR      |           |    (IEEE 830)     |           |    Sémantique     |
| (LinearSVC,       |           | - Adverbes flous  |           | (all-MiniLM-L6-v2 |
|  Calibrated SVM,  |           | - Modaux faibles  |           |  + Gardes Sécu)   |
|  DeBERTa-v3)      |           | - Spans exacts    |           |                   |
+-------------------+           +-------------------+           +-------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  6. Revue Human-in-the-Loop & Export Multi-Canaux (JSON, CSV, MD, PDF)          |
+---------------------------------------------------------------------------------+
```

### 2.1. Défense décimale et conservation des offsets
Pour éviter de couper les exigences au niveau des nombres décimaux (ex: *« 99.9% uptime »*), l'automate de segmentation protège les motifs `\d+\.\d+` et les listes hiérarchiques (`1.1`, `1.2`), tout en calculant les indices `start_char` et `end_char` garantissant la formule :
$$\text{raw\_document}[\text{start\_char} : \text{end\_char}] \equiv \text{requirement\_text}$$

---

## 3. Classification FR vs NFR : Comparaison Empirique des 3 Modèles

### 3.1. Les 3 modèles développés
1. **Baseline TF-IDF + LinearSVC :** Unigrammes et bigrammes avec pondération sous-linéaire ($C=1.0$). Inférence ultra-rapide (0.46 ms), mais sensible aux verbes d'action.
2. **SVM Optimisé & Calibré :** `FeatureUnion` combinant n-grammes de mots $(1, 2)$ et n-grammes de caractères $(3, 5)$, pondération équilibrée (`class_weight='balanced'`) et probabilités calibrées via scaling de Platt (`CalibratedClassifierCV`).
3. **Fine-tuned DeBERTa-v3-small :** Transformer avec attention désenchevêtrée (*Disentangled Attention*) séparant le vecteur de contenu du vecteur de position relative.

### 3.2. Résultats mesurés sur le Benchmark Humain Annoté (60 cas)

| Métrique | Baseline TF-IDF + SVM | SVM Optimisé & Calibré | Fine-Tuned DeBERTa-v3-small | Delta (DeBERTa vs Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Exactitude (Accuracy)** | 76.67% (46/60) | 80.00% (48/60) | **96.67% (58/60)** | **+20.00%** |
| **Macro-F1** | 76.64% | 79.80% | **96.67%** | **+20.03%** |
| **Précision FR** | 75.00% | 75.00% | **96.67%** | **+21.67%** |
| **Rappel FR** | 80.00% | 90.00% | **96.67%** | **+16.67%** |
| **F1-Score FR** | 77.42% | 81.82% | **96.67%** | **+19.25%** |
| **Précision NFR** | 78.57% | 87.50% | **96.67%** | **+18.10%** |
| **Rappel NFR** | 73.33% | 70.00% | **96.67%** | **+23.34%** |
| **F1-Score NFR** | 75.86% | 77.78% | **96.67%** | **+20.81%** |
| **Nombre d'Erreurs** | 14 erreurs | 12 erreurs | **2 erreurs** | **-85.7%** |
| **Latence CPU / Exigence** | **0.46 ms** | **0.35 ms** | 14.66 ms | +14.31 ms (~42×) |

### 3.3. Analyse scientifique des erreurs
- **Le piège des verbes d'action :** Sur l'exigence *« The system shall encrypt all sensitive customer data before writing to the database »*, le SVM prédit **FR** car il surpondère les mots *shall*, *write*, *database*. DeBERTa comprend que *encrypt* est une contrainte de sécurité opérationnelle et prédit **NFR** à 96% de confiance.
- **Compromis d'ingénierie :** Le SVM calibré (0.35 ms, 6.8 Mo) est parfait pour l'analyse interactive au fil de la frappe ; DeBERTa (14.66 ms, 560 Mo, 96.7% d'exactitude) est idéal pour les audits d'homologation et de conformité.

---

## 4. Détection d'Ambiguïtés IEEE 830 & Détection Sémantique de Doublons

1. **Règles Qualité IEEE 830 :**
   - *Adverbes vagues :* `quickly`, `efficiently`, `regularly`.
   - *Revendications subjectives :* `user-friendly`, `intuitive`, `flexible`.
   - *Modaux non-contraignants :* `should`, `could`, `may` au lieu de `shall`/`must`.
   - *Proportionnalité faible :* `as appropriate`, `if feasible`.
   - *Rejet de faux-positifs :* L'ingénieur peut rejeter une alerte légitime (*« fast Fourier transform »*) sans modifier le texte source.
2. **Détection Sémantique avec Gardes :**
   - Embeddings denses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) et similarité cosinus.
   - **Garde de Négation :** Détecte les polarités inversées entre deux phrases similaires.
   - **Garde de Conflit SLA / Numérique :** Empêche d'amalgamer des exigences aux seuils temporels contradictoires (ex. : 2 heures vs 6 heures).

---

## 5. Gouvernance Human-in-the-Loop & Export

- **Provenance Inviolable :** `predicted_label` et `model_score` sont immuables. L'expert enregistre sa décision dans `reviewer_label`.
- **Statut de Révision :** `PENDING` $\to$ `APPROVED` | `EDITED` | `REJECTED`.
- **Quatre canaux d'exportation :**
  1. JSON intégral (avec validateur Pydantic aller-retour).
  2. CSV tabulaire avec labels résolus pour Jira/DOORS.
  3. Synthèse Markdown exécutive.
  4. PDF d'audit ReportLab avec numérotation dynamique *Page X / Y* et indicateurs KPI.
