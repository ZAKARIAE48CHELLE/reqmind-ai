import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# Colors
C_BLACK = RGBColor(0x00, 0x00, 0x00)
C_GREY_DARK = RGBColor(0x59, 0x59, 0x59)
C_GREY_MID = RGBColor(0x7F, 0x7F, 0x7F)
C_GREY_LIGHT = RGBColor(0x99, 0x99, 0x99)

HEX_BLACK = "000000"
HEX_GREY_DARK = "595959"
HEX_GREY_MID = "7F7F7F"
HEX_GREY_LIGHT = "999999"
HEX_BG_GREY = "D9D9D9"
HEX_BG_LIGHT = "F2F2F2"

FONT_MAIN = "Calibri"
FONT_CODE = "Consolas"

def create_document():
    doc = docx.Document()
    
    # Page setup: A4, 2 cm margins (0.787 in)
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.787)
    section.bottom_margin = Inches(0.787)
    section.left_margin = Inches(0.787)
    section.right_margin = Inches(0.787)
    
    # Setup footer
    setup_footer(section, "Traitement Automatique du Langage Naturel (TALN)")
    
    # Title & Identity Block
    add_title_block(doc, 
                    module="Traitement Automatique du Langage Naturel (TALN)",
                    title="ReqMind AI — Analyseur Intelligent d'Exigences Logicielles",
                    subtitle="Guide Pédagogique Intégral : Architecture, Modèles, Heuristiques & Résultats")
    
    # Intro
    add_intro(doc, "Ce document constitue la synthèse pédagogique complète du projet ReqMind AI. Conçu pour expliciter chaque détail conceptuel, mathématique et logiciel, il détaille l'ingénierie des exigences par le TALN, la comparaison empirique des trois classifieurs (SVM baseline, SVM optimisé calibré et DeBERTa-v3), les algorithmes d'analyse qualité (IEEE 830), la détection de quasi-doublons sémantiques avec gardes de sécurité, la gouvernance Human-in-the-Loop et les résultats mesurés.")
    
    # -------------------------------------------------------------
    # PARTIE 1 : CONTEXTE INDUSTRIEL & PROBLÉMATIQUE DU PROJET
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 1 — Contexte Industriel & Problématique de l'Ingénierie des Exigences")
    
    add_paragraph(doc, "L'Ingénierie des Exigences (Requirements Engineering - RE) constitue la pierre angulaire de tout cycle de vie logiciel. Selon les études empiriques de l'IEEE et du Standish Group, plus de 70% des défaillances critiques et des dépassements budgétaires en production trouvent leur origine dans des spécifications d'exigences (SRS) incomplètes, ambiguës, mal classifiées ou redondantes.")
    
    add_section_heading(doc, "1.1  La distinction fondamentale : Exigences Fonctionnelles (FR) vs Non-Fonctionnelles (NFR)")
    add_paragraph(doc, "En ingénierie logicielle, une spécification d'exigences regroupe deux types distincts de besoins :")
    add_bullet(doc, "Exigences Fonctionnelles (FR - Functional Requirements) : Elles décrivent les actions, comportements, services et transformations de données que le logiciel doit accomplir en réponse à des entrées spécifiques (ex. : « Le système doit permettre aux clients d'ajouter un article au panier »).")
    add_bullet(doc, "Exigences Non-Fonctionnelles (NFR - Non-Functional Requirements) : Elles définissent les critères de qualité, contraintes opérationnelles, seuils de performance, exigences de sécurité et de conformité réglementaire régissant le comportement global du système (ex. : « Le temps de réponse de l'API de paiement ne doit pas dépasser 500 ms à pleine charge »).")
    
    add_tip(doc, "Dans les projets agiles et les cahiers des charges industriels, les ingénieurs mélangent fréquemment les préoccupations fonctionnelles et qualitatives dans une même phrase. Ce flou nuit à la conception architecturale : un développeur concevra une simple fonctionnalité là où un architecte réseau doit déployer une infrastructure à haute disponibilité.", "Pourquoi cette distinction est critique")
    
    add_section_heading(doc, "1.2  Les trois fléaux des spécifications textuelles")
    add_paragraph(doc, "Rédigées en langage naturel libre, les exigences souffrent de trois faiblesses endémiques :")
    add_numbered_item(doc, 1, "La mauvaise classification taxonomique : Des contraintes vitales de sécurité ou de latence sont traitées comme de simples tickets de fonctionnalités et sous-estimées dans les sprints.")
    add_numbered_item(doc, 2, "L'ambiguïté lexicale et syntaxique (Norme IEEE 830) : Présence d'adverbes vagues (« rapidement », « efficacement ») et de modaux non-contraignants (« devrait », « pourrait ») rendant l'exigence impossible à vérifier par un test automatique d'acceptation.")
    add_numbered_item(doc, 3, "La redondance et les quasi-doublons contradictoires : Dans les projets complexes impliquant de multiples équipes, des exigences similaires sont rédigées avec des formulations divergentes ou des valeurs d'accords de niveau de service (SLA) incompatibles.")
    
    add_section_heading(doc, "1.3  Les principes non-négociables de l'architecture ReqMind AI")
    add_paragraph(doc, "Pour apporter une réponse rigoureuse à ces défis, ReqMind AI a été conçu selon quatre impératifs :")
    add_bullet(doc, "Traitement 100% Local et Souverain : Aucune API commerciale tierce payante (OpenAI, Anthropic). Tous les modèles (SVM, SentenceTransformers, DeBERTa-v3) tournent localement sur CPU.")
    add_bullet(doc, "Préservation Absolue de la Provenance : Tout texte source est immuable. Les offsets exacts (caractères de début et de fin) sont enregistrés pour chaque exigence et chaque span ambigu.")
    add_bullet(doc, "Interdiction des Opérations Destructives : Le système ne fusionne ni ne supprime jamais automatiquement des exigences redondantes ; il émet des alertes pour l'expert humain.")
    add_bullet(doc, "Gouvernance Human-in-the-Loop (HITL) : Les prédictions machine (predicted_label, model_score) restent archivées intactes, tandis que l'adjudication de l'ingénieur (reviewer_label) prend la décision finale.")

    # -------------------------------------------------------------
    # PARTIE 2 : ARCHITECTURE TECHNIQUE & FLUX DE TRAITEMENT
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 2 — Architecture Technique & Pipeline de Traitement")
    
    add_paragraph(doc, "ReqMind AI repose sur un pipeline séquentiel hautement modulaire encapsulé dans le package Python 'src/reqmind/'. L'orchestrateur central coordonne les étapes d'ingestion, de segmentation, d'analyses NLP en parallèle, de revue humaine et d'export multi-canaux.")
    
    add_section_heading(doc, "2.1  Schéma du flux de données de bout en bout")
    add_code_block(doc, [
        "+---------------------------------------------------------------------------------+",
        "|  Document Source : Texte brut, Fichiers .txt, .md, .csv ou .xlsx                |",
        "+---------------------------------------------------------------------------------+",
        "                                         |",
        "                                         v",
        "+---------------------------------------------------------------------------------+",
        "|  1. Ingestion & Validation UTF-8 (src/reqmind/ingestion.py)                     |",
        "|     - Calcul du SHA-256 du document et vérification d'intégrité                 |",
        "+---------------------------------------------------------------------------------+",
        "                                         |",
        "                                         v",
        "+---------------------------------------------------------------------------------+",
        "|  2. Segmentation & Préservation des Offsets (src/reqmind/segmentation.py)       |",
        "|     - Découpage par puces, défense décimale (ex: 99.9% uptime), indices start/end|",
        "+---------------------------------------------------------------------------------+",
        "                                         |",
        "         +-------------------------------+-------------------------------+",
        "         |                               |                               |",
        "         v                               v                               v",
        "+-------------------+           +-------------------+           +-------------------+",
        "| 3. Classification |           | 4. Détection      |           | 5. Similarité     |",
        "|    FR vs NFR      |           |    Ambiguïtés     |           |    Sémantique     |",
        "| (LinearSVC,       |           |    (IEEE 830)     |           | (all-MiniLM-L6-v2 |",
        "|  Calibrated SVM,  |           | - Adverbes flous  |           |  + Garde Négation |",
        "|  DeBERTa-v3)      |           | - Modaux faibles  |           |  + Garde SLA)     |",
        "+-------------------+           +-------------------+           +-------------------+",
        "         |                               |                               |",
        "         +-------------------------------+-------------------------------+",
        "                                         |",
        "                                         v",
        "+---------------------------------------------------------------------------------+",
        "|  6. Établissement de la Fiche d'Audit & Adjudication (src/reqmind/review.py)    |",
        "|     - Validation humaine, bascule de statut (PENDING -> APPROVED/EDITED/REJECTED)|",
        "+---------------------------------------------------------------------------------+",
        "                                         |",
        "         +-------------------------------+-------------------------------+",
        "         |                               |                               |",
        "         v                               v                               v",
        "+-------------------+           +-------------------+           +-------------------+",
        "|  Export JSON      |           |  Export CSV / PDF |           |  Onglet Évaluation|",
        "|  (Schéma complet  |           |  (Feuille d'audit |           |  (Matrices Altair,|",
        "|   sans perte)     |           |   & rapport PDF)  |           |   Benchmark Test) |",
        "+-------------------+           +-------------------+           +-------------------+",
    ])
    
    add_section_heading(doc, "2.2  La segmentation à défense décimale et conservation des offsets")
    add_paragraph(doc, "Une erreur classique en TALN consiste à utiliser une segmentation naïve par expression régulière sur le point final ('. '). Dans un document d'exigences logicielles, cela produit des coupures catastrophiques :")
    add_bullet(doc, "L'exigence « The service shall maintain 99.9% availability during business hours. » serait scindée en deux segments incohérents à cause du point décimal.")
    add_bullet(doc, "Les numérotations hiérarchiques (« 1.1.2 The system shall... ») perdent leur identifiant de référence.")
    
    add_paragraph(doc, "Le moteur 'src/reqmind/segmentation.py' met en œuvre un automate à motifs protégés :")
    add_numbered_item(doc, 1, "Neutralisation des décimales et acronymes : Les motifs numériques du type '\\d+\\.\\d+' sont identifiés et protégés contre toute coupure.")
    add_numbered_item(doc, 2, "Détection des délimiteurs de structure : Détection robuste des puces Markdown ('*', '-'), des listes numérotées ('1.', '1.1.') et des retours chariot multiples.")
    add_numbered_item(doc, 3, "Calcul strict des offsets : Pour chaque exigence segmentée, la fonction calcule 'start_char' et 'end_char' dans le texte brut d'origine. Ainsi, 'raw_document[start_char:end_char] == requirement_text' est garanti à 100%.")

    add_image_with_caption(doc, "reports/figures/03_longueurs.png", "Figure 1 — Distribution des longueurs de textes des exigences en nombre de mots.")

    add_tip(doc, "La Figure 1 met en évidence que la majorité des exigences industrielles comportent entre 10 et 25 mots. C'est une plage idéale pour les transformers légers comme MiniLM et DeBERTa-v3, car le contexte sémantique complet tient dans une seule fenêtre de self-attention sans troncature.", "Observation sur la granularité textuelle")

    # -------------------------------------------------------------
    # PARTIE 3 : CLASSIFICATION FR vs NFR — ÉTUDE COMPARATIVE
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 3 — Classification FR vs NFR : Étude Approfondie des Modèles")
    
    add_paragraph(doc, "ReqMind AI intègre et compare trois architectures de classification complémentaires, allant du modèle statistique ultra-rapide au modèle de fondation de pointe adapté au domaine des exigences.")
    
    add_section_heading(doc, "3.1  Description détaillée des trois modèles implémentés")
    
    add_paragraph(doc, "1. Modèle 1 — Baseline TF-IDF + LinearSVC :")
    add_paragraph(doc, "Représentation du texte par matrice TF-IDF de n-grammes de mots (unigrammes et bigrammes) avec pondération sous-linéaire (sublinear_tf=True). Le classifieur est un Support Vector Machine linéaire (LinearSVC, C=1.0) minimisant la perte charnière (Hinge Loss). Sa vitesse d'inférence est exceptionnelle (moins de 0.5 ms par exigence), mais il est aveugle à l'ordre sémantique profond.")
    
    add_paragraph(doc, "2. Modèle 2 — SVM Optimisé & Calibré (FeatureUnion Mots + Caractères) :")
    add_paragraph(doc, "Pour surmonter les limites du vocabulaire hors-dictionnaire et des fautes de frappe, ce modèle exploite une FeatureUnion combinant :")
    add_bullet(doc, "N-grammes de mots (1, 2) capturant les collocations courantes ('shall allow', 'response time').")
    add_bullet(doc, "N-grammes de caractères (3, 5) capturant les racines lexicales et suffixes morphologiques ('encrypt-', 'secur-').")
    add_bullet(doc, "Pondération de classe équilibrée (class_weight='balanced') : Compense le déséquilibre naturel entre les nombreuses FR et les NFR plus rares.")
    add_bullet(doc, "Calibration de probabilités par scaling de Platt (CalibratedClassifierCV) : Les SVM classiques ne fournissent qu'une distance brute à l'hyperplan séparateur. Grâce à une régression logistique sigmoid entraînée par validation croisée à 5 plis, le modèle émet de véritables probabilités interprétables entre 0.0 et 1.0.")
    
    add_paragraph(doc, "3. Modèle 3 — Fine-tuned DeBERTa-v3-small (Transformer avec Attention Désenchevêtrée) :")
    add_paragraph(doc, "DeBERTa (Decoding-enhanced BERT with Disentangled Attention) représente l'état de l'art des encodeurs contextuels :")
    add_bullet(doc, "Attention désenchevêtrée (Disentangled Attention) : Contrairement à BERT qui additionne le vecteur de mot et son vecteur de position, DeBERTa utilise deux vecteurs séparés pour représenter le contenu et la position relative. L'attention entre deux mots est décomposée en quatre matrices croisées (contenu-contenu, contenu-position, position-contenu, position-position).")
    add_bullet(doc, "Fine-tuning supervisé : Entraîné avec l'optimiseur AdamW, un taux d'apprentissage avec warmup linéaire et une fonction de coût Cross-Entropy régularisée par Label Smoothing (0.1) pour éviter le sur-ajustement.")

    add_section_heading(doc, "3.2  Résultats expérimentaux sur le Benchmark Humain Annoté (60 Exemples)")
    add_paragraph(doc, "Le Benchmark Humain Annoté ('ReqMind_AI_Annotated_FR_NFR_Test_Set.xlsx') a été spécifiquement conçu pour évaluer la capacité des modèles à résoudre des cas complexes, des pièges syntaxiques et des exigences mixtes. Il est strictement équilibré (30 FR et 30 NFR).")
    
    # Table of benchmark results
    headers = ["Métrique d'Évaluation", "Baseline TF-IDF + SVM", "SVM Optimisé & Calibré", "DeBERTa-v3-small"]
    rows = [
        ["Exactitude (Accuracy)", "76.67% (46 / 60)", "80.00% (48 / 60)", "96.67% (58 / 60)"],
        ["Macro-F1", "76.64%", "79.80%", "96.67%"],
        ["Précision FR", "75.00%", "75.00%", "96.67%"],
        ["Rappel FR", "80.00%", "90.00%", "96.67%"],
        ["F1-Score FR", "77.42%", "81.82%", "96.67%"],
        ["Précision NFR", "78.57%", "87.50%", "96.67%"],
        ["Rappel NFR", "73.33%", "70.00%", "96.67%"],
        ["F1-Score NFR", "75.86%", "77.78%", "96.67%"],
        ["Nombre d'Erreurs Global", "14 erreurs", "12 erreurs", "2 erreurs (-85.7%)"],
        ["Latence Moyenne / Exigence (CPU)", "0.46 ms", "0.35 ms", "14.66 ms (~42x)"],
    ]
    add_table(doc, headers, rows)
    
    add_section_heading(doc, "3.3  Matrices de confusion sur le Benchmark de 60 exemples")
    add_paragraph(doc, "Les matrices de confusion mesurées expérimentalement illustrent la répartition exacte des prédictions :")
    
    # Confusion matrix visual representation
    cm_headers = ["Modèle Évalué", "Vrais Positifs (FR)", "Faux Positifs (NFR->FR)", "Faux Négatifs (FR->NFR)", "Vrais Négatifs (NFR)"]
    cm_rows = [
        ["Baseline TF-IDF + SVM", "24", "8", "6", "22"],
        ["SVM Optimisé & Calibré", "27", "9", "3", "21"],
        ["DeBERTa-v3-small", "29", "1", "1", "29"],
    ]
    add_table(doc, cm_headers, cm_rows)

    add_image_with_caption(doc, "reports/figures/optimized_svm_cv_calibration_curve.png", "Figure 2 — Courbe de calibration probabiliste du SVM optimisé après scaling de Platt (Brier score minimal).")
    
    add_image_with_caption(doc, "reports/figures/step3_deberta_training_curves.png", "Figure 3 — Courbes d'apprentissage de DeBERTa-v3-small montrant la convergence de la perte et du Macro-F1.")

    add_section_heading(doc, "3.4  Analyse scientifique des erreurs : Pourquoi DeBERTa surpasse le SVM")
    add_paragraph(doc, "L'écart de performance de +20 points d'exactitude (96.67% contre 76.67%) entre DeBERTa-v3 et la baseline SVM s'explique par la nature linguistique des exigences :")
    
    add_numbered_item(doc, 1, "Le piège des verbes d'action dans les exigences de sécurité :")
    add_paragraph(doc, "Considérons l'exigence suivante : « The system shall encrypt all sensitive customer data before writing to the database. »")
    add_bullet(doc, "Prédiction SVM : Prédit 'FR' à tort. Le sac de mots repère des verbes et substantifs d'action ('shall', 'write', 'database') typiques des fonctionnalités utilisateur.")
    add_bullet(doc, "Prédiction DeBERTa : Prédit 'NFR' avec 96% de confiance. L'attention contextuelle comprend que 'encrypt' s'applique comme une contrainte de sécurité sur une opération interne et non comme un service offert directement à l'utilisateur.")
    
    add_numbered_item(doc, 2, "La gestion des structures conditionnelles et de tolérance aux pannes :")
    add_paragraph(doc, "Sur l'exigence : « The system shall gracefully degrade services if third-party payment gateways fail to respond within 3 seconds. »")
    add_bullet(doc, "Le SVM échoue en raison de la présence de termes fonctionnels ('payment gateways', 'services').")
    add_bullet(doc, "DeBERTa identifie la clause de tolérance de panne ('gracefully degrade', 'fail to respond') comme une contrainte de fiabilité (NFR).")

    add_tip(doc, "Les deux seules erreurs commises par DeBERTa portent sur des cas limites où même deux experts humains hésitent : la génération automatique de journaux d'audit de sécurité (génération = fonction, audit = sécurité réglementaire). Ce constat justifie l'obligation du module Human-in-the-Loop.", "Analyse de la frontière de décision")

    # -------------------------------------------------------------
    # PARTIE 4 : DÉTECTION D'AMBIGUÏTÉS SELON LA NORME IEEE 830
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 4 — Détection d'Ambiguïtés & Défauts Qualité selon IEEE 830")
    
    add_paragraph(doc, "La norme internationale IEEE 830 (Recommended Practice for Software Requirements Specifications) stipule qu'une exigence logicielle de qualité doit être non-ambiguë, vérifiable, mesurable et non-triviale. ReqMind AI intègre un moteur déterministe d'analyse statique lexicale ('src/reqmind/ambiguity.py').")
    
    add_section_heading(doc, "4.1  Taxonomie des défauts détectés")
    add_paragraph(doc, "Le moteur classe les défauts en quatre catégories majeures :")
    
    defect_headers = ["Catégorie de Défaut", "Mots & Motifs Détectés", "Risque Industriel / Conséquence", "Recommandation de Remédiation"]
    defect_rows = [
        ["Adverbes Vagues / Flous", "approximately, quickly, efficiently, regularly, promptly, easily", "Impossible d'écrire un test d'acceptation automatique (Pass/Fail).", "Remplacer par un SLA mesurable (ex : '< 200 ms au 95e percentile')."],
        ["Revendications Subjectives", "user-friendly, intuitive, flexible, state-of-the-art, optimal", "Appréciation purement subjective dépendant de l'utilisateur final.", "Quantifier par une métrique de test utilisateur (ex : 'SUS > 80')."],
        ["Modaux Non-Contraignants", "should, could, may, might, ideally", "Viole la RFC 2119 : l'équipe de dev peut omettre l'exigence sans faute.", "Employer exclusivement les modaux stricts : 'shall' ou 'must'."],
        ["Proportionnalité Faible", "as appropriate, if feasible, to the extent possible, as needed", "Fournit une échappatoire juridique et technique à l'exécutant.", "Spécifier les conditions logiques strictes et exhaustives."],
    ]
    add_table(doc, defect_headers, defect_rows)

    add_section_heading(doc, "4.2  Algorithme de calcul des spans exacts et rejet des faux-positifs")
    add_paragraph(doc, "Pour chaque mot ou motif détecté :")
    add_numbered_item(doc, 1, "Le moteur calcule l'offset relatif de début (start_char) et de fin (end_char) au sein de la chaîne de l'exigence.")
    add_numbered_item(doc, 2, "Dans l'interface Streamlit (Onglet 3), les spans sont affichés sous forme de badges ambrés en surbrillance au-dessus du texte.")
    add_numbered_item(doc, 3, "Mécanisme de Rejet Humain (Dismiss) : Si un terme technique légitime est détecté à tort (ex : 'fast Fourier transform' ou 'high-speed bus'), l'ingénieur clique sur 'Rejeter l'alerte'. Le span est marqué 'dismissed = True' dans les métadonnées sans altérer le texte source.")

    # -------------------------------------------------------------
    # PARTIE 5 : DÉTECTION SÉMANTIQUE DE DOUBLONS & GARDES DE SÉCURITÉ
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 5 — Détection Sémantique de Doublons & Gardes de Sécurité")
    
    add_paragraph(doc, "Dans les spécifications industrielles volumineuses (plusieurs centaines de pages), la détection de redondances textuelles ne peut pas reposer sur une simple égalité de chaînes ou une distance de Levenshtein, car les rédacteurs emploient des synonymes et des tournures variées.")
    
    add_section_heading(doc, "5.1  Architecture Bi-Encodeur Dense ('all-MiniLM-L6-v2')")
    add_paragraph(doc, "ReqMind AI utilise le modèle de plongement sémantique 'sentence-transformers/all-MiniLM-L6-v2' :")
    add_bullet(doc, "Transformation de chaque exigence en un vecteur dense de 384 dimensions capturant le sens sémantique global.")
    add_bullet(doc, "Calcul matriciel vectorisé de la similarité cosinus entre toutes les paires d'exigences.")
    add_bullet(doc, "Filtrage par seuil de confiance configurable (seuil par défaut = 0.85) et extraction des paires candidates classées par score décroissant.")

    add_section_heading(doc, "5.2  Les deux Gardes de Sécurité essentielles")
    add_paragraph(doc, "La similarité cosinus pure présente un danger majeur en ingénierie : deux phrases presque identiques mais de sens opposé ont un score sémantique très élevé. ReqMind AI innove en introduisant deux gardes heuristiques défensives :")
    
    add_paragraph(doc, "1. La Garde de Négation (Negation Guard) :")
    add_paragraph(doc, "Analyse la polarité grammaticale des verbes ('shall' vs 'shall not', 'allow' vs 'prevent', 'never'). Si deux exigences partagent 90% de vocabulaire mais que l'une contient une négation, le système NE LES MARQUE PAS comme un doublon, mais lève une ALERTE DE CONFLIT POLAIRE.")
    
    add_paragraph(doc, "2. La Garde de Conflit Numérique / SLA (SLA Conflict Guard) :")
    add_paragraph(doc, "Exemple réel détecté par le système :")
    add_bullet(doc, "Exigence A : « The database backup shall complete within 2 hours. »")
    add_bullet(doc, "Exigence B : « The database backup shall complete within 6 hours. »")
    add_paragraph(doc, "La similarité cosinus brute entre A et B est de 0.94. Un algorithme naïf fusionnerait ces deux exigences. La Garde SLA de ReqMind AI extrait les valeurs numériques associées aux métriques temporelles et alerte immédiatement l'architecte : il ne s'agit pas d'un doublon anodin, mais d'une contradiction de spécification.")

    # -------------------------------------------------------------
    # PARTIE 6 : GOUVERNANCE HUMAN-IN-THE-LOOP & MACHINE À ÉTATS
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 6 — Gouvernance Human-in-the-Loop & Machine à États")
    
    add_paragraph(doc, "ReqMind AI respecte scrupuleusement le principe d'assistance à la décision : l'intelligence artificielle propose, l'ingénieur dispose. Le moteur 'src/reqmind/review.py' gère le cycle de vie de révision.")
    
    add_section_heading(doc, "6.1  La machine à états finis du statut d'audit")
    add_code_block(doc, [
        "                   +------------------------+",
        "                   |        PENDING         | (Statut initial après analyse)",
        "                   +------------------------+",
        "                               |",
        "        +----------------------+----------------------+",
        "        |                      |                      |",
        "        v                      v                      v",
        "+---------------+      +---------------+      +---------------+",
        "|   APPROVED    |      |    EDITED     |      |   REJECTED    |",
        "| (Label validé |      | (Label altéré |      | (Exigence non |",
        "|  par l'expert)|      |  par l'expert)|      |  pertinente)  |",
        "+---------------+      +---------------+      +---------------+",
    ])
    
    add_section_heading(doc, "6.2  Préservation de la traçabilité et non-mutation")
    add_paragraph(doc, "Dans la structure Pydantic 'RequirementRecord' :")
    add_bullet(doc, "'predicted_label' : Contient la prédiction d'origine du modèle (ex : 'FR'). Ce champ est strictement en lecture seule.")
    add_bullet(doc, "'model_score' : Contient la probabilité calibrée ou la confiance du modèle (ex : 0.88).")
    add_bullet(doc, "'reviewer_label' : Contient la décision souveraine de l'expert (ex : 'NFR').")
    add_bullet(doc, "'review_notes' : Consigne la justification technique de l'adjudicateur pour les audits futurs.")

    # -------------------------------------------------------------
    # PARTIE 7 : MODULE D'ÉVALUATION EMPIRIQUE (ONGLET 8)
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 7 — Module d'Évaluation Empirique & Analyse des Désaccords")
    
    add_paragraph(doc, "Développé pour garantir une validation académique irréprochable, l'Onglet 8 de l'application Streamlit ('app/components/evaluation_view.py' et 'src/reqmind/evaluation.py') offre un laboratoire d'évaluation comparative en temps réel.")
    
    add_section_heading(doc, "7.1  Validation stricte des jeux de données de test")
    add_paragraph(doc, "Lorsqu'un utilisateur charge un fichier Excel ou CSV de test, le validateur vérifie :")
    add_bullet(doc, "La présence obligatoire des colonnes 'Requirement_ID', 'Requirement_Text', 'Gold_Label'.")
    add_bullet(doc, "L'absence d'identifiants dupliqués et de textes vides.")
    add_bullet(doc, "L'appartenance stricte des étiquettes attendues à l'ensemble légal {'FR', 'NFR'}.")
    
    add_section_heading(doc, "7.2  Comparaison multi-modèles côte à côte et inspecteur de désaccords")
    add_paragraph(doc, "L'onglet permet de sélectionner simultanément deux ou trois modèles pour les confronter au même jeu de données étiqueté :")
    add_bullet(doc, "Génération dynamique de graphiques à barres comparatifs sous Altair (Macro-F1, Exactitude, Latence).")
    add_bullet(doc, "Grille de matrices de confusion côte à côte facilitant la comparaison visuelle.")
    add_bullet(doc, "Inspecteur de Désaccords (Disagreement Inspector) : Isole dans une table dédiée toutes les exigences où au moins deux modèles ont divergé. Cela permet à l'ingénieur de comprendre immédiatement les cas d'ambiguïté linguistique.")

    # -------------------------------------------------------------
    # PARTIE 8 : EXPORT MULTI-CANAUX TRAÇABLE & COMPTE-RENDU PDF
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 8 — Architecture d'Export Multi-Canaux & Traçabilité")
    
    add_paragraph(doc, "Pour s'intégrer harmonieusement dans les chaînes d'outils industrielles (Jira, DOORS, GitLab), ReqMind AI fournit quatre formats d'exportation complémentaires :")
    
    add_section_heading(doc, "8.1  Les quatre canaux d'exportation")
    add_numbered_item(doc, 1, "Export JSON Intégral : Sérialisation complète du graphe d'objets Pydantic. Conserve la provenance du document, les offsets exacts de chaque exigence et de chaque span ambigu, les scores et l'historique complet d'audit. Intègre un validateur de désérialisation aller-retour (round-trip test).")
    add_numbered_item(doc, 2, "Export CSV Tabulaire : Tableau à plat contenant les identifiants, les textes, les labels d'origine et le label résolu final (donnant priorité à l'expert humain), prêt pour un import dans Excel ou Jira.")
    add_numbered_item(doc, 3, "Export Markdown Exécutif : Synthèse lisible récapitulant la répartition globale des exigences, le taux de révision et la liste des doublons potentiels.")
    add_numbered_item(doc, 4, "Export PDF Exécutif via ReportLab : Document PDF généré dynamiquement avec canvas à deux passes (_NumberedCanvas) assurant la numérotation automatique (« Page X sur Y »), des cartes de synthèse KPI en en-tête et un tableau d'audit structuré.")

    # -------------------------------------------------------------
    # PARTIE 9 : GUIDE DE DÉMONSTRATION EN 5 MINUTES
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 9 — Guide de Démonstration pour Soutenance (5 Minutes)")
    
    add_paragraph(doc, "Ce scénario pas-à-pas a été éprouvé pour présenter l'ensemble des capacités de ReqMind AI lors d'une soutenance académique :")
    
    add_numbered_item(doc, 1, "Minute 1 — Ingestion & Segmentation (Onglet 1) : Cliquer sur le scénario prédéfini 'E-Commerce Platform SRS'. Lancer l'analyse. Montrer dans l'Onglet 2 que les exigences sont découpées proprement sans casser les nombres décimaux (ex: '99.9% uptime').")
    add_numbered_item(doc, 2, "Minute 2 — Inspection de la Qualité IEEE 830 (Onglet 3) : Pointer les badges ambrés en surbrillance sur les termes 'quickly' et 'user-friendly'. Montrer l'explication méthodologique. Cliquer sur 'Rejeter l'alerte' pour démontrer le rejet de faux positif.")
    add_numbered_item(doc, 3, "Minute 3 — Détection de Doublons & Gardes SLA (Onglet 4) : Afficher les paires d'exigences avec similarité cosinus > 0.85. Mettre en valeur l'alerte de conflit SLA entre deux délais d'expiration divergents.")
    add_numbered_item(doc, 4, "Minute 4 — Revue Humaine & Export PDF (Onglets 5 & 6) : Modifier le label d'une exigence ambiguë (FR -> NFR) dans le plan de travail. Télécharger le rapport PDF généré avec la numérotation dynamique et les KPI d'audit.")
    add_numbered_item(doc, 5, "Minute 5 — Laboratoire d'Évaluation (Onglet 8) : Charger le test set de 60 exemples. Lancer la comparaison simultanée SVM vs DeBERTa. Commenter le bond de +20 points d'exactitude (76.7% -> 96.7%) et commenter la matrice de confusion interactive.")

    # -------------------------------------------------------------
    # PARTIE 10 : SYNTHÈSE DES ENSEIGNEMENTS, LIMITES & PERSPECTIVES
    # -------------------------------------------------------------
    add_part_heading(doc, "Partie 10 — Synthèse des Enseignements, Limites & Perspectives")
    
    add_paragraph(doc, "Le projet ReqMind AI démontre avec succès la viabilité d'un système d'aide à la décision en TALN 100% local pour le génie logiciel. Les enseignements clés sont les suivants :")
    
    add_bullet(doc, "Supériorité confirmée des Transformers pour la sémantique complexe : DeBERTa-v3 surmonte le piège des verbes d'action là où les modèles statistiques restent aveugles à l'architecture de la phrase.")
    add_bullet(doc, "Utilité des modèles légers pour l'interactivité en temps réel : Le SVM calibré (0.35 ms) reste la solution idéale pour l'analyse au fil de la frappe dans un éditeur de texte.")
    add_bullet(doc, "Indispensabilité du Human-in-the-Loop : Le langage naturel contient une part d'ambiguïté irréductible qui exige la supervision finale d'un ingénieur du domaine.")
    
    add_section_heading(doc, "10.1  Limites actuelles du système")
    add_bullet(doc, "Classification binaire simplifiée : Certaines exigences hybrides mériteraient une approche multi-labels (ex: une fonction qui intègre une contrainte de chiffrement).")
    add_bullet(doc, "Dépendance à la langue anglaise : Les règles de qualité IEEE 830 et les jeux d'entraînement actuels sont optimisés pour les spécifications rédigées en anglais.")
    
    add_section_heading(doc, "10.2  Perspectives d'évolution")
    add_bullet(doc, "Classification multi-classes selon ISO/IEC 25010 : Subdiviser la classe NFR en sous-classes spécifiques (Sécurité, Performance, Fiabilité, Maintenabilité, Utilisabilité).")
    add_bullet(doc, "Intégration d'ontologies métier : Utiliser des graphes de connaissances logiciels pour vérifier la cohérence terminologique entre composants.")
    
    # Save document
    output_path = "ReqMind_AI_Synthese_Projet.docx"
    doc.save(output_path)
    print(f"Document saved successfully: {output_path}")

# =============================================================
# HELPER FUNCTIONS ACCORDING TO DOCUMENT-STYLE SPEC
# =============================================================

def setup_footer(section, module_name):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add thin grey top rule to footer paragraph
    pBdr = parse_xml(f'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:top w:val="single" w:sz="4" w:space="6" w:color="{HEX_GREY_LIGHT}"/></w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)
    
    r1 = p.add_run(f"Zakariae CHELLE · 5IASDT G1 · {module_name}          Page ")
    r1.font.name = FONT_MAIN
    r1.font.size = Pt(8)
    r1.font.color.rgb = C_GREY_DARK
    
    # Page field
    add_field(p, "PAGE")
    
    r2 = p.add_run(" / ")
    r2.font.name = FONT_MAIN
    r2.font.size = Pt(8)
    r2.font.color.rgb = C_GREY_DARK
    
    # Total pages field
    add_field(p, "NUMPAGES")

def add_field(paragraph, field_name):
    run = paragraph.add_run()
    fldChar1 = parse_xml(r'<w:fldChar xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fldCharType="begin"/>')
    instrText = parse_xml(f'<w:instrText xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xml:space="preserve"> {field_name} </w:instrText>')
    fldChar2 = parse_xml(r'<w:fldChar xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fldCharType="separate"/>')
    fldChar3 = parse_xml(r'<w:fldChar xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fldCharType="end"/>')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)
    run.font.name = FONT_MAIN
    run.font.size = Pt(8)
    run.font.color.rgb = C_GREY_DARK

def add_title_block(doc, module, title, subtitle):
    # Course line small grey 10 pt
    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_after = Pt(4)
    r_course = p_course.add_run(module)
    r_course.font.name = FONT_MAIN
    r_course.font.size = Pt(10)
    r_course.font.color.rgb = C_GREY_DARK
    
    # Title bold 16 pt
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run(title)
    r_title.font.name = FONT_MAIN
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = C_BLACK
    
    # Subtitle 12 pt
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run(subtitle)
    r_sub.font.name = FONT_MAIN
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = C_BLACK
    
    # Line 1 with thin black top rule
    p_id1 = doc.add_paragraph()
    p_id1.paragraph_format.space_before = Pt(4)
    p_id1.paragraph_format.space_after = Pt(2)
    pBdr_top = parse_xml(f'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:top w:val="single" w:sz="6" w:space="6" w:color="{HEX_BLACK}"/></w:pBdr>')
    p_id1._p.get_or_add_pPr().append(pBdr_top)
    
    r_name_label = p_id1.add_run("Nom et prénom : ")
    r_name_label.font.name = FONT_MAIN
    r_name_label.font.size = Pt(11)
    
    r_name = p_id1.add_run("Zakariae CHELLE")
    r_name.font.name = FONT_MAIN
    r_name.font.size = Pt(11)
    r_name.font.bold = True
    
    # Add tab to right edge
    add_right_tab(p_id1, Inches(6.69))
    
    r_grp_label = p_id1.add_run("\tGroupe : ")
    r_grp_label.font.name = FONT_MAIN
    r_grp_label.font.size = Pt(11)
    
    r_grp = p_id1.add_run("5IASDT G1")
    r_grp.font.name = FONT_MAIN
    r_grp.font.size = Pt(11)
    r_grp.font.bold = True
    
    # Line 2 with thin black bottom rule
    p_id2 = doc.add_paragraph()
    p_id2.paragraph_format.space_before = Pt(2)
    p_id2.paragraph_format.space_after = Pt(14)
    pBdr_bot = parse_xml(f'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:bottom w:val="single" w:sz="6" w:space="6" w:color="{HEX_BLACK}"/></w:pBdr>')
    p_id2._p.get_or_add_pPr().append(pBdr_bot)
    
    r_mod_label = p_id2.add_run("Module : ")
    r_mod_label.font.name = FONT_MAIN
    r_mod_label.font.size = Pt(11)
    
    r_mod = p_id2.add_run(module)
    r_mod.font.name = FONT_MAIN
    r_mod.font.size = Pt(11)
    r_mod.font.bold = True

def add_right_tab(paragraph, position):
    pPr = paragraph._p.get_or_add_pPr()
    tabs = parse_xml(f'<w:tabs xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:tab w:val="right" w:pos="{int(position.pt * 20)}"/></w:tabs>')
    pPr.append(tabs)

def add_intro(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = FONT_MAIN
    run.font.size = Pt(10.5)
    run.font.italic = True
    run.font.color.rgb = C_GREY_DARK

def add_part_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    
    # Black bottom rule
    pBdr = parse_xml(f'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:bottom w:val="single" w:sz="8" w:space="4" w:color="{HEX_BLACK}"/></w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)
    
    run = p.add_run(text)
    run.font.name = FONT_MAIN
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.color.rgb = C_BLACK

def add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    
    # Grey bottom rule
    pBdr = parse_xml(f'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:bottom w:val="single" w:sz="6" w:space="4" w:color="{HEX_GREY_LIGHT}"/></w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)
    
    run = p.add_run(text)
    run.font.name = FONT_MAIN
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = C_BLACK

def add_paragraph(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    format_rich_text(p, text)
    return p

def add_bullet(doc, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    format_rich_text(p, text)
    return p

def add_numbered_item(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    
    r_num = p.add_run(f"{num}.\t")
    r_num.font.name = FONT_MAIN
    r_num.font.size = Pt(11)
    r_num.font.bold = True
    
    format_rich_text(p, text)
    return p

def add_tip(doc, text, label="Pour comprendre"):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    
    # Left grey border
    pBdr = parse_xml(f'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:left w:val="single" w:sz="12" w:space="8" w:color="{HEX_GREY_MID}"/></w:pBdr>')
    p._p.get_or_add_pPr().append(pBdr)
    
    r_label = p.add_run(f"{label} : ")
    r_label.font.name = FONT_MAIN
    r_label.font.size = Pt(10.5)
    r_label.font.bold = True
    r_label.font.italic = True
    r_label.font.color.rgb = C_GREY_DARK
    
    r_text = p.add_run(text)
    r_text.font.name = FONT_MAIN
    r_text.font.size = Pt(10.5)
    r_text.font.italic = True
    r_text.font.color.rgb = C_GREY_DARK

def add_code_block(doc, lines):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.69)
    
    # Thin grey border
    tcBdr = parse_xml(f'<w:tcBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                      f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                      f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                      f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                      f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                      f'</w:tcBdr>')
    cell._tc.get_or_add_tcPr().append(tcBdr)
    
    # Margin inside cell
    tcMar = parse_xml(r'<w:tcMar xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                      r'<w:top w:w="80" w:type="dxa"/>'
                      r'<w:bottom w:w="80" w:type="dxa"/>'
                      r'<w:left w:w="120" w:type="dxa"/>'
                      r'<w:right w:w="120" w:type="dxa"/>'
                      r'</w:tcMar>')
    cell._tc.get_or_add_tcPr().append(tcMar)
    
    # Fill lines
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    
    for i, line in enumerate(lines):
        if i > 0:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.keep_with_next = True
        run = p.add_run(line if line else " ")
        run.font.name = FONT_CODE
        run.font.size = Pt(8.5)
        run.font.color.rgb = C_BLACK

def add_image_with_caption(doc, img_rel_path, caption):
    if not os.path.exists(img_rel_path):
        return
    
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(10)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    
    # Add image scaled to max width 5.2 in
    run_img = p_img.add_run()
    run_img.add_picture(img_rel_path, width=Inches(5.0))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(12)
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = FONT_MAIN
    run_cap.font.size = Pt(9)
    run_cap.font.italic = True
    run_cap.font.color.rgb = C_GREY_DARK

def add_table(doc, headers, data_rows):
    tbl = doc.add_table(rows=len(data_rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    
    # Header row
    hdr_cells = tbl.rows[0].cells
    tbl.rows[0]._tr.get_or_add_trPr().append(parse_xml(r'<w:tblHeader xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
    tbl.rows[0]._tr.get_or_add_trPr().append(parse_xml(r'<w:cantSplit xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
    
    for i, title in enumerate(headers):
        cell = hdr_cells[i]
        # Shading D9D9D9
        shd = parse_xml(f'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:val="clear" w:fill="{HEX_BG_GREY}"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        
        # Border
        tcBdr = parse_xml(f'<w:tcBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                          f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{HEX_BLACK}"/>'
                          f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{HEX_BLACK}"/>'
                          f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                          f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                          f'</w:tcBdr>')
        cell._tc.get_or_add_tcPr().append(tcBdr)
        
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(title)
        run.font.name = FONT_MAIN
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.color.rgb = C_BLACK
    
    # Data rows
    for r_idx, row_data in enumerate(data_rows):
        row = tbl.rows[r_idx + 1]
        row._tr.get_or_add_trPr().append(parse_xml(r'<w:cantSplit xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            tcBdr = parse_xml(f'<w:tcBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                              f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                              f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                              f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                              f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_GREY_LIGHT}"/>'
                              f'</w:tcBdr>')
            cell._tc.get_or_add_tcPr().append(tcBdr)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.05
            
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
            run = p.add_run(str(val))
            run.font.name = FONT_MAIN
            run.font.size = Pt(9)
            run.font.color.rgb = C_BLACK
            
            # Highlight key metric row or bold first column
            if c_idx == 0:
                run.font.bold = True

def format_rich_text(paragraph, text):
    # Splits on inline code `...` and bold **...**
    parts = text.split("`")
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inline code
            r = paragraph.add_run(part)
            r.font.name = FONT_CODE
            r.font.size = Pt(9.5)
            r.font.color.rgb = C_BLACK
        else:
            # Check for bold **
            subparts = part.split("**")
            for j, subpart in enumerate(subparts):
                if not subpart:
                    continue
                r = paragraph.add_run(subpart)
                r.font.name = FONT_MAIN
                r.font.size = Pt(11)
                r.font.color.rgb = C_BLACK
                if j % 2 == 1:
                    r.font.bold = True

if __name__ == "__main__":
    create_document()
