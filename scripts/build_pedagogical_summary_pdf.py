import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

# Color constants (Strict B&W / Pedagogical Greys)
C_BLACK = colors.HexColor("#000000")
C_GREY_DARK = colors.HexColor("#595959")
C_GREY_MID = colors.HexColor("#7F7F7F")
C_GREY_LIGHT = colors.HexColor("#999999")
C_BG_GREY = colors.HexColor("#D9D9D9")
C_BG_LIGHT = colors.HexColor("#F8F8F8")

MODULE_NAME = "Traitement Automatique du Langage Naturel (TALN)"

class PedagogicalNumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page counting and B&W running footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(C_GREY_DARK)
        
        # A4 width = 595.27 pt, height = 841.89 pt
        # Margin = 20 mm ~ 56.7 pt
        left_margin = 20 * mm
        right_margin = 595.27 - (20 * mm)
        bottom_pos = 15 * mm
        
        # Thin grey top rule for footer
        self.setStrokeColor(C_GREY_LIGHT)
        self.setLineWidth(0.5)
        self.line(left_margin, bottom_pos + 12, right_margin, bottom_pos + 12)
        
        footer_text = f"Zakariae CHELLE · 5IASDT G1 · {MODULE_NAME}          Page {self._pageNumber} / {page_count}"
        self.drawCentredString(595.27 / 2.0, bottom_pos, footer_text)
        self.restoreState()


def build_pdf():
    output_filename = "ReqMind_AI_Synthese_Projet.pdf"
    
    # 2 cm margins on A4
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    
    usable_width = 595.27 - (40 * mm)  # ~481.89 pt (~17 cm)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    style_course = ParagraphStyle(
        "CourseHeader",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=C_GREY_DARK,
        alignment=1, # Centred
        spaceAfter=4,
    )
    
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=C_BLACK,
        alignment=1,
        spaceAfter=4,
    )
    
    style_sub = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11.5,
        leading=15,
        textColor=C_BLACK,
        alignment=1,
        spaceAfter=12,
    )
    
    style_intro = ParagraphStyle(
        "IntroParagraph",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        textColor=C_GREY_DARK,
        spaceBefore=6,
        spaceAfter=12,
    )
    
    style_p = ParagraphStyle(
        "BodyParagraph",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=C_BLACK,
        spaceBefore=3,
        spaceAfter=5,
    )
    
    style_bullet = ParagraphStyle(
        "BulletItem",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=C_BLACK,
        leftIndent=14,
        firstLineIndent=-10,
        spaceBefore=2,
        spaceAfter=3,
    )
    
    style_h1 = ParagraphStyle(
        "PartHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=17,
        textColor=C_BLACK,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )
    
    style_h2 = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=C_BLACK,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    
    style_caption = ParagraphStyle(
        "ImgCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=C_GREY_DARK,
        alignment=1,
        spaceBefore=4,
        spaceAfter=10,
    )
    
    style_code = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=C_BLACK,
    )
    
    story = []
    
    # 1. Title Block
    story.append(Paragraph(MODULE_NAME, style_course))
    story.append(Paragraph("ReqMind AI — Analyseur Intelligent d'Exigences Logicielles", style_title))
    story.append(Paragraph("Guide Pédagogique Intégral : Architecture, Modèles, Heuristiques & Résultats", style_sub))
    
    # Identity Block
    id_data = [
        [
            Paragraph("Nom et prénom : <b>Zakariae CHELLE</b>", style_p),
            Paragraph("Groupe : <b>5IASDT G1</b>", ParagraphStyle("RightP", parent=style_p, alignment=2))
        ],
        [
            Paragraph(f"Module : <b>{MODULE_NAME}</b>", style_p),
            ""
        ]
    ]
    id_table = Table(id_data, colWidths=[usable_width * 0.65, usable_width * 0.35])
    id_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.75, C_BLACK),
        ('LINEBELOW', (0, 1), (-1, 1), 0.75, C_BLACK),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('SPAN', (0, 1), (1, 1)),
    ]))
    story.append(id_table)
    story.append(Spacer(1, 10))
    
    # Intro
    intro_txt = ("Ce document constitue la synthèse pédagogique complète du projet ReqMind AI. "
                 "Conçu pour expliciter chaque détail conceptuel, mathématique et logiciel, il détaille l'ingénierie "
                 "des exigences par le TALN, la comparaison empirique des trois classifieurs (SVM baseline, SVM optimisé "
                 "calibré et DeBERTa-v3), les algorithmes d'analyse qualité (IEEE 830), la détection de quasi-doublons "
                 "sémantiques avec gardes de sécurité, la gouvernance Human-in-the-Loop et les résultats mesurés.")
    story.append(Paragraph(intro_txt, style_intro))
    
    # -------------------------------------------------------------
    # PARTIE 1
    # -------------------------------------------------------------
    story.append(Paragraph("Partie 1 — Contexte Industriel & Problématique de l'Ingénierie des Exigences", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BLACK, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("L'Ingénierie des Exigences (Requirements Engineering - RE) constitue la pierre angulaire de tout cycle de vie logiciel. Selon les études empiriques de l'IEEE et du Standish Group, plus de 70% des défaillances critiques et des dépassements budgétaires en production trouvent leur origine dans des spécifications d'exigences (SRS) incomplètes, ambiguës, mal classifiées ou redondantes.", style_p))
    
    story.append(Paragraph("1.1  La distinction fondamentale : Exigences Fonctionnelles (FR) vs Non-Fonctionnelles (NFR)", style_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_GREY_LIGHT, spaceBefore=1, spaceAfter=6))
    
    story.append(Paragraph("• <b>Exigences Fonctionnelles (FR) :</b> Elles décrivent les actions, comportements, services et transformations de données que le logiciel doit accomplir en réponse à des entrées spécifiques (ex. : <i>« Le système doit permettre aux clients d'ajouter un article au panier »</i>).", style_bullet))
    story.append(Paragraph("• <b>Exigences Non-Fonctionnelles (NFR) :</b> Elles définissent les critères de qualité, contraintes opérationnelles, seuils de performance, exigences de sécurité et de conformité réglementaire régissant le comportement global du système (ex. : <i>« Le temps de réponse de l'API de paiement ne doit pas dépasser 500 ms à pleine charge »</i>).", style_bullet))
    
    story.append(make_tip_box("Pourquoi cette distinction est critique", 
                              "Dans les projets agiles et les cahiers des charges industriels, les ingénieurs mélangent fréquemment les préoccupations fonctionnelles et qualitatives dans une même phrase. Ce flou nuit à la conception architecturale : un développeur concevra une simple fonctionnalité là où un architecte réseau doit déployer une infrastructure à haute disponibilité.", 
                              usable_width))
    
    story.append(Paragraph("1.2  Les trois fléaux des spécifications textuelles", style_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_GREY_LIGHT, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph("<b>1. La mauvaise classification taxonomique :</b> Des contraintes vitales de sécurité ou de latence sont traitées comme de simples tickets de fonctionnalités et sous-estimées dans les sprints.", style_p))
    story.append(Paragraph("<b>2. L'ambiguïté lexicale et syntaxique (Norme IEEE 830) :</b> Présence d'adverbes vagues (« rapidement », « efficacement ») et de modaux non-contraignants (« devrait », « pourrait ») rendant l'exigence impossible à vérifier par un test automatique d'acceptation.", style_p))
    story.append(Paragraph("<b>3. La redondance et les quasi-doublons contradictoires :</b> Dans les projets complexes impliquant de multiples équipes, des exigences similaires sont rédigées avec des formulations divergentes ou des valeurs de SLA incompatibles.", style_p))

    # -------------------------------------------------------------
    # PARTIE 2
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("Partie 2 — Architecture Technique & Pipeline de Traitement", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BLACK, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("ReqMind AI repose sur un pipeline séquentiel hautement modulaire encapsulé dans le package Python <code>src/reqmind/</code>. L'orchestrateur central coordonne les étapes d'ingestion, de segmentation, d'analyses NLP en parallèle, de revue humaine et d'export multi-canaux.", style_p))
    
    schema_lines = [
        "+---------------------------------------------------------------------------------+",
        "|  Document Source : Texte brut, Fichiers .txt, .md, .csv ou .xlsx                |",
        "+---------------------------------------------------------------------------------+",
        "                                         |",
        "                                         v",
        "+---------------------------------------------------------------------------------+",
        "|  1. Ingestion & Validation UTF-8 (src/reqmind/ingestion.py)                     |",
        "+---------------------------------------------------------------------------------+",
        "                                         |",
        "                                         v",
        "+---------------------------------------------------------------------------------+",
        "|  2. Segmentation & Defense Decimale (src/reqmind/segmentation.py)               |",
        "|     - Offsets exacts start_char et end_char preserves                           |",
        "+---------------------------------------------------------------------------------+",
        "                                         |",
        "         +-------------------------------+-------------------------------+",
        "         |                               |                               |",
        "         v                               v                               v",
        "+-------------------+           +-------------------+           +-------------------+",
        "| 3. Classification |           | 4. Ambiguites     |           | 5. Similarite     |",
        "|    FR vs NFR      |           |    (IEEE 830)     |           |    Semantique     |",
        "| (LinearSVC,       |           | - Adverbes flous  |           | (all-MiniLM-L6-v2 |",
        "|  Calibrated SVM,  |           | - Modaux faibles  |           |  + Gardes Secu)   |",
        "|  DeBERTa-v3)      |           | - Spans exacts    |           |                   |",
        "+-------------------+           +-------------------+           +-------------------+",
        "                                         |",
        "                                         v",
        "+---------------------------------------------------------------------------------+",
        "|  6. Revue Human-in-the-Loop & Export Multi-Canaux (JSON, CSV, MD, PDF)          |",
        "+---------------------------------------------------------------------------------+",
    ]
    story.append(make_code_box(schema_lines, usable_width, style_code))
    
    story.append(Paragraph("2.2  La segmentation à défense décimale et conservation des offsets", style_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_GREY_LIGHT, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph("Une erreur classique en TALN consiste à découper aveuglément par les points. L'exigence <i>« The service shall maintain 99.9% availability during business hours. »</i> serait alors découpée à tort à cause du point décimal. ReqMind AI protège les motifs numériques <code>\\d+\\.\\d+</code> et les numérotations hiérarchiques (1.1, 1.2), tout en enregistrant <code>start_char</code> et <code>end_char</code> pour chaque élément.", style_p))
    
    if os.path.exists("reports/figures/03_longueurs.png"):
        story.append(Spacer(1, 6))
        img = Image("reports/figures/03_longueurs.png", width=14.5 * cm, height=6.5 * cm)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Paragraph("Figure 1 — Distribution des longueurs de textes des exigences en nombre de mots.", style_caption))

    # -------------------------------------------------------------
    # PARTIE 3
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("Partie 3 — Classification FR vs NFR : Étude Approfondie des Modèles", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BLACK, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("ReqMind AI intègre et compare trois architectures de classification :", style_p))
    story.append(Paragraph("• <b>Modèle 1 (Baseline TF-IDF + LinearSVC) :</b> Unigrammes et bigrammes avec pondération sous-linéaire et LinearSVC (C=1.0). Inférence ultra-rapide (< 0.5 ms), mais aveugle aux dépendances à longue distance.", style_bullet))
    story.append(Paragraph("• <b>Modèle 2 (SVM Optimisé & Calibré) :</b> FeatureUnion combinant n-grammes de mots (1, 2) et n-grammes de caractères (3, 5), pondération équilibrée (class_weight='balanced') et probabilités calibrées via le scaling de Platt (CalibratedClassifierCV).", style_bullet))
    story.append(Paragraph("• <b>Modèle 3 (DeBERTa-v3-small Fine-Tuned) :</b> Architecture Transformer avec attention désenchevêtrée (Disentangled Attention) décomposant l'attention entre contenu et position relative. Régularisée par Label Smoothing (0.1).", style_bullet))
    
    story.append(Paragraph("3.2  Résultats expérimentaux sur le Benchmark Humain Annoté (60 Exemples)", style_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_GREY_LIGHT, spaceBefore=1, spaceAfter=6))
    
    # Table of metrics
    tbl_data = [
        ["Métrique d'Évaluation", "Baseline SVM", "SVM Calibré", "DeBERTa-v3-small"],
        ["Exactitude (Accuracy)", "76.67% (46/60)", "80.00% (48/60)", "96.67% (58/60)"],
        ["Macro-F1", "76.64%", "79.80%", "96.67%"],
        ["Précision FR", "75.00%", "75.00%", "96.67%"],
        ["Rappel FR", "80.00%", "90.00%", "96.67%"],
        ["F1-Score FR", "77.42%", "81.82%", "96.67%"],
        ["Précision NFR", "78.57%", "87.50%", "96.67%"],
        ["Rappel NFR", "73.33%", "70.00%", "96.67%"],
        ["F1-Score NFR", "75.86%", "77.78%", "96.67%"],
        ["Nombre d'Erreurs Global", "14 erreurs", "12 erreurs", "2 erreurs (-85.7%)"],
        ["Latence CPU / Exigence", "0.46 ms", "0.35 ms", "14.66 ms (~42x)"],
    ]
    t_metrics = Table(tbl_data, colWidths=[usable_width * 0.40, usable_width * 0.20, usable_width * 0.20, usable_width * 0.20])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_GREY),
        ('LINEABOVE', (0, 0), (-1, 0), 1, C_BLACK),
        ('LINEBELOW', (0, 0), (-1, 0), 1, C_BLACK),
        ('LINEBELOW', (0, -1), (-1, -1), 1, C_BLACK),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_GREY_LIGHT),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_metrics)
    
    if os.path.exists("reports/figures/optimized_svm_cv_calibration_curve.png"):
        story.append(Spacer(1, 8))
        img2 = Image("reports/figures/optimized_svm_cv_calibration_curve.png", width=14.5 * cm, height=6.0 * cm)
        img2.hAlign = 'CENTER'
        story.append(img2)
        story.append(Paragraph("Figure 2 — Courbe de calibration du SVM optimisé après scaling de Platt.", style_caption))
        
    if os.path.exists("reports/figures/step3_deberta_training_curves.png"):
        story.append(Spacer(1, 6))
        img3 = Image("reports/figures/step3_deberta_training_curves.png", width=14.5 * cm, height=6.2 * cm)
        img3.hAlign = 'CENTER'
        story.append(img3)
        story.append(Paragraph("Figure 3 — Courbes d'apprentissage de DeBERTa-v3-small (Convergence perte et Macro-F1).", style_caption))

    story.append(Paragraph("3.4  Pourquoi DeBERTa surpasse les modèles classiques", style_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_GREY_LIGHT, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph("Sur l'exigence : <i>« The system shall encrypt all sensitive customer data before writing to the database. »</i>, le SVM prédit <b>FR</b> à tort, trompé par les verbes d'action (<i>shall, write, database</i>). DeBERTa prédit <b>NFR</b> avec 96% de confiance car son attention désenchevêtrée comprend que <i>encrypt</i> est une contrainte de sécurité appliquée à l'opération et non une fonction métier offerte à l'utilisateur.", style_p))

    # -------------------------------------------------------------
    # PARTIE 4
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("Partie 4 — Détection d'Ambiguïtés & Défauts Qualité selon IEEE 830", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BLACK, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("La norme IEEE 830 exige que toute exigence soit vérifiable et testable. ReqMind AI analyse les textes à travers quatre classes de défauts :", style_p))
    
    def_data = [
        ["Catégorie", "Mots Détectés", "Risque Industriel", "Recommandation"],
        ["Adverbes Vagues", "quickly, efficiently, regularly", "Test impossible à automatiser.", "Remplacer par un SLA mesurable."],
        ["Revendications Subjectives", "user-friendly, intuitive", "Appréciation subjective.", "Remplacer par un score d'utilisabilité."],
        ["Modaux Non-Contraignants", "should, could, may, might", "Viole la RFC 2119 (facultatif).", "Employer obligatoirement 'shall'."],
        ["Proportionnalité Faible", "as appropriate, if feasible", "Échappatoire technique floue.", "Définir les conditions logiques exactes."],
    ]
    t_def = Table(def_data, colWidths=[usable_width * 0.22, usable_width * 0.25, usable_width * 0.25, usable_width * 0.28])
    t_def.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_BG_GREY),
        ('LINEABOVE', (0, 0), (-1, 0), 1, C_BLACK),
        ('LINEBELOW', (0, 0), (-1, 0), 1, C_BLACK),
        ('LINEBELOW', (0, -1), (-1, -1), 1, C_BLACK),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_GREY_LIGHT),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_def)
    
    story.append(Paragraph("<b>Gestion des Faux-Positifs (Dismiss) :</b> Si un terme technique légitime est détecté (ex. : <i>« fast Fourier transform »</i>), l'ingénieur clique sur 'Rejeter l'alerte' dans l'interface. L'alerte est désactivée dans l'audit sans modifier le texte source.", style_p))

    # -------------------------------------------------------------
    # PARTIE 5
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("Partie 5 — Détection Sémantique de Doublons & Gardes de Sécurité", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BLACK, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("ReqMind AI utilise le bi-encodeur dense <code>sentence-transformers/all-MiniLM-L6-v2</code> (384 dimensions) pour calculer la similarité cosinus entre exigences. Pour éviter les erreurs grossières des approches vectorielles pures, le système introduit deux gardes logiques :", style_p))
    story.append(Paragraph("• <b>La Garde de Négation :</b> Alerte si deux exigences partageant une forte similarité syntaxique ont une polarité inversée (ex. : <i>shall allow</i> vs <i>shall not allow</i>).", style_bullet))
    story.append(Paragraph("• <b>La Garde de Conflit SLA / Numérique :</b> Empêche de fusionner deux exigences partageant la même formulation mais avec des seuils numériques divergents (ex. : <i>« complete within 2 hours »</i> vs <i>« complete within 6 hours »</i>).", style_bullet))

    # -------------------------------------------------------------
    # PARTIE 6
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("Partie 6 — Gouvernance Human-in-the-Loop & Traçabilité Multi-Canaux", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BLACK, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("<b>Machine à états finis de révision :</b> Chaque exigence suit le cycle <code>PENDING -> APPROVED / EDITED / REJECTED</code>. Le champ <code>predicted_label</code> reste immuable pour garantir l'auditabilité, tandis que l'adjudication de l'ingénieur est enregistrée dans <code>reviewer_label</code>.", style_p))
    story.append(Paragraph("<b>Quatre canaux d'exportation :</b>", style_p))
    story.append(Paragraph("1. <b>JSON Intégral :</b> Sérialisation Pydantic complète avec offsets exacts et validateur aller-retour.", style_bullet))
    story.append(Paragraph("2. <b>CSV Tabulaire :</b> Feuille d'audit avec labels résolus pour import Jira/DOORS.", style_bullet))
    story.append(Paragraph("3. <b>Markdown Exécutif :</b> Synthèse lisible et distributions globales.", style_bullet))
    story.append(Paragraph("4. <b>PDF Haute Fidélité :</b> Document ReportLab avec canvas numéroté (« Page X / Y ») et cartes KPI.", style_bullet))

    # -------------------------------------------------------------
    # PARTIE 7
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("Partie 7 — Guide de Démonstration en 5 Minutes pour Soutenance", style_h1))
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BLACK, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("<b>Minute 1 (Ingestion) :</b> Charger le scénario 'E-Commerce Platform SRS'. Montrer la segmentation propre sans rupture des décimales (ex: '99.9% uptime').", style_bullet))
    story.append(Paragraph("<b>Minute 2 (Qualité IEEE 830) :</b> Pointer les badges en surbrillance sur 'quickly' et 'user-friendly'. Démontrer le bouton de rejet de faux positif.", style_bullet))
    story.append(Paragraph("<b>Minute 3 (Doublons & SLA) :</b> Afficher les paires à similarité > 0.85 et mettre en avant l'alerte de conflit numérique SLA.", style_bullet))
    story.append(Paragraph("<b>Minute 4 (Revue & Export) :</b> Basculer le statut en APPROVED/EDITED et télécharger le rapport d'audit PDF complet.", style_bullet))
    story.append(Paragraph("<b>Minute 5 (Laboratoire d'Évaluation) :</b> Dans l'Onglet 8, lancer la comparaison SVM vs DeBERTa sur le test set de 60 cas. Mettre en valeur le bond d'exactitude (+20 pts : 76.7% -> 96.7%) et la matrice de confusion interactive.", style_bullet))

    # Build Document
    doc.build(story, canvasmaker=PedagogicalNumberedCanvas)
    print(f"PDF successfully built: {output_filename}")


def make_tip_box(title, text, width):
    p_content = Paragraph(f"<b><i>{title} :</i></b> <i>{text}</i>", ParagraphStyle(
        "TipContent",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=C_GREY_DARK,
    ))
    t = Table([[p_content]], colWidths=[width])
    t.setStyle(TableStyle([
        ('LINEBEFORE', (0, 0), (0, 0), 2.5, C_GREY_MID),
        ('LEFTMARGIN', (0, 0), (0, 0), 10),
        ('RIGHTMARGIN', (0, 0), (0, 0), 5),
        ('TOPPADDING', (0, 0), (0, 0), 4),
        ('BOTTOMPADDING', (0, 0), (0, 0), 4),
    ]))
    return t


def make_code_box(lines, width, style):
    content = [Paragraph(l.replace(" ", "&nbsp;"), style) for l in lines]
    t = Table([[content]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, C_GREY_LIGHT),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


if __name__ == "__main__":
    build_pdf()
