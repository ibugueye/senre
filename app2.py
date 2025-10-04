import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
from datetime import datetime, timedelta
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import base64
import random

# Configuration de la page
st.set_page_config(
    page_title="Guide Complet Contrôle de Gestion",
    page_icon="📚",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 3px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .concept-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .formula-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .exercise-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .calculator-box {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .variant-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #64b5f6;
        margin: 1rem 0;
        font-style: italic;
    }
    .export-box {
        background-color: #fff9c4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .quiz-box {
        background-color: #fce4ec;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #e91e63;
        margin: 1rem 0;
    }
    .montecarlo-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">📚 GUIDE COMPLET : CONTRÔLE DE GESTION</div>', unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("🎯 Navigation du Guide")
chapitre = st.sidebar.radio(
    "Choisissez un chapitre:",
    ["🏠 Accueil & Fondements",
     "💰 Budgets Opérationnels",
     "📊 Analyse des Écarts",
     "🏗️ Évaluation d'Investissement",
     "💸 Budget de Trésorerie",
     "📈 Tableaux de Bord",
     "🎓 Exercices & Cas Pratiques",
     "🧠 Quiz de Validation"]
)

# Variables globales pour l'analyse des écarts
ecart_ventes = 0
ecart_couts = 0
ecart_marge = 0

def outils_prevision():
    st.markdown('<div class="section-header">🛠️ Outils de Prévision Avancés</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="concept-card">
    <h4>🎯 Pourquoi les prévisions sont-elles cruciales ?</h4>
    <p>Les prévisions permettent d'anticiper l'avenir et de prendre des décisions éclairées.
    En contrôle de gestion, une bonne prévision peut faire la différence entre le succès et l'échec.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Régression Linéaire", "📊 Méthodes Saisonnères", "🎯 Méthodes Qualitatives", "🧮 Calculateurs", "📋 Bonnes Pratiques"])

    with tab1:
        st.subheader("📈 Régression Linéaire - L'outil fondamental")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-card">
            <h4>🎯 Définition</h4>
            <p>La <strong>régression linéaire</strong> est une méthode statistique qui modélise la relation
            entre une variable dépendante (Y) et une ou plusieurs variables indépendantes (X).</p>
            <p><strong>Objectif :</strong> Trouver la droite qui passe "au plus près" des points observés.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="formula-box">
            <h4>🧮 Formule de base</h4>
            <strong>Y = aX + b</strong><br>
            Où :<br>
            • <strong>Y</strong> = Variable à prévoir (ex: ventes)<br>
            • <strong>X</strong> = Variable explicative (ex: temps, budget pub)<br>
            • <strong>a</strong> = Pente de la droite (coefficient)<br>
            • <strong>b</strong> = Ordonnée à l'origine<br><br>
            <strong>Calcul des coefficients :</strong><br>
            a = Σ[(Xi - X̄)(Yi - Ȳ)] / Σ(Xi - X̄)²<br>
            b = Ȳ - aX̄
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-card">
            <h4>📊 Interprétation des résultats</h4>
            <strong>Coefficient de détermination R² :</strong>
            <div class="formula-box">
            R² = 1 - (Σ(Yi - Ŷi)² / Σ(Yi - Ȳ)²)
            </div>
            <p><strong>Signification :</strong></p>
            <ul>
            <li><strong>R² = 1</strong> : Relation parfaite</li>
            <li><strong>R² = 0.8</strong> : Très bonne relation</li>
            <li><strong>R² = 0.5</strong> : Relation moyenne</li>
            <li><strong>R² < 0.3</strong> : Relation faible</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="exercise-box">
            <h4>🎯 Exemple pratique</h4>
            <p><strong>Scénario :</strong> Relation entre budget publicitaire et ventes</p>
            <p>Budget (X) : 10k€, 20k€, 30k€, 40k€, 50k€</p>
            <p>Ventes (Y) : 100k€, 180k€, 260k€, 340k€, 420k€</p>
            <p><strong>Équation :</strong> Ventes = 8 × Budget + 20</p>
            <p><strong>Interprétation :</strong> Chaque 1 000€ investi en pub génère 8 000€ de ventes supplémentaires</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📊 Méthodes de Série Temporelle")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-card">
            <h4>🔄 Composantes d'une série temporelle</h4>
            <div class="formula-box">
            <strong>Yt = Tt + St + Ct + Et</strong><br>
            Où :<br>
            • <strong>Tt</strong> = Tendence (Trend)<br>
            • <strong>St</strong> = Saisonnalité (Seasonality)<br>
            • <strong>Ct</strong> = Cyclique (Cycle)<br>
            • <strong>Et</strong> = Aléatoire (Error)
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="concept-card">
            <h4>📈 Lissage Exponentiel Simple</h4>
            <div class="formula-box">
            <strong>Ŷt+1 = α × Yt + (1-α) × Ŷt</strong><br>
            Où :<br>
            • <strong>Ŷt+1</strong> = Prévision période suivante<br>
            • <strong>Yt</strong> = Valeur actuelle<br>
            • <strong>Ŷt</strong> = Prévision actuelle<br>
            • <strong>α</strong> = Constante de lissage (0 < α < 1)
            </div>
            <p><strong>Choix de α :</strong></p>
            <ul>
            <li><strong>α élevé (0.7-0.9)</strong> : Réaction rapide aux changements</li>
            <li><strong>α moyen (0.3-0.7)</strong> : Équilibre réactivité/stabilité</li>
            <li><strong>α faible (0.1-0.3)</strong> : Lissage important</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-card">
            <h4>🎯 Méthode de Holt-Winters</h4>
            <p><strong>Pour les séries avec tendance et saisonnalité</strong></p>
            <div class="formula-box">
            <strong>Composantes :</strong><br>
            • <strong>Niveau :</strong> Lt = α(Yt/St-s) + (1-α)(Lt-1 + Tt-1)<br>
            • <strong>Tendance :</strong> Tt = β(Lt - Lt-1) + (1-β)Tt-1<br>
            • <strong>Saisonnalité :</strong> St = γ(Yt/Lt) + (1-γ)St-s<br><br>
            <strong>Prévision :</strong><br>
            Ŷt+h = (Lt + h × Tt) × St-s+h
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="concept-card">
            <h4>📊 Moyenne Mobile</h4>
            <div class="formula-box">
            <strong>MMn(t) = (Yt + Yt-1 + ... + Yt-n+1) / n</strong><br>
            Où :<br>
            • <strong>n</strong> = Nombre de périodes dans la moyenne<br>
            • <strong>Yt</strong> = Valeur à la période t
            </div>
            <p><strong>Applications :</strong></p>
            <ul>
            <li><strong>MM3</strong> : Court terme, réactif</li>
            <li><strong>MM6</strong> : Moyen terme, équilibré</li>
            <li><strong>MM12</strong> : Long terme, très lissé</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.subheader("🎯 Méthodes Qualitatives")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-card">
            <h4>👥 Méthode Delphi</h4>
            <p><strong>Processus :</strong></p>
            <ol>
            <li>Sélection d'experts</li>
            <li>Questionnaire anonyme</li>
            <li>Synthèse des réponses</li>
            <li>Retour aux experts avec résultats</li>
            <li>Nouveau tour jusqu'à consensus</li>
            </ol>
            <p><strong>Avantages :</strong></p>
            <ul>
            <li>Évite l'influence des personnalités dominantes</li>
            <li>Convergence vers un consensus</li>
            <li>Utilise l'expertise collective</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="concept-card">
            <h4>📊 Avis des Forces de Vente</h4>
            <p><strong>Méthodologie :</strong></p>
            <div class="formula-box">
            Prévision = Σ(Estimation commercial × Coefficient de confiance) / Nombre de commerciaux
            </div>
            <p><strong>Facteurs à considérer :</strong></p>
            <ul>
            <li>Historique de précision du commercial</li>
            <li>Pipeline de ventes</li>
            <li>Conditions du marché local</li>
            <li>Actions commerciales planifiées</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-card">
            <h4>📈 Études de Marché</h4>
            <p><strong>Techniques :</strong></p>
            <ul>
            <li><strong>Sondages</strong> : Questionnaires quantitatifs</li>
            <li><strong>Groupes de discussion</strong> : Entretiens qualitatifs</li>
            <li><strong>Analyse de la concurrence</strong> : Benchmarking</li>
            <li><strong>Tests de concept</strong> : Validation d'idées</li>
            </ul>
            <p><strong>Indicateurs clés :</strong></p>
            <div class="formula-box">
            Intention d'achat = (Nombre de "Très probable") × 0.8 + (Nombre de "Probable") × 0.5
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="concept-card">
            <h4>🎯 Jugement d'Expert</h4>
            <p><strong>Méthode :</strong> Combinaison d'opinions d'experts</p>
            <div class="formula-box">
            Prévision pondérée = Σ(Prévision expert × Poids expert) / Σ Poids
            </div>
            <p><strong>Critères de pondération :</strong></p>
            <ul>
            <li>Expérience dans le domaine</li>
            <li>Précision des prévisions passées</li>
            <li>Connaissance du marché</li>
            <li>Ancienneté dans l'entreprise</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.subheader("🧮 Calculateurs de Prévision")
        st.markdown("""
        <div class="concept-card">
        <h4>📈 Calculateur de Régression Linéaire</h4>
        <p>Entrez vos données pour calculer automatiquement l'équation de prévision</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Données d'entrée**")
            n_points = st.slider("Nombre de points de données:", 3, 20, 5)
            data = []
            for i in range(n_points):
                col_a, col_b = st.columns(2)
                with col_a:
                    x = st.number_input(f"X{i+1} (Variable explicative):", value=(i+1)*10, key=f"x{i}")
                with col_b:
                    y = st.number_input(f"Y{i+1} (Variable à prévoir):", value=(i+1)*15, key=f"y{i}")
                data.append((x, y))
        with col2:
            if st.button("Calculer la régression"):
                df = pd.DataFrame(data, columns=['X', 'Y'])
                x_mean = df['X'].mean()
                y_mean = df['Y'].mean()
                numerator = ((df['X'] - x_mean) * (df['Y'] - y_mean)).sum()
                denominator = ((df['X'] - x_mean) ** 2).sum()
                a = numerator / denominator if denominator != 0 else 0
                b = y_mean - a * x_mean
                model = LinearRegression()
                model.fit(df[['X']], df['Y'])
                r2 = r2_score(df['Y'], model.predict(df[['X']]))
                st.markdown(f"""
                <div class="success-box">
                <h4>📊 Résultats de la Régression</h4>
                <div class="formula-box">
                <strong>Équation de prévision :</strong><br>
                Y = {a:.2f}X + {b:.2f}
                </div>
                <p><strong>Interprétation :</strong></p>
                <ul>
                <li>Quand X augmente de 1, Y augmente de {a:.2f}</li>
                <li>Quand X = 0, Y = {b:.2f}</li>
                <li>Qualité du modèle : R² = {r2:.3f}</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                fig = px.scatter(df, x='X', y='Y', title='Régression Linéaire')
                x_range = np.linspace(df['X'].min(), df['X'].max(), 100)
                y_pred = a * x_range + b
                fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Droite de régression'))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Calculateur de Moyenne Mobile")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Paramètres de la moyenne mobile**")
            donnees_historiques = st.text_area("Données historiques (séparées par des virgules):",
                                             "100, 120, 110, 130, 125, 140, 135, 150")
            periode_mm = st.slider("Période de la moyenne mobile:", 2, 12, 3)
            if st.button("Calculer moyenne mobile"):
                try:
                    data_list = [float(x.strip()) for x in donnees_historiques.split(',')]
                    mm_values = []
                    for i in range(len(data_list) - periode_mm + 1):
                        moyenne = sum(data_list[i:i+periode_mm]) / periode_mm
                        mm_values.append(moyenne)
                    prevision = sum(data_list[-periode_mm:]) / periode_mm
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>📈 Résultats Moyenne Mobile</h4>
                    <p><strong>Dernière moyenne mobile :</strong> {mm_values[-1]:.2f}</p>
                    <p><strong>Prévision période suivante :</strong> {prevision:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=data_list, mode='lines+markers', name='Données réelles'))
                    fig.add_trace(go.Scatter(y=mm_values, mode='lines', name=f'Moyenne Mobile {periode_mm}'))
                    fig.update_layout(title='Moyenne Mobile')
                    st.plotly_chart(fig, use_container_width=True)
                except ValueError:
                    st.error("Veuillez entrer des nombres valides séparés par des virgules")

    with tab5:
        st.subheader("📋 Bonnes Pratiques de Prévision")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-card">
            <h4>✅ Méthodologie recommandée</h4>
            <p><strong>1. Combiner plusieurs méthodes</strong></p>
            <div class="formula-box">
            Prévision finale = (Prévision quantitative × 0.6) + (Prévision qualitative × 0.4)
            </div>
            <p><strong>2. Valider avec des tests statistiques</strong></p>
            <ul>
            <li>Test de normalité des résidus</li>
            <li>Analyse de l'autocorrélation</li>
            <li>Test de stationnarité</li>
            </ul>
            <p><strong>3. Utiliser des intervalles de confiance</strong></p>
            <div class="formula-box">
            Intervalle = Prévision ± (Z × Écart-type)
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="concept-card">
            <h4>📊 Mesure de la performance</h4>
            <p><strong>MAPE (Mean Absolute Percentage Error)</strong></p>
            <div class="formula-box">
            MAPE = (1/n) × Σ|(Réel - Prévision)/Réel| × 100%
            </div>
            <p><strong>Interprétation MAPE :</strong></p>
            <ul>
            <li>< 10% : Excellente précision</li>
            <li>10-20% : Bonne précision</li>
            <li>20-50% : Précision moyenne</li>
            <li> > 50% : Précision faible</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-card">
            <h4>⚠️ Pièges à éviter</h4>
            <p><strong>1. Surajustement (Overfitting)</strong></p>
            <p>Un modèle trop complexe qui s'ajuste parfaitement aux données passées mais généralise mal.</p>
            <p><strong>2. Biais de confirmation</strong></p>
            <p>Tendre à favoriser les informations qui confirment ses croyances initiales.</p>
            <p><strong>3. Ignorer les points de rupture</strong></p>
            <p>Les modèles supposent que les tendances passées continuent, ce qui n'est pas toujours vrai.</p>
            <p><strong>4. Négliger les facteurs externes</strong></p>
            <p>Changements réglementaires, innovations technologiques, crises économiques.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="concept-card">
            <h4>🔄 Processus d'amélioration continue</h4>
            <p><strong>Cycle de prévision :</strong></p>
            <ol>
            <li><strong>Planifier</strong> : Définir objectifs et méthodes</li>
            <li><strong>Collecter</strong> : Données historiques et informations</li>
            <li><strong>Modéliser</strong> : Appliquer les méthodes choisies</li>
            <li><strong>Valider</strong> : Tester la précision</li>
            <li><strong>Adjuster</strong> : Corriger basé sur le feedback</li>
            <li><strong>Documenter</strong> : Enregistrer hypothèses et résultats</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Application Pratique : Prévision des Ventes")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Simulation de prévision**")
        methode = st.selectbox("Méthode de prévision:",
                              ["Régression linéaire", "Moyenne mobile", "Lissage exponentiel"])
        if methode == "Régression linéaire":
            budget_pub = st.number_input("Budget publicitaire (k€):", 10, 100, 50)
            equation = st.text_input("Équation de régression (Y = aX + b):", "Y = 0.8X + 20")
            if st.button("Calculer prévision"):
                try:
                    parts = equation.split('=')
                    if len(parts) == 2:
                        droite = parts[1].strip()
                        if '+' in droite:
                            a_str, b_str = droite.split('+')
                            a = float(a_str.replace('X', '').strip())
                            b = float(b_str.strip())
                        else:
                            a = float(droite.replace('X', '').strip())
                            b = 0
                    prevision = a * budget_pub + b
                    st.success(f"**Prévision des ventes : {prevision:.0f} k€**")
                except:
                    st.error("Format d'équation invalide. Utilisez le format: Y = aX + b")
    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>📈 Checklist de validation</h4>
        <p><strong>Avant de valider une prévision :</strong></p>
        <ul>
        <li>✓ Les données historiques sont-elles complètes ?</li>
        <li>✓ Les hypothèses sont-elles documentées ?</li>
        <li>✓ L'intervalle de confiance est-il calculé ?</li>
        <li>✓ Les facteurs saisonniers sont-ils pris en compte ?</li>
        <li>✓ La méthode est-elle adaptée au contexte ?</li>
        <li>✓ Y a-t-il un plan de contingence ?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def generate_pdf(content, title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))

    for line in content.split('\n'):
        if line.strip():
            elements.append(Paragraph(line, styles['Normal']))
        else:
            elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def monte_carlo_simulation(investissement, flux, duree, iterations=1000):
    results = []
    for _ in range(iterations):
        simulated_flux = [f * random.uniform(0.9, 1.1) for f in flux]
        van = -investissement
        for annee, f in enumerate(simulated_flux, 1):
            van += f / ((1 + 0.1) ** annee)
        results.append(van)
    return results

if chapitre == "🏠 Accueil & Fondements":
    st.markdown('<div class="section-header">🎯 Rôles et Missions Stratégiques du Contrôleur de Gestion</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Piloter la Performance")
        st.markdown("""
        <div class="concept-card">
        <h4>🎯 Définition et suivi des KPIs</h4>
        <ul>
            <li><strong>Chiffre d'affaires</strong> : Mesure l'activité commerciale</li>
            <li><strong>Marge commerciale</strong> : CA - Coût des ventes</li>
            <li><strong>Rentabilité</strong> : Résultat net / CA</li>
            <li><strong>Productivité</strong> : Output / Input</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("🔍 Analyser les Écarts")
        st.markdown("""
        <div class="concept-card">
        <h4>Méthodologie d'analyse</h4>
        <ol>
            <li>Comparer réel vs budget</li>
            <li>Identifier les écarts significatifs</li>
            <li>Analyser les causes racines</li>
            <li>Proposer des actions correctives</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.subheader("📋 Élaborer les Budgets")
        st.markdown("""
        <div class="concept-card">
        <h4>Processus budgétaire</h4>
        <ul>
            <li><strong>Budget des ventes</strong> : Point de départ</li>
            <li><strong>Budget de production</strong> : Planification</li>
            <li><strong>Budget des approvisionnements</strong> : Optimisation</li>
            <li><strong>Budgets financiers</strong> : Synthèse</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("💡 Éclairer la Prise de Décision")
        st.markdown("""
        <div class="concept-card">
        <h4>Rôle stratégique</h4>
        <ul>
            <li>Fournir des analyses fiables</li>
            <li>Simuler des scénarios</li>
            <li>Évaluer les investissements</li>
            <li>Anticiper les risques</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Importance Cruciale de la Gestion Budgétaire</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("""
        **🎯 Anticipation**
        - Regarder vers l'avenir
        - Se poser les bonnes questions
        - Préparer les ressources
        """)
    with col2:
        st.info("""
        **🔄 Coordination**
        - Langage commun
        - Cohérence entre services
        - Objectifs partagés
        """)
    with col3:
        st.info("""
        **👥 Responsabilisation**
        - Autonomie des responsables
        - Objectifs clairs
        - Redevabilité
        """)
    with col4:
        st.info("""
        **📊 Contrôle**
        - Référence de performance
        - Détection des dérives
        - Ajustements rapides
        """)

elif chapitre == "💰 Budgets Opérationnels":
    st.markdown('<div class="section-header">💰 La Boucle Budgétaire : Processus Séquentiel</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;'>
    <h4>🔄 Processus Budgétaire Séquentiel</h4>
    <p>Budget Ventes → Budget Production → Budget Approvisionnements → Budgets Financiers</p>
    </div>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 Budget des Ventes", "🏭 Budget de Production", "📦 Budget des Approvisionnements"])
    with tab1:
        st.subheader("📈 Budget des Ventes - Point de Départ")
        st.markdown("""
        <div class="concept-card">
        <h4>🎯 Méthodologie d'élaboration</h4>
        <ol>
            <li><strong>Analyse du passé</strong> : Étude des ventes historiques et tendances</li>
            <li><strong>Étude de marché</strong> : Conjoncture économique, concurrence</li>
            <li><strong>Actions commerciales</strong> : Nouveaux produits, campagnes</li>
            <li><strong>Objectifs stratégiques</strong> : Parts de marché à gagner</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        outils_prevision()
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Méthodes quantitatives**")
            st.checkbox("Régression linéaire")
            st.checkbox("Séries temporelles")
            st.checkbox("Lissage exponentiel")
        with col2:
            st.write("**Méthodes qualitatives**")
            st.checkbox("Avis forces de vente")
            st.checkbox("Études de marché")
            st.checkbox("Jury d'experts")
        col1, col2 = st.columns(2)
        with col1:
            ca_historique = st.number_input("CA historique moyen (€):", 100000, 1000000, 500000)
            croissance_marche = st.slider("Croissance du marché (%):", -10.0, 20.0, 3.0)
        with col2:
            budget_pub = st.number_input("Budget publicité (€):", 0, 100000, 25000)
            objectif_pdm = st.slider("Objectif part de marché (%):", 1.0, 50.0, 15.0)
        if st.button("Calculer la prévision"):
            prevision = ca_historique * (1 + croissance_marche/100) * (1 + budget_pub/100000)
            st.success(f"**Prévision des ventes : €{prevision:,.0f}**")
            st.markdown("""
            <div class="export-box">
            <h4>📥 Export des résultats</h4>
            </div>
            """, unsafe_allow_html=True)
            content = f"""
            PRÉVISION DES VENTES
            -------------------
            CA historique : {ca_historique:,.0f}€
            Croissance marché : {croissance_marche}%
            Budget publicité : {budget_pub:,.0f}€
            Objectif PDM : {objectif_pdm}%

            RÉSULTAT :
            Prévision des ventes : {prevision:,.0f}€
            """
            pdf = generate_pdf(content, "Prévision des Ventes")
            st.download_button(
                label="⬇️ Télécharger en PDF",
                data=pdf,
                file_name="prevision_ventes.pdf",
                mime="application/pdf"
            )
    with tab2:
        st.subheader("🏭 Budget de Production")
        st.markdown("""
        <div class="formula-box">
        <strong>Formule fondamentale :</strong><br>
        Quantité à produire = (Ventes prévues + Stock final cible) - Stock initial
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            ventes_prevues = st.number_input("Ventes prévues (unités):", 1000, 50000, 10000)
            stock_initial = st.number_input("Stock initial (unités):", 0, 10000, 2000)
            stock_final_cible = st.number_input("Stock final cible (unités):", 0, 10000, 2500)
        with col2:
            temps_unitaire = st.number_input("Temps de production/unité (h):", 0.1, 10.0, 2.0)
            cout_horaire = st.number_input("Coût horaire main d'œuvre (€):", 10.0, 100.0, 25.0)
            cout_matiere = st.number_input("Coût matière/unité (€):", 1.0, 100.0, 15.0)
        if st.button("Calculer le budget production"):
            quantite_production = ventes_prevues + stock_final_cible - stock_initial
            cout_main_oeuvre = quantite_production * temps_unitaire * cout_horaire
            cout_matiere_total = quantite_production * cout_matiere
            st.markdown(f"""
            <div class="success-box">
            <h4>📋 Budget de Production</h4>
            <p><strong>Quantité à produire</strong> : {quantite_production:,.0f} unités</p>
            <p><strong>Coût main d'œuvre</strong> : €{cout_main_oeuvre:,.0f}</p>
            <p><strong>Coût matières</strong> : €{cout_matiere_total:,.0f}</p>
            <p><strong>Coût total production</strong> : €{cout_main_oeuvre + cout_matiere_total:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            data = {
                "Poste": ["Quantité à produire", "Coût main d'œuvre", "Coût matières", "Coût total"],
                "Valeur": [f"{quantite_production:,.0f} unités", f"€{cout_main_oeuvre:,.0f}", f"€{cout_matiere_total:,.0f}", f"€{cout_main_oeuvre + cout_matiere_total:,.0f}"]
            }
            df_export = pd.DataFrame(data)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.markdown("""
            <div class="export-box">
            <h4>📥 Export des résultats</h4>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Télécharger en CSV",
                data=csv,
                file_name="budget_production.csv",
                mime="text/csv"
            )
    with tab3:
        st.subheader("📦 Budget des Approvisionnements et Optimisation")
        st.markdown("""
        <div class="formula-box">
        <strong>Formule des besoins :</strong><br>
        Quantités à acheter = (Besoins production + Stock final cible matières) - Stock initial matières
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-box">
        <strong>Formule de Wilson :</strong><br>
        QEC = √(2 × D × Cc / Cp)<br>
        Où :<br>
        D = Demande annuelle<br>
        Cc = Coût de passation de commande<br>
        Cp = Coût de possession unitaire annuel
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            demande_annuelle = st.number_input("Demande annuelle (unités):", 1000, 100000, 12000)
            cout_commande = st.number_input("Coût de commande (€):", 10.0, 500.0, 50.0)
        with col2:
            cout_possession = st.number_input("Coût possession/unité/an (€):", 0.1, 10.0, 2.0)
            delai_livraison = st.number_input("Délai livraison (jours):", 1, 30, 7)
        if st.button("Calculer QEC"):
            qec = math.sqrt((2 * demande_annuelle * cout_commande) / cout_possession)
            n_commandes = demande_annuelle / qec
            stock_moyen = qec / 2
            cout_total = (demande_annuelle * cout_commande / qec) + (stock_moyen * cout_possession)
            st.markdown(f"""
            <div class="success-box">
            <h4>🎯 Résultats d'Optimisation</h4>
            <p><strong>Quantité économique</strong> : {qec:.0f} unités</p>
            <p><strong>Nombre de commandes/an</strong> : {n_commandes:.1f}</p>
            <p><strong>Stock moyen</strong> : {stock_moyen:.0f} unités</p>
            <p><strong>Coût total optimal</strong> : €{cout_total:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            data = {
                "Indicateur": ["Quantité économique (QEC)", "Nombre de commandes/an", "Stock moyen", "Coût total optimal"],
                "Valeur": [f"{qec:.0f} unités", f"{n_commandes:.1f}", f"{stock_moyen:.0f} unités", f"€{cout_total:,.0f}"]
            }
            df_export = pd.DataFrame(data)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.markdown("""
            <div class="export-box">
            <h4>📥 Export des résultats</h4>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Télécharger en CSV",
                data=csv,
                file_name="optimisation_stocks.csv",
                mime="text/csv"
            )

elif chapitre == "📊 Analyse des Écarts":
    st.markdown('<div class="section-header">🔍 Méthodologie d\'Analyse des Écarts</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="concept-card">
    <h4>🎯 Processus d'analyse</h4>
    <ol>
        <li><strong>Identification</strong> : Repérer les écarts significatifs</li>
        <li><strong>Quantification</strong> : Mesurer l'ampleur des écarts</li>
        <li><strong>Analyse causale</strong> : Comprendre les raisons</li>
        <li><strong>Action corrective</strong> : Proposer des solutions</li>
        <li><strong>Suivi</strong> : Vérifier l'efficacité des actions</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Données Réelles**")
        ventes_reel = st.number_input("Ventes réelles (€):", 0, 1000000, 480000)
        couts_reel = st.number_input("Coûts réels (€):", 0, 800000, 350000)
    with col2:
        st.write("**Données Budgétées**")
        ventes_budget = st.number_input("Ventes budgétées (€):", 0, 1000000, 500000)
        couts_budget = st.number_input("Coûts budgétés (€):", 0, 800000, 320000)
    with col3:
        st.write("**Calculs automatiques**")
        if st.button("Calculer les écarts"):
            ecart_ventes, ecart_couts, ecart_marge
            ecart_ventes = ventes_reel - ventes_budget
            ecart_couts = couts_reel - couts_budget
            marge_reel = ventes_reel - couts_reel
            marge_budget = ventes_budget - couts_budget
            ecart_marge = marge_reel - marge_budget
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Écart Ventes", f"€{ecart_ventes:,.0f}",
                         f"{(ecart_ventes/ventes_budget*100):+.1f}%")
            with col2:
                st.metric("Écart Coûts", f"€{ecart_couts:,.0f}",
                         f"{(ecart_couts/couts_budget*100):+.1f}%")
            with col3:
                st.metric("Écart Marge", f"€{ecart_marge:,.0f}",
                         f"{(ecart_marge/marge_budget*100):+.1f}%")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Ventes", "Coûts", "Marge"],
                               y=[ecart_ventes, ecart_couts, ecart_marge],
                               marker_color=['blue', 'red', 'green']))
            fig.update_layout(title="Analyse des Écarts",
                             yaxis_title="Montant (€)")
            st.plotly_chart(fig, use_container_width=True)
    cause_selection = st.selectbox("Type d'écart à analyser:",
                                  ["Écart sur ventes", "Écart sur coûts", "Écart sur marge"])
    if cause_selection == "Écart sur ventes":
        st.write("**Causes possibles :**")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("❌ Conjoncture économique défavorable")
            st.checkbox("📉 Baisse de la demande")
            st.checkbox("🎯 Erreur de prévision")
        with col2:
            st.checkbox("🏪 Concurrence accrue")
            st.checkbox("👥 Problèmes de force de vente")
            st.checkbox("📱 Défauts du produit")
    elif cause_selection == "Écart sur coûts":
        st.write("**Causes possibles :**")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("📈 Hausse des prix matières premières")
            st.checkbox("⚡ Augmentation des coûts énergétiques")
            st.checkbox("👥 Hausse des salaires")
        with col2:
            st.checkbox("🏭 Baisse de productivité")
            st.checkbox("📦 Problèmes d'approvisionnement")
            st.checkbox("🔧 Pannes techniques")
    st.subheader("📄 Génération de Rapport d'Analyse")
    with st.form("rapport_analyse"):
        periode = st.text_input("Période analysée:", "Janvier 2024")
        causes_identifiees = st.text_area("Causes identifiées:")
        actions_proposees = st.text_area("Actions correctives proposées:")
        responsable = st.text_input("Responsable suivi:")
        submitted = st.form_submit_button("Générer le rapport")
        if submitted:
            rapport_content = f"""
            RAPPORT D'ANALYSE DES ÉCARTS - {periode}

            ÉCARTS CALCULÉS:
            - Ventes : €{ecart_ventes:,.0f} ({(ecart_ventes/ventes_budget*100) if ventes_budget != 0 else 0:+.1f}%)
            - Coûts : €{ecart_couts:,.0f} ({(ecart_couts/couts_budget*100) if couts_budget != 0 else 0:+.1f}%)
            - Marge : €{ecart_marge:,.0f} ({(ecart_marge/marge_budget*100) if marge_budget != 0 else 0:+.1f}%)

            CAUSES IDENTIFIÉES:
            {causes_identifiees}

            ACTIONS CORRECTIVES:
            {actions_proposees}

            RESPONSABLE SUIVI: {responsable}

            Date: {datetime.now().strftime('%d/%m/%Y')}
            """
            pdf = generate_pdf(rapport_content, f"Rapport d'Analyse des Écarts - {periode}")
            st.markdown("""
            <div class="export-box">
            <h4>📥 Export du rapport</h4>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Télécharger en PDF",
                data=pdf,
                file_name=f"rapport_analyse_{periode.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

elif chapitre == "🏗️ Évaluation d'Investissement":
    st.markdown('<div class="section-header">🏗️ Méthodes d\'Évaluation d\'Investissement</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="concept-card">
    <h4>🎯 Les trois méthodes principales</h4>
    <div class="formula-box">
    <strong>1. Valeur Actuelle Nette (VAN)</strong><br>
    VAN = Σ [Ft / (1 + i)^t] - I₀<br>
    Règle : VAN > 0 → Projet acceptable
    </div>
    <div class="formula-box">
    <strong>2. Taux de Rendement Interne (TRI)</strong><br>
    TRI = i tel que VAN = 0<br>
    Règle : TRI > Coût du capital → Projet acceptable
    </div>
    <div class="formula-box">
    <strong>3. Délai de Récupération (Payback)</strong><br>
    Temps pour récupérer l'investissement initial<br>
    Règle : Plus court = Moins risqué
    </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Caractéristiques de l'investissement**")
        investissement_initial = st.number_input("Investissement initial (€):", 10000, 1000000, 100000)
        duree_projet = st.slider("Durée du projet (années):", 1, 10, 5)
        taux_actualisation = st.slider("Taux d'actualisation (%):", 1.0, 20.0, 8.0)
    with col2:
        st.write("**Flux de trésorerie annuels**")
        flux = []
        for i in range(duree_projet):
            flux.append(st.number_input(f"Flux année {i+1} (€):", -50000, 500000, 30000 + i*5000))
    if st.button("Évaluer la rentabilité"):
        van = -investissement_initial
        for annee, flux_annee in enumerate(flux, 1):
            van += flux_annee / ((1 + taux_actualisation/100) ** annee)
        def calcul_tri(investissement, flux_list, duree):
            tri_min, tri_max = 0.0, 100.0
            for _ in range(100):
                tri_test = (tri_min + tri_max) / 2
                van_test = -investissement
                for annee, f in enumerate(flux_list, 1):
                    van_test += f / ((1 + tri_test/100) ** annee)
                if abs(van_test) < 100:
                    return tri_test
                elif van_test > 0:
                    tri_min = tri_test
                else:
                    tri_max = tri_test
            return (tri_min + tri_max) / 2
        tri = calcul_tri(investissement_initial, flux, duree_projet)
        cumul_flux = -investissement_initial
        payback = None
        for annee, flux_annee in enumerate(flux, 1):
            cumul_flux += flux_annee
            if cumul_flux >= 0 and payback is None:
                if annee == 1:
                    payback = 1
                else:
                    payback = annee - 1 + (-(cumul_flux - flux_annee) / flux_annee)
        col1, col2, col3 = st.columns(3)
        with col1:
            couleur_van = "green" if van > 0 else "red"
            st.metric("VAN", f"€{van:,.0f}",
                     "Rentable" if van > 0 else "Non rentable")
        with col2:
            st.metric("TRI", f"{tri:.1f}%")
        with col3:
            if payback:
                st.metric("Payback", f"{payback:.1f} ans")
        if van > 0 and tri > taux_actualisation:
            st.success("✅ **RECOMMANDATION** : Le projet est rentable et peut être accepté")
        else:
            st.error("❌ **RECOMMANDATION** : Le projet n'est pas suffisamment rentable")
        annees = list(range(duree_projet + 1))
        flux_cumules = [-investissement_initial]
        for i, f in enumerate(flux):
            flux_cumules.append(flux_cumules[-1] + f)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=annees, y=flux_cumules, mode='lines+markers', name='Flux cumulés'))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Évolution des flux de trésorerie cumulés",
                         xaxis_title="Années", yaxis_title="Flux cumulés (€)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="montecarlo-box">
        <h4>🎲 Simulation Monte Carlo (Analyse de Risque)</h4>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Lancer la simulation Monte Carlo"):
            results = monte_carlo_simulation(investissement_initial, flux, duree_projet)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=results, nbinsx=30, name='Distribution VAN'))
            fig.add_vline(x=van, line_dash="dash", line_color="red", annotation_text="VAN déterministe")
            fig.update_layout(title="Distribution des VAN (Simulation Monte Carlo)",
                             xaxis_title="VAN (€)", yaxis_title="Fréquence")
            st.plotly_chart(fig, use_container_width=True)
            van_moyen = np.mean(results)
            van_mediane = np.median(results)
            van_p10 = np.percentile(results, 10)
            van_p90 = np.percentile(results, 90)
            st.markdown(f"""
            <div class="success-box">
            <h4>Résultats de la Simulation :</h4>
            <p><strong>VAN moyen :</strong> {van_moyen:,.0f}€</p>
            <p><strong>VAN médian :</strong> {van_mediane:,.0f}€</p>
            <p><strong>VAN P10 (scénario pessimiste) :</strong> {van_p10:,.0f}€</p>
            <p><strong>VAN P90 (scénario optimiste) :</strong> {van_p90:,.0f}€</p>
            <p><strong>Probabilité VAN > 0 :</strong> {sum(1 for r in results if r > 0)/len(results):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        data = {
            "Indicateur": ["VAN", "TRI", "Payback", "VAN moyen (Monte Carlo)", "VAN médian (Monte Carlo)"],
            "Valeur": [f"€{van:,.0f}", f"{tri:.1f}%", f"{payback:.1f} ans", f"€{van_moyen:,.0f}", f"€{van_mediane:,.0f}"]
        }
        df_export = pd.DataFrame(data)
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.markdown("""
        <div class="export-box">
        <h4>📥 Export des résultats</h4>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Télécharger en CSV",
            data=csv,
            file_name="evaluation_investissement.csv",
            mime="text/csv"
        )

elif chapitre == "💸 Budget de Trésorerie":
    st.markdown('<div class="section-header">💸 Élaboration du Budget de Trésorerie</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="concept-card">
    <h4>🎯 Objectifs du budget de trésorerie</h4>
    <ul>
        <li><strong>Anticiper</strong> les besoins de financement</li>
        <li><strong>Éviter</strong> les situations de découvert</li>
        <li><strong>Optimiser</strong> la gestion des liquidités</li>
        <li><strong>Planifier</strong> les investissements</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Paramètres de base**")
        tresorerie_initial = st.number_input("Trésorerie initiale (€):", 0, 500000, 50000)
        delai_client = st.slider("Délai paiement clients (jours):", 0, 90, 30)
        delai_fournisseur = st.slider("Délai paiement fournisseurs (jours):", 0, 90, 60)
    with col2:
        st.write("**Charges récurrentes**")
        salaires = st.number_input("Salaires mensuels (€):", 0, 200000, 50000)
        charges_fixes = st.number_input("Charges fixes (€):", 0, 50000, 15000)
        remboursement = st.number_input("Remboursement emprunt (€):", 0, 50000, 10000)
    if st.button("Générer le budget de trésorerie"):
        mois = [f"Mois {i+1}" for i in range(12)]
        data_tresorerie = []
        tresorerie_courante = tresorerie_initial
        for i, mois_nom in enumerate(mois):
            ventes = 80000 + np.random.normal(0, 10000)
            achats = 45000 + np.random.normal(0, 5000)
            encaissements = ventes * 0.7
            decaissements = (achats * 0.6 + salaires + charges_fixes + remboursement)
            flux_net = encaissements - decaissements
            tresorerie_courante += flux_net
            data_tresorerie.append({
                'Mois': mois_nom,
                'Encaissements': encaissements,
                'Décaissements': decaissements,
                'Flux Net': flux_net,
                'Trésorerie Cumulée': tresorerie_courante
            })
        df_tresorerie = pd.DataFrame(data_tresorerie)
        st.subheader("📊 Tableau de Trésorerie Prévisionnel")
        st.dataframe(df_tresorerie.style.format({
            'Encaissements': '€{:,.0f}',
            'Décaissements': '€{:,.0f}',
            'Flux Net': '€{:,.0f}',
            'Trésorerie Cumulée': '€{:,.0f}'
        }), use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tresorerie['Mois'], y=df_tresorerie['Trésorerie Cumulée'],
                               name='Trésorerie Cumulée', line=dict(color='green', width=3)))
        fig.add_trace(go.Bar(x=df_tresorerie['Mois'], y=df_tresorerie['Flux Net'],
                           name='Flux Net Mensuel', marker_color='orange'))
        fig.update_layout(title='Évolution de la Trésorerie sur 12 mois',
                         barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
        seuil_alerte = st.number_input("Seuil d'alerte trésorerie (€):", 0, 50000, 10000)
        mois_critiques = [m for m in data_tresorerie if m['Trésorerie Cumulée'] < seuil_alerte]
        if mois_critiques:
            st.warning(f"🚨 **Alertes Trésorerie** : {len(mois_critiques)} mois sous le seuil")
            for mc in mois_critiques:
                st.write(f"- {mc['Mois']} : €{mc['Trésorerie Cumulée']:,.0f}")
        csv = df_tresorerie.to_csv(index=False).encode('utf-8')
        st.markdown("""
        <div class="export-box">
        <h4>📥 Export des résultats</h4>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Télécharger en CSV",
            data=csv,
            file_name="budget_tresorerie.csv",
            mime="text/csv"
        )

elif chapitre == "📈 Tableaux de Bord":
    st.markdown('<div class="section-header">📈 Tableaux de Bord de Pilotage</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="concept-card">
    <h4>🎯 Les 5 tableaux de bord essentiels</h4>
    1. **📊 Tableau de bord commercial** : Ventes, marges, portefeuille clients
    2. **🏭 Tableau de bord production** : Productivité, qualité, coûts
    3. **💰 Tableau de bord financier** : Rentabilité, trésorerie, équilibre
    4. **👥 Tableau de bord RH** : Productivité, turnover, compétences
    5. **🎯 Tableau de bord stratégique** : KPIs stratégiques, objectifs long terme
    </div>
    """, unsafe_allow_html=True)
    type_tableau = st.selectbox("Type de tableau de bord:",
                               ["Commercial", "Production", "Financier", "RH", "Stratégique"])
    if type_tableau == "Commercial":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("📈 Chiffre d'affaires")
            st.checkbox("💰 Marge commerciale")
            st.checkbox("📊 Taux de marge")
        with col2:
            st.checkbox("👥 Portefeuille clients")
            st.checkbox("🎯 Parts de marché")
            st.checkbox("📦 Volume des ventes")
        with col3:
            st.checkbox("📱 Canaux de distribution")
            st.checkbox("⭐ Satisfaction client")
            st.checkbox("🔄 Taux de fidélisation")
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    data_demo = pd.DataFrame({
        'Mois': dates,
        'Ventes': np.random.normal(100000, 15000, 12),
        'Coûts': np.random.normal(70000, 8000, 12),
        'Production': np.random.normal(5000, 500, 12)
    })
    data_demo['Marge'] = data_demo['Ventes'] - data_demo['Coûts']
    col1, col2 = st.columns(2)
    with col1:
        fig_ventes = px.line(data_demo, x='Mois', y='Ventes',
                           title='Évolution des Ventes')
        st.plotly_chart(fig_ventes, use_container_width=True)
    with col2:
        fig_marge = px.bar(data_demo, x='Mois', y='Marge',
                          title='Évolution de la Marge')
        st.plotly_chart(fig_marge, use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CA Moyen", f"€{data_demo['Ventes'].mean():,.0f}")
    with col2:
        st.metric("Marge Moyenne", f"€{data_demo['Marge'].mean():,.0f}")
    with col3:
        taux_marge = (data_demo['Marge'].mean() / data_demo['Ventes'].mean()) * 100
        st.metric("Taux de Marge", f"{taux_marge:.1f}%")
    with col4:
        st.metric("Productivité", f"{(data_demo['Production'].mean() / data_demo['Coûts'].mean()):.2f}")

elif chapitre == "🎓 Exercices & Cas Pratiques":
    st.markdown('<div class="section-header">🎓 Exercices et Cas Pratiques</div>', unsafe_allow_html=True)
    exercice = st.selectbox("Choisissez un exercice:",
                           ["Calcul de seuil de rentabilité",
                            "Analyse d'écarts complexes",
                            "Évaluation d'investissement",
                            "Construction budget complet",
                            "Optimisation des stocks"])

    if exercice == "Calcul de seuil de rentabilité":
        st.subheader("📊 Exercice : Calcul du Seuil de Rentabilité")
        st.markdown("""
        <div class="exercise-box">
        <h4>Énoncé :</h4>
        <p>Une entreprise produit et vend des widgets. Les données sont :</p>
        <ul>
            <li>Prix de vente unitaire : 50€</li>
            <li>Coût variable unitaire : 30€</li>
            <li>Charges fixes annuelles : 100 000€</li>
        </ul>
        <p><strong>Questions :</strong></p>
        <ol>
            <li>Calculer la marge sur coût variable unitaire</li>
            <li>Déterminer le seuil de rentabilité en quantité et en CA</li>
            <li>Calculer le point mort (date de rentabilité)</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("💡 Solution détaillée"):
            st.markdown("""
            <div class="success-box">
            <h4>Solution :</h4>
            <p><strong>1. Marge sur coût variable unitaire :</strong></p>
            <div class="formula-box">
            MCV unitaire = Prix vente - Coût variable = 50€ - 30€ = 20€
            </div>
            <p><strong>2. Seuil de rentabilité :</strong></p>
            <div class="formula-box">
            SR (quantité) = Charges fixes / MCV unitaire = 100 000€ / 20€ = 5 000 unités<br>
            SR (CA) = 5 000 × 50€ = 250 000€
            </div>
            <p><strong>3. Point mort :</strong></p>
            <p>Si l'entreprise vend 8 000 unités par an :</p>
            <div class="formula-box">
            Point mort = (5 000 / 8 000) × 12 mois = 7,5 mois<br>
            L'entreprise devient rentable fin juillet
            </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="calculator-box">
        <h4>🧮 Calculateur Interactif</h4>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            prix_vente = st.number_input("Prix de vente unitaire (€):", 10, 200, 50)
            cout_variable = st.number_input("Coût variable unitaire (€):", 1, 100, 30)
            charges_fixes = st.number_input("Charges fixes annuelles (€):", 1000, 1000000, 100000)
            ventes_annuelles = st.number_input("Ventes annuelles prévues (unités):", 1000, 50000, 8000)
        with col2:
            if st.button("Calculer"):
                mcv = prix_vente - cout_variable
                sr_quantite = charges_fixes / mcv
                sr_ca = sr_quantite * prix_vente
                point_mort = (sr_quantite / ventes_annuelles) * 12
                st.markdown(f"""
                <div class="success-box">
                <h4>Résultats :</h4>
                <p><strong>Marge sur coût variable :</strong> {mcv:.2f}€</p>
                <p><strong>Seuil de rentabilité (quantité) :</strong> {sr_quantite:,.0f} unités</p>
                <p><strong>Seuil de rentabilité (CA) :</strong> {sr_ca:,.0f}€</p>
                <p><strong>Point mort :</strong> {point_mort:.1f} mois</p>
                </div>
                """, unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, ventes_annuelles], y=[0, ventes_annuelles*prix_vente],
                                       name="Chiffre d'affaires", line=dict(color='green')))
                fig.add_trace(go.Scatter(x=[0, ventes_annuelles], y=[charges_fixes, charges_fixes + ventes_annuelles*cout_variable],
                                       name="Coûts totaux", line=dict(color='red')))
                fig.add_vline(x=sr_quantite, line_dash="dash", line_color="blue")
                fig.update_layout(title="Seuil de Rentabilité",
                                 xaxis_title="Quantité", yaxis_title="Montant (€)")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="variant-box">
        <h4>🔄 Variante pour approfondir</h4>
        <p>Que se passe-t-il si :</p>
        <ul>
            <li>Le prix de vente augmente de 10% ?</li>
            <li>Les charges fixes augmentent de 20 000€ ?</li>
            <li>Le coût variable diminue de 5€ ?</li>
        </ul>
        <p><strong>Question supplémentaire :</strong> Quel devrait être le prix de vente pour atteindre un seuil de rentabilité de 4 000 unités ?</p>
        </div>
        """, unsafe_allow_html=True)

    elif exercice == "Analyse d'écarts complexes":
        st.subheader("🔍 Exercice : Analyse d'Écarts Complexes")
        st.markdown("""
        <div class="exercise-box">
        <h4>Énoncé :</h4>
        <p>Pour un produit donné, vous disposez des informations suivantes :</p>
        <table>
        <tr><th></th><th>Budget</th><th>Réel</th></tr>
        <tr><td>Quantité produite</td><td>1 000 unités</td><td>1 200 unités</td></tr>
        <tr><td>Heures de main d'œuvre</td><td>500 heures</td><td>550 heures</td></tr>
        <tr><td>Coût horaire</td><td>25€/h</td><td>28€/h</td></tr>
        <tr><td>Coût matières/unité</td><td>15€</td><td>16€</td></tr>
        </table>
        <p><strong>Questions :</strong></p>
        <ol>
            <li>Calculer l'écart total sur coûts de production</li>
            <li>Décomposer l'écart en écarts sur quantité et sur prix</li>
            <li>Analyser les causes possibles</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("💡 Solution détaillée"):
            st.markdown("""
            <div class="success-box">
            <h4>Solution :</h4>
            <p><strong>1. Écart total :</strong></p>
            <div class="formula-box">
            Coût budget = (1 000 × 15€) + (500 × 25€) = 15 000€ + 12 500€ = 27 500€<br>
            Coût réel = (1 200 × 16€) + (550 × 28€) = 19 200€ + 15 400€ = 34 600€<br>
            Écart total = 34 600€ - 27 500€ = 7 100€ (défavorable)
            </div>
            <p><strong>2. Décomposition :</strong></p>
            <div class="formula-box">
            <strong>Écart sur quantité :</strong><br>
            Matières : (1 200 - 1 000) × 15€ = 3 000€<br>
            Main d'œuvre : (550 - 500) × 25€ = 1 250€<br>
            Total écart quantité = 4 250€
            <strong>Écart sur prix :</strong><br>
            Matières : 1 200 × (16€ - 15€) = 1 200€<br>
            Main d'œuvre : 550 × (28€ - 25€) = 1 650€<br>
            Total écart prix = 2 850€
            </div>
            <p><strong>3. Analyse des causes :</strong></p>
            <ul>
                <li>Écart quantité défavorable : Production supérieure au budget</li>
                <li>Écart prix défavorable : Hausse du coût horaire et des matières</li>
                <li>À investiguer : Productivité de la main d'œuvre</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="calculator-box">
        <h4>🧮 Calculateur Interactif</h4>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Données Budgétées**")
            qte_budget = st.number_input("Quantité produite (budget):", 100, 5000, 1000)
            heures_budget = st.number_input("Heures de main d'œuvre (budget):", 100, 2000, 500)
            cout_horaire_budget = st.number_input("Coût horaire (budget, €):", 10, 50, 25)
            cout_matiere_budget = st.number_input("Coût matière/unité (budget, €):", 1, 50, 15)
        with col2:
            st.write("**Données Réelles**")
            qte_reel = st.number_input("Quantité produite (réel):", 100, 5000, 1200)
            heures_reel = st.number_input("Heures de main d'œuvre (réel):", 100, 2000, 550)
            cout_horaire_reel = st.number_input("Coût horaire (réel, €):", 10, 50, 28)
            cout_matiere_reel = st.number_input("Coût matière/unité (réel, €):", 1, 50, 16)
        if st.button("Analyser les écarts"):
            cout_budget = (qte_budget * cout_matiere_budget) + (heures_budget * cout_horaire_budget)
            cout_reel = (qte_reel * cout_matiere_reel) + (heures_reel * cout_horaire_reel)
            ecart_total = cout_reel - cout_budget
            ecart_qte_matiere = (qte_reel - qte_budget) * cout_matiere_budget
            ecart_qte_maindoeuvre = (heures_reel - heures_budget) * cout_horaire_budget
            ecart_prix_matiere = qte_reel * (cout_matiere_reel - cout_matiere_budget)
            ecart_prix_maindoeuvre = heures_reel * (cout_horaire_reel - cout_horaire_budget)
            st.markdown(f"""
            <div class="success-box">
            <h4>Résultats :</h4>
            <p><strong>Coût budgété :</strong> {cout_budget:,.0f}€</p>
            <p><strong>Coût réel :</strong> {cout_reel:,.0f}€</p>
            <p><strong>Écart total :</strong> {ecart_total:,.0f}€ ({'défavorable' if ecart_total > 0 else 'favorable'})</p>
            <p><strong>Écart sur quantité :</strong> {ecart_qte_matiere + ecart_qte_maindoeuvre:,.0f}€</p>
            <p><strong>Écart sur prix :</strong> {ecart_prix_matiere + ecart_prix_maindoeuvre:,.0f}€</p>
            </div>
            """, unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Coût budgété", "Coût réel"],
                               y=[cout_budget, cout_reel],
                               name="Coûts totaux",
                               marker_color=['green', 'red']))
            fig.update_layout(title="Comparaison Coûts Budgétés vs Réels",
                             yaxis_title="Montant (€)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="variant-box">
        <h4>🔄 Variante pour approfondir</h4>
        <p>Et si on ajoutait :</p>
        <ul>
            <li>Un écart sur volume de production (1 200 vs 1 000 unités) ?</li>
            <li>Un écart sur rendement (heures théoriques vs heures réelles) ?</li>
            <li>Un écart sur mix (plusieurs produits avec des coûts différents) ?</li>
        </ul>
        <p><strong>Question supplémentaire :</strong> Comment prioriser les actions correctives ?</p>
        </div>
        """, unsafe_allow_html=True)

    elif exercice == "Évaluation d'investissement":
        st.subheader("🏗️ Cas Pratique : Évaluation d'un Projet d'Investissement")
        st.markdown("""
        <div class="exercise-box">
        <h4>Énoncé :</h4>
        <p>L'entreprise TechInnov envisage d'investir dans une nouvelle ligne de production pour un produit innovant.
        Voici les données disponibles :</p>
        <ul>
            <li>Investissement initial : 500 000€</li>
            <li>Durée de vie du projet : 5 ans</li>
            <li>Flux de trésorerie annuels estimés : 120 000€, 150 000€, 180 000€, 200 000€, 150 000€</li>
            <li>Taux d'actualisation : 10%</li>
            <li>Valeur résiduelle en fin de projet : 50 000€</li>
        </ul>
        <p><strong>Questions :</strong></p>
        <ol>
            <li>Calculer la VAN du projet</li>
            <li>Déterminer le TRI</li>
            <li>Calculer le délai de récupération (Payback)</li>
            <li>Faire une recommandation</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("💡 Solution détaillée"):
            st.markdown("""
            <div class="success-box">
            <h4>Solution :</h4>
            <p><strong>1. Calcul de la VAN :</strong></p>
            <div class="formula-box">
            VAN = -500 000 + 120 000/(1.1) + 150 000/(1.1)² + 180 000/(1.1)³ + 200 000/(1.1)⁴ + (150 000 + 50 000)/(1.1)⁵<br>
            VAN = -500 000 + 109 091 + 123 967 + 135 135 + 136 603 + 123 572<br>
            VAN = 128 368€
            </div>
            <p><strong>2. Calcul du TRI :</strong></p>
            <p>Le TRI est le taux qui annule la VAN. Par approximation :</p>
            <div class="formula-box">
            TRI ≈ 14.5%
            </div>
            <p><strong>3. Délai de récupération :</strong></p>
            <div class="formula-box">
            • Année 1 : 120 000€ (cumul : 120 000€)<br>
            • Année 2 : 150 000€ (cumul : 270 000€)<br>
            • Année 3 : 180 000€ (cumul : 450 000€)<br>
            • Année 4 : 200 000€ (cumul : 650 000€)<br>
            Le payback est atteint entre la 3ème et la 4ème année.<br>
            Précisément : 3 + (500 000 - 450 000)/200 000 = 3.25 années
            </div>
            <p><strong>4. Recommandation :</strong></p>
            <ul>
                <li>VAN > 0 : Projet rentable</li>
                <li>TRI (14.5%) > Taux d'actualisation (10%)</li>
                <li>Payback acceptable (3.25 ans)</li>
                <li>→ <strong>Accepter le projet</strong></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="calculator-box">
        <h4>🧮 Calculateur Interactif</h4>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            investissement = st.number_input("Investissement initial (€):", 10000, 1000000, 500000)
            duree = st.slider("Durée du projet (années):", 1, 10, 5)
            taux = st.slider("Taux d'actualisation (%):", 1, 20, 10)
            valeur_residuelle = st.number_input("Valeur résiduelle (€):", 0, 200000, 50000)
            flux = []
            for i in range(duree):
                flux.append(st.number_input(f"Flux année {i+1} (€):", -100000, 500000, 120000 + i*30000))
        with col2:
            if st.button("Évaluer le projet"):
                van = -investissement
                for annee, f in enumerate(flux, 1):
                    van += f / ((1 + taux/100) ** annee)
                van += valeur_residuelle / ((1 + taux/100) ** duree)
                def calcul_tri(inv, flux_list, duree, vr):
                    tri_min, tri_max = 0.0, 50.0
                    for _ in range(100):
                        tri_test = (tri_min + tri_max) / 2
                        van_test = -inv
                        for annee, f in enumerate(flux_list, 1):
                            van_test += f / ((1 + tri_test/100) ** annee)
                        van_test += vr / ((1 + tri_test/100) ** duree)
                        if abs(van_test) < 100:
                            return tri_test
                        elif van_test > 0:
                            tri_min = tri_test
                        else:
                            tri_max = tri_test
                    return (tri_min + tri_max) / 2
                tri = calcul_tri(investissement, flux, duree, valeur_residuelle)
                cumul = -investissement
                payback = None
                for annee, f in enumerate(flux, 1):
                    cumul += f
                    if cumul >= 0 and payback is None:
                        if annee == 1:
                            payback = 1
                        else:
                            payback = annee - 1 + (-(cumul - f) / f)
                st.markdown(f"""
                <div class="success-box">
                <h4>Résultats :</h4>
                <p><strong>VAN :</strong> {van:,.0f}€ ({'Rentable' if van > 0 else 'Non rentable'})</p>
                <p><strong>TRI :</strong> {tri:.1f}%</p>
                <p><strong>Payback :</strong> {payback:.1f} ans</p>
                </div>
                """, unsafe_allow_html=True)
                annees = list(range(duree + 1))
                flux_cumules = [-investissement]
                for i, f in enumerate(flux):
                    flux_cumules.append(flux_cumules[-1] + f)
                flux_cumules[-1] += valeur_residuelle
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=annees, y=flux_cumules, mode='lines+markers', name='Flux cumulés'))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(title="Flux de Trésorerie Cumulés",
                                 xaxis_title="Années", yaxis_title="Montant (€)")
                st.plotly_chart(fig, use_container_width=True)
                if st.button("Analyse de risque (Monte Carlo)"):
                    results = monte_carlo_simulation(investissement, flux, duree)
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(x=results, nbinsx=30, name='Distribution VAN'))
                    fig_mc.add_vline(x=van, line_dash="dash", line_color="red", annotation_text="VAN déterministe")
                    fig_mc.update_layout(title="Distribution des VAN (Simulation Monte Carlo)",
                                       xaxis_title="VAN (€)", yaxis_title="Fréquence")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    van_moyen = np.mean(results)
                    van_mediane = np.median(results)
                    van_p10 = np.percentile(results, 10)
                    van_p90 = np.percentile(results, 90)
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>Résultats de la Simulation :</h4>
                    <p><strong>VAN moyen :</strong> {van_moyen:,.0f}€</p>
                    <p><strong>VAN médian :</strong> {van_mediane:,.0f}€</p>
                    <p><strong>VAN P10 (scénario pessimiste) :</strong> {van_p10:,.0f}€</p>
                    <p><strong>VAN P90 (scénario optimiste) :</strong> {van_p90:,.0f}€</p>
                    <p><strong>Probabilité VAN > 0 :</strong> {sum(1 for r in results if r > 0)/len(results):.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("""
        <div class="variant-box">
        <h4>🔄 Variante pour approfondir</h4>
        <p>Et si on considérait :</p>
        <ul>
            <li>Un taux d'actualisation variable selon les années ?</li>
            <li>Des flux négatifs en début de projet ?</li>
            <li>Une option d'abandon après 3 ans ?</li>
        </ul>
        <p><strong>Question supplémentaire :</strong> Comment intégrer le risque dans l'évaluation ?</p>
        </div>
        """, unsafe_allow_html=True)

    elif exercice == "Construction budget complet":
        st.subheader("📊 Cas Pratique : Construction d'un Budget Complet")
        st.markdown("""
        <div class="exercise-box">
        <h4>Énoncé :</h4>
        <p>L'entreprise EcoBois fabrique des meubles en bois. Pour l'année N+1, les prévisions sont :</p>
        <ul>
            <li>Ventes prévues : 10 000 unités à 200€/unité</li>
            <li>Stock initial de produits finis : 1 000 unités</li>
            <li>Stock final souhaité : 1 500 unités</li>
            <li>Coût matière première : 80€/unité</li>
            <li>Main d'œuvre : 30€/unité (2h à 15€/h)</li>
            <li>Charges fixes : 500 000€</li>
        </ul>
        <p><strong>Questions :</strong></p>
        <ol>
            <li>Établir le budget des ventes</li>
            <li>Établir le budget de production</li>
            <li>Établir le budget des approvisionnements (stock initial matières = 50 000€, stock final souhaité = 60 000€)</li>
            <li>Calculer le résultat prévisionnel</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("💡 Solution détaillée"):
            st.markdown("""
            <div class="success-box">
            <h4>Solution :</h4>
            <p><strong>1. Budget des ventes :</strong></p>
            <div class="formula-box">
            CA prévisionnel = 10 000 × 200€ = 2 000 000€
            </div>
            <p><strong>2. Budget de production :</strong></p>
            <div class="formula-box">
            Quantité à produire = Ventes + Stock final - Stock initial<br>
            = 10 000 + 1 500 - 1 000 = 10 500 unités<br><br>
            Coût de production :<br>
            • Matières premières : 10 500 × 80€ = 840 000€<br>
            • Main d'œuvre : 10 500 × 30€ = 315 000€<br>
            • Charges fixes : 500 000€<br>
            → Coût total = 1 655 000€
            </div>
            <p><strong>3. Budget des approvisionnements :</strong></p>
            <div class="formula-box">
            Besoin en matières = 10 500 × 80€ = 840 000€<br>
            Achats = Besoin + Stock final - Stock initial<br>
            = 840 000 + 60 000 - 50 000 = 850 000€
            </div>
            <p><strong>4. Résultat prévisionnel :</strong></p>
            <div class="formula-box">
            Résultat = CA - Coût de production<br>
            = 2 000 000 - 1 655 000 = 345 000€
            </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="calculator-box">
        <h4>🧮 Calculateur Interactif</h4>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            ventes_unites = st.number_input("Ventes prévues (unités):", 1000, 50000, 10000)
            prix_vente = st.number_input("Prix de vente unitaire (€):", 50, 500, 200)
            stock_initial_pf = st.number_input("Stock initial produits finis (unités):", 0, 5000, 1000)
            stock_final_pf = st.number_input("Stock final souhaité produits finis (unités):", 0, 5000, 1500)
            cout_matiere = st.number_input("Coût matière première/unité (€):", 10, 200, 80)
        with col2:
            cout_maindoeuvre = st.number_input("Coût main d'œuvre/unité (€):", 10, 100, 30)
            charges_fixes = st.number_input("Charges fixes (€):", 10000, 1000000, 500000)
            stock_initial_mp = st.number_input("Stock initial matières premières (€):", 10000, 200000, 50000)
            stock_final_mp = st.number_input("Stock final souhaité matières premières (€):", 10000, 200000, 60000)
        if st.button("Calculer le budget"):
            ca = ventes_unites * prix_vente
            qte_produire = ventes_unites + stock_final_pf - stock_initial_pf
            cout_prod_matiere = qte_produire * cout_matiere
            cout_prod_maindoeuvre = qte_produire * cout_maindoeuvre
            cout_prod_total = cout_prod_matiere + cout_prod_maindoeuvre + charges_fixes
            besoin_mp = qte_produire * cout_matiere
            achats_mp = besoin_mp + stock_final_mp - stock_initial_mp
            resultat = ca - cout_prod_total
            st.markdown(f"""
            <div class="success-box">
            <h4>Résultats :</h4>
            <p><strong>Budget des ventes :</strong> {ca:,.0f}€</p>
            <p><strong>Quantité à produire :</strong> {qte_produire:,.0f} unités</p>
            <p><strong>Coût de production :</strong> {cout_prod_total:,.0f}€</p>
            <p><strong>Budget approvisionnements :</strong> {achats_mp:,.0f}€</p>
            <p><strong>Résultat prévisionnel :</strong> {resultat:,.0f}€</p>
            </div>
            """, unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Chiffre d'affaires", "Coût de production"],
                               y=[ca, cout_prod_total],
                               name="Montants",
                               marker_color=['green', 'red']))
            fig.update_layout(title="Budget Prévisionnel",
                             yaxis_title="Montant (€)")
            st.plotly_chart(fig, use_container_width=True)
            data = {
                "Poste": ["Chiffre d'affaires", "Coût de production", "Budget approvisionnements", "Résultat prévisionnel"],
                "Valeur": [f"{ca:,.0f}€", f"{cout_prod_total:,.0f}€", f"{achats_mp:,.0f}€", f"{resultat:,.0f}€"]
            }
            df_export = pd.DataFrame(data)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.markdown("""
            <div class="export-box">
            <h4>📥 Export des résultats</h4>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Télécharger en CSV",
                data=csv,
                file_name="budget_complet.csv",
                mime="text/csv"
            )
        st.markdown("""
        <div class="variant-box">
        <h4>🔄 Variante pour approfondir</h4>
        <p>Et si on ajoutait :</p>
        <ul>
            <li>Un budget des investissements (achat d'une nouvelle machine) ?</li>
            <li>Un budget de trésorerie (délais de paiement clients/fournisseurs) ?</li>
            <li>Une analyse de sensibilité (variation des ventes de ±10%) ?</li>
        </ul>
        <p><strong>Question supplémentaire :</strong> Comment intégrer ce budget dans un tableau de bord de pilotage ?</p>
        </div>
        """, unsafe_allow_html=True)

    elif exercice == "Optimisation des stocks":
        st.subheader("📦 Cas Pratique : Optimisation des Stocks avec le Modèle de Wilson")
        st.markdown("""
        <div class="exercise-box">
        <h4>Énoncé :</h4>
        <p>L'entreprise StockOpt veut optimiser la gestion des stocks d'un composant électronique.
        Les données sont :</p>
        <ul>
            <li>Demande annuelle : 12 000 unités</li>
            <li>Coût de passation d'une commande : 45€</li>
            <li>Coût de possession unitaire annuel : 1.5€</li>
            <li>Délai de livraison : 7 jours</li>
            <li>Nombre de jours ouvrés par an : 250</li>
        </ul>
        <p><strong>Questions :</strong></p>
        <ol>
            <li>Calculer la quantité économique de commande (QEC)</li>
            <li>Déterminer le nombre de commandes par an</li>
            <li>Calculer le stock de sécurité (pour couvrir 3 jours de consommation)</li>
            <li>Déterminer le point de commande</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("💡 Solution détaillée"):
            st.markdown("""
            <div class="success-box">
            <h4>Solution :</h4>
            <p><strong>1. Quantité économique de commande :</strong></p>
            <div class="formula-box">
            QEC = √(2 × 12 000 × 45 / 1.5) = √720 000 ≈ 849 unités
            </div>
            <p><strong>2. Nombre de commandes par an :</strong></p>
            <div class="formula-box">
            Nombre de commandes = 12 000 / 849 ≈ 14 commandes/an
            </div>
            <p><strong>3. Stock de sécurité :</strong></p>
            <div class="formula-box">
            Consommation journalière = 12 000 / 250 = 48 unités/jour<br>
            Stock de sécurité = 3 × 48 = 144 unités
            </div>
            <p><strong>4. Point de commande :</strong></p>
            <div class="formula-box">
            Point de commande = (7 × 48) + 144 = 480 unités
            </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="calculator-box">
        <h4>🧮 Calculateur Interactif</h4>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            demande = st.number_input("Demande annuelle (unités):", 1000, 100000, 12000)
            cout_commande = st.number_input("Coût de passation d'une commande (€):", 10, 200, 45)
            cout_possession = st.number_input("Coût de possession unitaire annuel (€):", 0.1, 10, 1.5)
            delai = st.number_input("Délai de livraison (jours):", 1, 30, 7)
            jours_ouvres = st.number_input("Nombre de jours ouvrés par an:", 200, 300, 250)
            stock_securite_jours = st.number_input("Stock de sécurité (jours de consommation):", 1, 10, 3)
        with col2:
            if st.button("Optimiser les stocks"):
                qec = math.sqrt((2 * demande * cout_commande) / cout_possession)
                n_commandes = demande / qec
                conso_journaliere = demande / jours_ouvres
                stock_securite = stock_securite_jours * conso_journaliere
                point_commande = (delai * conso_journaliere) + stock_securite
                st.markdown(f"""
                <div class="success-box">
                <h4>Résultats :</h4>
                <p><strong>Quantité économique (QEC) :</strong> {qec:.0f} unités</p>
                <p><strong>Nombre de commandes/an :</strong> {n_commandes:.1f}</p>
                <p><strong>Stock de sécurité :</strong> {stock_securite:.0f} unités</p>
                <p><strong>Point de commande :</strong> {point_commande:.0f} unités</p>
                </div>
                """, unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, qec, qec], y=[0, qec/2, 0],
                                       fill='tozeroy', name="Stock moyen"))
                fig.add_trace(go.Scatter(x=[0, point_commande, point_commande], y=[0, 0, qec],
                                       mode='lines', name="Point de commande"))
                fig.add_hline(y=stock_securite, line_dash="dash", line_color="red",
                            annotation_text="Stock de sécurité")
                fig.update_layout(title="Gestion Optimale des Stocks",
                                 xaxis_title="Quantité", yaxis_title="Niveau de stock")
                st.plotly_chart(fig, use_container_width=True)
                data = {
                    "Indicateur": ["QEC", "Nombre de commandes/an", "Stock de sécurité", "Point de commande"],
                    "Valeur": [f"{qec:.0f} unités", f"{n_commandes:.1f}", f"{stock_securite:.0f} unités", f"{point_commande:.0f} unités"]
                }
                df_export = pd.DataFrame(data)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.markdown("""
                <div class="export-box">
                <h4>📥 Export des résultats</h4>
                </div>
                """, unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Télécharger en CSV",
                    data=csv,
                    file_name="optimisation_stocks.csv",
                    mime="text/csv"
                )
        st.markdown("""
        <div class="variant-box">
        <h4>🔄 Variante pour approfondir</h4>
        <p>Et si on considérait :</p>
        <ul>
            <li>Des remises quantitatives (ex: -5% si commande > 1 000 unités) ?</li>
            <li>Une demande saisonnière (variation de ±20% selon les mois) ?</li>
            <li>Un taux de service de 95% (risque de rupture) ?</li>
        </ul>
        <p><strong>Question supplémentaire :</strong> Comment intégrer cette optimisation dans un ERP ?</p>
        </div>
        """, unsafe_allow_html=True)

elif chapitre == "🧠 Quiz de Validation":
    st.markdown('<div class="section-header">🧠 Quiz de Validation des Connaissances</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="quiz-box">
    <h4>🎯 Testez vos connaissances en contrôle de gestion !</h4>
    <p>Ce quiz couvre les principaux concepts abordés dans le guide.
    Sélectionnez la bonne réponse pour chaque question.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quiz questions and answers
    quiz_questions = [
        {
            "question": "1. Quel est l'objectif principal du contrôle de gestion ?",
            "options": [
                "A. Maximiser les profits à court terme",
                "B. Piloter la performance et aider à la prise de décision",
                "C. Remplacer la direction générale",
                "D. Gérer uniquement les aspects financiers"
            ],
            "answer": 1,
            "explanation": "Le contrôle de gestion a pour objectif principal de **piloter la performance** et d'**aider à la prise de décision** en fournissant des analyses et des outils de gestion."
        },
        {
            "question": "2. Quelle formule calcule le seuil de rentabilité en quantité ?",
            "options": [
                "A. Seuil = Charges fixes / Prix de vente unitaire",
                "B. Seuil = Charges fixes / Marge sur coût variable unitaire",
                "C. Seuil = Charges variables / Prix de vente unitaire",
                "D. Seuil = (Charges fixes + Charges variables) / Prix de vente unitaire"
            ],
            "answer": 1,
            "explanation": "Le seuil de rentabilité en quantité se calcule avec la formule : **Seuil = Charges fixes / Marge sur coût variable unitaire**. La marge sur coût variable est la différence entre le prix de vente et le coût variable unitaire."
        },
        {
            "question": "3. Que mesure le TRI (Taux de Rendement Interne) ?",
            "options": [
                "A. Le temps nécessaire pour récupérer l'investissement initial",
                "B. Le taux d'actualisation qui annule la VAN",
                "C. Le rendement moyen du marché",
                "D. Le coût du capital de l'entreprise"
            ],
            "answer": 1,
            "explanation": "Le **TRI** (Taux de Rendement Interne) est le **taux d'actualisation qui annule la VAN** (Valeur Actuelle Nette). Il représente le taux de rentabilité intrinsèque du projet."
        },
        {
            "question": "4. Dans le modèle de Wilson, que représente la QEC ?",
            "options": [
                "A. Quantité Economique de Commande",
                "B. Qualité Exigée par le Client",
                "C. Quotient d'Efficacité Commercial",
                "D. Quotient d'Equilibre des Coûts"
            ],
            "answer": 0,
            "explanation": "La **QEC** signifie **Quantité Economique de Commande**. Elle représente la quantité optimale à commander pour minimiser les coûts totaux de gestion des stocks (coûts de passation + coûts de possession)."
        },
        {
            "question": "5. Quel indicateur mesure l'écart relatif entre le réel et le budget ?",
            "options": [
                "A. Le R²",
                "B. Le MAPE (Mean Absolute Percentage Error)",
                "C. Le TRI",
                "D. Le Payback"
            ],
            "answer": 1,
            "explanation": "Le **MAPE** (Mean Absolute Percentage Error) est un indicateur qui mesure **l'écart relatif moyen entre les valeurs réelles et les prévisions**. Il est souvent utilisé pour évaluer la précision des modèles de prévision."
        },
        {
            "question": "6. Quelle méthode qualitative utilise un processus itératif avec des experts ?",
            "options": [
                "A. La régression linéaire",
                "B. La méthode Delphi",
                "C. Le lissage exponentiel",
                "D. La moyenne mobile"
            ],
            "answer": 1,
            "explanation": "La **méthode Delphi** est une méthode qualitative qui utilise un **processus itératif** avec des experts pour parvenir à un consensus sur une prévision ou une décision."
        },
        {
            "question": "7. Quel est l'objectif principal d'un budget de trésorerie ?",
            "options": [
                "A. Calculer les impôts à payer",
                "B. Anticiper les besoins de financement et éviter les découverts",
                "C. Déterminer les salaires des employés",
                "D. Fixer les objectifs de vente"
            ],
            "answer": 1,
            "explanation": "L'objectif principal d'un **budget de trésorerie** est d'**anticiper les besoins de financement** et d'**éviter les situations de découvert** en planifiant les encaissements et les décaissements."
        },
        {
            "question": "8. Quelle composante n'est PAS incluse dans une série temporelle ?",
            "options": [
                "A. Tendence (Trend)",
                "B. Saisonnalité (Seasonality)",
                "C. Aléatoire (Error)",
                "D. Taux d'actualisation"
            ],
            "answer": 3,
            "explanation": "Le **taux d'actualisation** n'est pas une composante d'une série temporelle. Les composantes classiques sont : **Tendence (Trend)**, **Saisonnalité (Seasonality)**, **Cyclique (Cycle)** et **Aléatoire (Error)**."
        },
        {
            "question": "9. Quel est le rôle du stock de sécurité dans la gestion des stocks ?",
            "options": [
                "A. Maximiser les coûts de possession",
                "B. Couvrir les variations de la demande et éviter les ruptures",
                "C. Réduire le nombre de commandes",
                "D. Augmenter le délai de livraison"
            ],
            "answer": 1,
            "explanation": "Le **stock de sécurité** a pour rôle de **couvrir les variations de la demande** et d'**éviter les ruptures de stock** en cas de délais de livraison imprévus ou de demande supérieure à la prévision."
        },
        {
            "question": "10. Quelle méthode utilise la formule Y = aX + b ?",
            "options": [
                "A. La méthode Delphi",
                "B. La régression linéaire",
                "C. Le lissage exponentiel",
                "D. La moyenne mobile"
            ],
            "answer": 1,
            "explanation": "La formule **Y = aX + b** est utilisée dans la **régression linéaire**, où **Y** est la variable dépendante, **X** la variable indépendante, **a** la pente et **b** l'ordonnée à l'origine."
        }
    ]

    # Quiz display and scoring
    score = 0
    for i, q in enumerate(quiz_questions):
        st.markdown(f"""
        <div class="quiz-box">
        <p><strong>Question {i+1} :</strong> {q["question"]}</p>
        </div>
        """, unsafe_allow_html=True)
        user_answer = st.radio(
            "Sélectionnez votre réponse :",
            q["options"],
            key=f"q{i}"
        )
        if user_answer[0] == q["options"][q["answer"]][0]:
            score += 1
        with st.expander("Voir l'explication"):
            st.markdown(f"""
            <div class="success-box">
            <p><strong>Explication :</strong> {q["explanation"]}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    if st.button("Valider le quiz"):
        st.markdown(f"""
        <div class="success-box">
        <h4>🎉 Résultats du Quiz</h4>
        <p>Vous avez obtenu <strong>{score}/10</strong> bonnes réponses.</p>
        <p><strong>Niveau :</strong>
        {"⭐⭐⭐ Expert" if score == 10 else
         "⭐⭐ Avancé" if score >= 7 else
         "⭐ Intermédiaire" if score >= 5 else
         "Débutant"} ({score*10}%)</p>
        </div>
        """, unsafe_allow_html=True)
        if score < 5:
            st.markdown("""
            <div class="variant-box">
            <p>📚 <strong>Conseil :</strong> Revoyez les chapitres sur les <strong>fondamentaux du contrôle de gestion</strong> et les <strong>méthodes d'analyse des écarts</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        elif score < 8:
            st.markdown("""
            <div class="variant-box">
            <p>📊 <strong>Conseil :</strong> Approfondissez les <strong>méthodes de prévision</strong> et les <strong>techniques d'optimisation</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="variant-box">
            <p>🎓 <strong>Félicitations !</strong> Vous maîtrisez les concepts clés. Passez aux <strong>cas pratiques avancés</strong> et explorez les <strong>simulations de risque</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

# Pied de page éducatif
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <strong>📚 Guide Complet Contrôle de Gestion</strong><br>
    Méthodes • Calculs • Applications • Exercices • Quiz<br>
    Développé pour la formation et la pratique professionnelle
    par Ibrahima Coumba Gueye Xataxeli
</div>
""", unsafe_allow_html=True)
