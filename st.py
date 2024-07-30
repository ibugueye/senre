import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="SEN-RE Dashboard", layout="wide")
st.title("Tableau de Bord - Société Sénégalaise de Réassurances")
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# URL de l'image
image_url = "https://i0.wp.com/sen-re.com/wp-content/uploads/2019/12/Logo-et-texte.jpg?resize=1536%2C410&ssl=1"

# Télécharger l'image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Afficher l'image
st.image(image, caption='Logo de la Société Sénégalaise de Réassurances (SEN-RE)', use_column_width=True)

# Fonction pour créer des projections
def create_projections(data, years):
    X = years[:len(data)].reshape(-1, 1)
    y = np.array(data).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = years[len(data):].reshape(-1, 1)
    future_y = model.predict(future_X)
    return future_y.flatten()

# Données
years = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
chiffre_affaires = [17029921, 16398887, 14544231, 14001133, 16428113]
fonds_propres = [7646768, 8007888, 8547092, 9513876, 10898977]
roe = [9.84, 9.50, 10.59, 14.66, 12.71]
provisions_techniques = [22436693, 21583960, 21767230, 20723063, 24906680]

# Créer les projections
ca_projections = create_projections(chiffre_affaires, years)
fp_projections = create_projections(fonds_propres, years)
roe_projections = create_projections(roe, years)
pt_projections = create_projections(provisions_techniques, years)

# Chiffre d'affaires
st.header("Chiffre d'affaires")
fig_ca = px.line(x=years, y=chiffre_affaires + list(ca_projections), 
                 labels={"x": "Année", "y": "Chiffre d'affaires (milliers de FCFA)"},
                 title="Chiffre d'affaires - Historique et Projections")
fig_ca.add_scatter(x=years[:5], y=chiffre_affaires, mode='markers', name='Données historiques')
fig_ca.add_scatter(x=years[5:], y=ca_projections, mode='markers', name='Projections')
st.plotly_chart(fig_ca)

# Fonds propres
st.header("Fonds propres")
fig_fp = px.line(x=years, y=fonds_propres + list(fp_projections), 
                 labels={"x": "Année", "y": "Fonds propres (milliers de FCFA)"},
                 title="Fonds propres - Historique et Projections")
fig_fp.add_scatter(x=years[:5], y=fonds_propres, mode='markers', name='Données historiques')
fig_fp.add_scatter(x=years[5:], y=fp_projections, mode='markers', name='Projections')
st.plotly_chart(fig_fp)

# ROE
st.header("ROE")
fig_roe = px.line(x=years, y=roe + list(roe_projections), 
                  labels={"x": "Année", "y": "ROE (%)"},
                  title="ROE - Historique et Projections")
fig_roe.add_scatter(x=years[:5], y=roe, mode='markers', name='Données historiques')
fig_roe.add_scatter(x=years[5:], y=roe_projections, mode='markers', name='Projections')
st.plotly_chart(fig_roe)

# Provisions techniques
st.header("Provisions techniques")
fig_pt = px.line(x=years, y=provisions_techniques + list(pt_projections), 
                 labels={"x": "Année", "y": "Provisions techniques (milliers de FCFA)"},
                 title="Provisions techniques - Historique et Projections")
fig_pt.add_scatter(x=years[:5], y=provisions_techniques, mode='markers', name='Données historiques')
fig_pt.add_scatter(x=years[5:], y=pt_projections, mode='markers', name='Projections')
st.plotly_chart(fig_pt)

# Tableau récapitulatif
st.header("Tableau récapitulatif")
recap_data = {
    "Année": years,
    "Chiffre d'affaires": chiffre_affaires + list(ca_projections),
    "Fonds propres": fonds_propres + list(fp_projections),
    "ROE (%)": roe + list(roe_projections),
    "Provisions techniques": provisions_techniques + list(pt_projections)
}
df_recap = pd.DataFrame(recap_data)
st.dataframe(df_recap)

# Chiffre d'affaires par zones
st.header("Chiffre d'affaires par zones (2018)")
ca_zones = {
    "Zones": ["AFRIQUE AUSTRALE", "AFRIQUE CENTRALE", "AFRIQUE L'EST O. INDIEN", "AFRIQUE DE L'OUEST", 
              "AFRIQUE DU NORD", "ASIE", "MOYEN ORIENT", "EUROPE"],
    "Chiffre d'affaires": [42275677, 423419940, 337752335, 13341334820, 1299947817, 781278249, 201115398, 988403],
    "Pourcentage": [0.26, 2.58, 2.06, 81.21, 7.91, 4.76, 1.22, 0.01]
}
df_ca_zones = pd.DataFrame(ca_zones)
fig_ca_zones = px.pie(df_ca_zones, values="Chiffre d'affaires", names='Zones', title="Répartition du chiffre d'affaires par zones")

st.dataframe(df_ca_zones)

# Chiffre d'affaires par branches
st.header("Chiffre d'affaires par branches (2018)")
ca_branches = {
    "Branches": ["VIE", "INCENDIE", "TRANSPORTS", "AUTOMOBILE", "RISQUES TECHNIQUES", "RISQUES DIVERS", "AVIATION"],
    "Chiffre d'affaires": [1033416886, 5640869619, 1353554278, 3023508564, 1200898531, 4081868025, 93996737],
    "Pourcentage": [6.29, 34.34, 8.24, 18.40, 7.31, 24.85, 0.57]
}
df_ca_branches = pd.DataFrame(ca_branches)
fig_ca_branches = px.pie(df_ca_branches, values="Chiffre d'affaires", names='Branches', title="Répartition du chiffre d'affaires par branches")

st.plotly_chart(fig_ca_branches)
st.dataframe(df_ca_branches)

# Ratios prudentiels
st.header("Ratios prudentiels (2018)")
ratios = {
    "Ratio": ["Primes Nettes / Fonds Propres", "Provisions Techniques / Fonds Propres", 
              "Provisions Techniques+Fonds Propres / Primes Nettes", "RATIOS COMBINES"],
    "Valeur": ["137.31%", "228.52%", "239.25%", "98.21%"]
}
df_ratios = pd.DataFrame(ratios)
st.dataframe(df_ratios)

# Données des ratios prudentiels
ratios_prudentiels = {
    "Ratio": [
        "Primes Nettes / Fonds Propres",
        "Provisions Techniques / Fonds Propres",
        "Provisions Techniques+Fonds Propres / Primes Nettes",
        "RATIOS COMBINES"
    ],
    "Valeur": [137.31, 228.52, 239.25, 98.21]
}

# Création du DataFrame
df_ratios = pd.DataFrame(ratios_prudentiels)

# Création du graphique
fig = px.bar(df_ratios, x='Ratio', y='Valeur', 
             title='Ratios Prudentiels',
             labels={'Valeur': 'Pourcentage (%)'},
             text='Valeur')

# Personnalisation du graphique
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

# Affichage du graphique dans Streamlit
st.plotly_chart(fig)

# Affichage du tableau des données
st.write("Tableau des Ratios Prudentiels")
st.dataframe(df_ratios)
