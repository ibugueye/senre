import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Fonction pour récupérer les données de l'API Flask
def get_data(endpoint):
    try:
        response = requests.get(f"http://localhost:5001/api/{endpoint}")
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from API: {e}")
        return None

# Configuration de la page Streamlit
st.set_page_config(page_title="SEN-RE Dashboard", layout="wide")
st.title("Tableau de Bord - SenRé")

# Chiffre d'affaires
ca_data = get_data('chiffre_affaires')
if ca_data:
    ca_df = pd.DataFrame(ca_data)
    
    # Afficher le DataFrame
    st.subheader("Chiffre d'Affaires par Zone")
    st.dataframe(ca_df)
    
    # Créer et afficher le graphique
    fig_ca = px.bar(ca_df, x='Zones', y='2018', title="Répartition Géographique du Chiffre d'Affaires (2018)")
    st.plotly_chart(fig_ca)
else:
    st.warning("Impossible de charger les données du chiffre d'affaires")

# Répartition géographique
geo_data = get_data('repartition_geographique')
geo_df = pd.DataFrame(geo_data)
fig_geo = px.pie(geo_df, values='2018', names='Zones', title="Répartition Géographique du Chiffre d'Affaires (2018)")
st.plotly_chart(fig_geo)

# Structure du portefeuille
portfolio_data = get_data('structure_portefeuille')
if portfolio_data:
    portfolio_df = pd.DataFrame(portfolio_data)
    fig_portfolio = px.pie(portfolio_df, values='2018', names='Branches', title="Structure du Portefeuille (2018)")
    st.plotly_chart(fig_portfolio)
else:
    st.warning("Unable to load portfolio data")
# Affichage des ratios clés
st.subheader("Ratios Clés (2018)")
col1, col2, col3 = st.columns(3)
col1.metric("ROE", "12.71%")
col2.metric("Ratio Combiné Net", "98.21%")
col3.metric("Taux de Frais Généraux", "7.66%")

