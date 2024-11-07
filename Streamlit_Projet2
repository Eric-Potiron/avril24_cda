# -*- coding: utf-8 -*-
# Streamlit Application Code
# cd Potiron\"DATA ANALYST"\STREAMLIT
# streamlit run Streamlit_Projet.py

import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import warnings

# Pour éviter les messages warning
warnings.filterwarnings('ignore')

# Mise en cache des chargements de données pour accélérer les rechargements
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Fonctions de visualisation
def plot_time_series(df, x, y, title, xlabel, ylabel, color):
    plt.figure(figsize=(8, 4))
    plt.plot(df[x], df[y], marker='.', linestyle='-', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    st.pyplot(plt)

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matrice de corrélation')
    st.pyplot(plt)

def forecast_temperature(df, n_years=10):
    model = ARIMA(df['Var. Temp.'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_years)
    plt.figure(figsize=(8, 4))
    plt.plot(df['Année'], df['Var. Temp.'], label="Température actuelle")
    plt.plot(range(len(df), len(df) + n_years), forecast, label="Prévision future", linestyle='--')
    plt.title('Prévision de la température pour les prochaines années')
    plt.xlabel('Année')
    plt.ylabel('Température (°C)')
    plt.legend()
    st.pyplot(plt)

def plot_missing_data(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Visualisation des valeurs manquantes")
    st.pyplot(plt)

# Fonction pour afficher les informations d'un DataFrame
def display_data_info(df, name, source_url, description):
    rows, cols = df.shape
    num_duplicates = df.duplicated().sum()
    manquantes = df.isna().sum().to_frame().T
    info_df = pd.DataFrame({
        'Column': df.columns, 
        'Non-Null Count': [df[col].notnull().sum() for col in df.columns], 
        'Dtype': [df[col].dtype for col in df.columns]
    }).T
    info_df.columns = info_df.iloc[0]
    info_df = info_df[1:]
    
    st.write(f"**Source** : {source_url}")
    st.markdown(description, unsafe_allow_html=True)
    st.write(f"**Le dataframe** {name} **contient** {rows} lignes et {cols} colonnes.")
    st.write(f"Nombre de **doublons** : {num_duplicates}")
    st.write(f"**Valeurs manquantes** :")
    st.dataframe(manquantes)
    st.write(f"**Informations** :")
    st.dataframe(info_df)
    st.write(f"**En-tête :**")
    st.dataframe(df.head())

# Préparation du dataset final
@st.cache_data
def prepare_final_dataset():
    # Chargement et transformation des différents fichiers
    df1 = load_data('C:/Potiron/DATA ANALYST/STREAMLIT/Zonal annual means.csv')
    df_pib = load_data('C:/Potiron/DATA ANALYST/STREAMLIT/global-gdp-over-the-long-run.csv')
    df4 = load_data('C:/Potiron/DATA ANALYST/STREAMLIT/owid-co2-data.csv')
    # Transformation et fusion des fichiers
    df_final = pd.merge(df1, df_pib, left_on='Year', right_on='Year', how='outer')
    df_final = pd.merge(df_final, df4, left_on='Year', right_on='year', how='outer')
    df_final.rename(columns={'Year': 'Année', 'Glob': 'Var. Temp.', 'NHem': 'Hém. Nord', 'SHem': 'Hém. Sud'}, inplace=True)
    return df_final

# Initialisation de la barre latérale
st.sidebar.title("Sommaire")
pages = {
    "Projet": "page_contexte",
    "Jeux de données sources": "page_data_sources",
    "Pertinence des données": "page_pertinence_donnees",
    "Préparation des données": "page_preparation_donnees",
    "Dataset final & Data Visualization": "page_dataset_final",
    "Modélisation": "page_modelisation",
    "Machine Learning": "page_machine_learning",
    "Conclusions": "page_conclusions",
    "Limites": "page_limites"
}
page_name = st.sidebar.radio("Navigation", list(pages.keys()))

# Affichage de la page sélectionnée
if page_name == "Projet":
    st.write("### Contexte")
    st.markdown(
        """
        <p class="justified-text">
        Il y a 20 000 ans, la Terre était en pleine période glaciaire. 
        Le contexte du réchauffement climatique actuel est à rapprocher de cette évolution.
        </p>
        """,
        unsafe_allow_html=True
    )

elif page_name == "Jeux de données sources":
    st.write("### Jeux de données sources")
    df1 = load_data('C:/Potiron/DATA ANALYST/STREAMLIT/Zonal annual means.csv')
    display_data_info(df1, "Zonal annual means", "https://data.giss.nasa.gov/gistemp/", "<p>Données sur les températures moyennes...</p>")
    df_pib = load_data('C:/Potiron/DATA ANALYST/STREAMLIT/global-gdp-over-the-long-run.csv')
    display_data_info(df_pib, "Global GDP", "https://ourworldindata.org/", "<p>Données sur le PIB mondial...</p>")
    df4 = load_data('C:/Potiron/DATA ANALYST/STREAMLIT/owid-co2-data.csv')
    display_data_info(df4, "OWID CO2 Data", "https://github.com/owid/co2-data", "<p>Données sur les émissions de CO2...</p>")

elif page_name == "Pertinence des données":
    st.write("### Pertinence des données")
    df1 = load_data('C:/Potiron/DATA ANALYST/STREAMLIT/Zonal annual means.csv')
    plot_time_series(df1, 'Year', 'Glob', 'Variations mondiales des températures globales', 'Année', 'Température Globale (°C)', 'blue')
    plot_missing_data(df1)  # Visualisation des valeurs manquantes

elif page_name == "Préparation des données":
    st.write("### Préparation des données")
    st.markdown(
        """
        Préparation des fichiers pour l'analyse finale.
        """,
        unsafe_allow_html=True
    )

elif page_name == "Dataset final & Data Visualization":
    st.write("### Dataset final & Data Visualization")
    final_df = prepare_final_dataset()
    st.write(final_df.head())
    st.write("#### Analyse de corrélation")
    plot_correlation_matrix(final_df)
    st.write("#### Prévision des températures")
    forecast_temperature(final_df)

elif page_name == "Modélisation":
    st.write("## Modélisation")
    st.markdown("Section dédiée à la modélisation.")

elif page_name == "Machine Learning":
    st.write("## Machine Learning")
    st.markdown("Section dédiée à l'implémentation de modèles Machine Learning.")

elif page_name == "Conclusions":
    st.write("## Conclusions")
    st.markdown("Conclusion sur les observations et analyses.")

elif page_name == "Limites":
    st.write("## Limites")
    st.markdown("Limites de l'analyse et des données disponibles.")
