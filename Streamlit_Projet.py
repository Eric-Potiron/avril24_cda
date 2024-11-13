# -*- coding: utf-8 -*-

# cd Potiron\"DATA ANALYST"\STREAMLIT
# streamlit run Streamlit_Projet.py

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from pathlib import Path

# Définir le répertoire de base (répertoire du script courant)
BASE_DIR = Path(__file__).resolve().parent

# Charger les fichiers CSV
def load_csv(filename, sep=',', encoding='utf-8', header=0):
    return pd.read_csv(BASE_DIR / filename, sep=sep, encoding=encoding, header=header)

try:
    css_path = BASE_DIR / "styles.css"
    with css_path.open(encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Le fichier CSS 'styles.css' est introuvable.")
except Exception as e:
    st.error(f"Une erreur s'est produite lors du chargement du fichier CSS : {e}")

st.sidebar.title("Sommaire")

pages = ["Projet", "Jeux de données sources", "Pertinence des données", "Préparation des données",
         "Dataset final & DataVizualization", "Modélisation", "Prédictions", "Limites", "Conclusions"]

page = st.sidebar.radio("", pages)

# Ajouter des commentaires en dessous du sommaire
st.sidebar.write("---")  # Ligne de séparation facultative
st.sidebar.write("Cohorte avril 2024 / DA")
st.sidebar.write("Sujet : Températures Terrestres")
st.sidebar.write("Eric Potiron")

# PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if page == pages[0]: # PAGE 0 *** Projet ***
    st.write("### <u>Contexte</u>", unsafe_allow_html=True)

    # Texte justifié pour la page Contexte
    st.markdown(
        """
        <p class="justified-text">
        Il y a 20 000 ans, la Terre était en pleine période glaciaire, appelée le Dernier Maximum Glaciaire. Une grande partie de l'hémisphère nord était recouverte d'épaisses calottes glaciaires, notamment en Amérique du Nord, en Europe et en Asie. Les glaciers atteignaient parfois plusieurs kilomètres d'épaisseur, transformant les paysages et modifiant les écosystèmes. Le niveau des océans était plus bas d'environ 120 mètres, car une grande quantité d'eau était piégée sous forme de glace. En conséquence, des ponts terrestres reliaient des terres aujourd'hui séparées par des mers, facilitant la migration des espèces, y compris les humains. Les régions proches de l'équateur restaient relativement plus chaudes et humides, servant de refuges pour la biodiversité. La végétation dans les zones tempérées et froides était dominée par la toundra et la steppe, avec très peu de forêts…
        </p>
        <p class="justified-text">
        Un réchauffement climatique de +5°C nous sépare de cette période. Qui a duré 20.000 ans.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("### <u>Objectif</u>", unsafe_allow_html=True)

    # Texte justifié pour l'objectif du projet
    st.markdown(
        """
        <p class="justified-text">
        L'objectif de ce projet est de démontrer comment l'apprentissage des techniques de data analyse peut être appliqué pour étudier et comprendre les dynamiques du réchauffement climatique. En utilisant des jeux de données historiques sur les variations de température, les émissions de gaz à effet de serre, et des indicateurs économiques et démographiques, ce projet vise à :
            <br>
            <ul class="justified-text">
                <li>Explorer et visualiser les tendances mondiales liées au réchauffement climatique et à l'industrialisation.</li>
                <li>Évaluer les corrélations entre différents facteurs, tels que le PIB, la croissance démographique et les émissions de CO₂, pour identifier les relations clés influençant le climat.</li>
                <li>Utiliser des modèles de régression et d'autres techniques analytiques pour prédire les variations futures des températures mondiales en fonction des données actuelles et des scénarios projetés.</li>
                <li>Mettre en avant les limites et les défis de l'analyse de données environnementales, en insistant sur l'importance de la contextualisation des résultats et des modèles dans le cadre d'une compréhension plus large des phénomènes climatiques.</li>
            </ul>
            Ce projet cherche à illustrer l'application concrète des compétences en data analyse pour contribuer à une meilleure compréhension des problématiques environnementales, tout en servant de base de réflexion pour développer des stratégies d'adaptation face au changement climatique.
        </p>
        <p class="justified-text">
        Les 3 principaux gaz à effet de serre sont le dioxyde de carbone (CO²), le méthane (CH4) et le protoxyde d’azote (N²O). Le CO² est responsable de 65% de l’effet de serre anthropique, c’est-à-dire dû aux activités humaines.
        </p>
        <p class="justified-text">
        C’est sur ce dernier point que ce projet portera.
        </p>
        """,
        unsafe_allow_html=True
    )


# PAGE 1 *** PAGE 1  *** PAGE 1 *** PAGE 1  *** PAGE 1 *** PAGE 1  *** PAGE 1 *** PAGE 1  *** PAGE 1 *** PAGE 1  *** PAGE 1 *** PAGE 1  *** PAGE 1 *** PAGE 1  *** PAGE 1 *** PAGE 1  *** PAGE 1
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[1]: # PAGE 1 *** Jeux de données sources ***
    st.write("### Jeux de données sources")

    # Chemin relatif pour chaque fichier CSV
    csv_files = {
        "Zonal annual means.csv": BASE_DIR / 'Zonal annual means.csv',
        "Southern Hemisphere-mean monthly, seasonal, and annual means.csv": BASE_DIR / 'Southern Hemisphere-mean monthly, seasonal, and annual means.csv',
        "Northern Hemisphere-mean monthly, seasonal, and annual means.csv": BASE_DIR / 'Northern Hemisphere-mean monthly, seasonal, and annual means.csv',
        "Global-mean monthly, seasonal, and annual means.csv": BASE_DIR / 'Global-mean monthly, seasonal, and annual means.csv',
        "global-gdp-over-the-long-run.csv": BASE_DIR / 'global-gdp-over-the-long-run.csv',
        "owid-co2-data.csv": BASE_DIR / 'owid-co2-data.csv'
    }

    # Exemple d'affichage dynamique pour chaque fichier CSV
    for csv_name, csv_path in csv_files.items():
        if st.checkbox(f"### 📁 **{csv_name}**"):
            header = 0 if 'gdp' in csv_name or 'co2' in csv_name else 1
            df = load_csv(csv_path, header=header)

            rows, cols = df.shape
            num_duplicates = df.duplicated().sum()
            manquantes = df.isna().sum().to_frame().T
            info = pd.DataFrame({
                'Column': df.columns,
                'Non-Null Count': [df[col].notnull().sum() for col in df.columns],
                'Dtype': [df[col].dtype for col in df.columns]
            })
            info = info.T
            info.columns = info.iloc[0]
            info = info[1:]

            st.write(f"**Le dataframe contient** {rows} lignes et {cols} colonnes.")
            st.write(f"Le nombre de **doublons** est de : {num_duplicates}")
            st.write(f"**Valeurs manquantes :**")
            st.dataframe(manquantes)
            st.write(f"**Informations :**")
            st.dataframe(info)
            st.write(f"**En tête :**")  
            st.write(df.head())

            if 'Zonal' in csv_name:
                st.write("Source : NASA")
                st.write("Accès libre : [NASA Data](https://data.giss.nasa.gov/gistemp/)")
                st.markdown(
                    """
                    <p class="justified-text">
                    Le fichier contient des données annuelles moyennes de variations de température pour différentes régions du globe, de 1880 à une date récente. La NASA collecte ces données via divers moyens tels que des stations météorologiques, des bouées océaniques, et des satellites.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
            elif 'gdp' in csv_name:
                st.write("Source : OCDE")
                st.write("Accès libre : [Our World in Data](https://ourworldindata.org/)")
                st.markdown(
                    """
                    <p class="justified-text">
                    Ce fichier donne une vision de l’évolution du PIB mondial depuis l’an 1 jusqu’à 2022, ajusté en fonction de l'inflation.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )


# PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[2]: # PAGE 2 *** Pertinence des données ***
    st.write("## Pertinence des données")

    st.markdown(
        """
        <div class="justified-text">
            Cette section explore la pertinence des différentes données en mettant en lumière les variations de température, les évolutions du PIB, les émissions de CO2, et d'autres indicateurs environnementaux. Les graphiques ci-dessous illustrent ces relations et leur impact sur le climat global.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Disposition des boutons "Tout afficher" et "Tout masquer"
    col1, col2 = st.columns([1, 1])
    with col1:
        show_all = st.button("Tout afficher")
    with col2:
        hide_all = st.button("Tout masquer")

    # Gestion de l'affichage des sections
    if show_all:
        display_sections = True
    elif hide_all:
        display_sections = False
    else:
        display_sections = None

    # Chemins relatifs pour les fichiers CSV
    csv_paths = {
        "Zonal annual means": BASE_DIR / 'Zonal annual means.csv',
        "Global GDP": BASE_DIR / 'global-gdp-over-the-long-run.csv',
        "OWID CO2 data": BASE_DIR / 'owid-co2-data.csv'
    }

    # 1. Variations mondiales des températures globales par année
    if show_all or st.checkbox("Variations mondiales des températures globales par année"):
        df_temp = pd.read_csv(csv_paths["Zonal annual means"], header=0)
        plt.figure(figsize=(10, 5))
        plt.plot(df_temp['Year'], df_temp['Glob'], color='blue', marker='.', linestyle='-')
        plt.title('Variations mondiales des températures globales par année')
        plt.xlabel('Année')
        plt.ylabel('Température Globale (°C)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
            """
            <div class="justified-text">
                L'analyse de la courbe montre une augmentation graduelle des températures globales, soulignant le réchauffement climatique progressif depuis le début du 20ème siècle. Cette tendance est particulièrement marquée à partir des années 1960.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Variations des températures pour les hémisphères Nord et Sud
    if show_all or st.checkbox("Variations mondiales des températures pour les hémisphères Nord et Sud"):
        plt.figure(figsize=(10, 5))
        plt.plot(df_temp['Year'], df_temp['NHem'], color='green', marker='.', linestyle='-', label='Hémisphère Nord')
        plt.plot(df_temp['Year'], df_temp['SHem'], color='orange', marker='.', linestyle='-', label='Hémisphère Sud')
        plt.title('Variations mondiales des températures pour les hémisphères Nord et Sud')
        plt.xlabel('Année')
        plt.ylabel('Température (°C)')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
            """
            <div class="justified-text">
                La courbe indique des différences dans les tendances de réchauffement entre les deux hémisphères, l'hémisphère Nord affichant une hausse plus prononcée que le Sud, probablement en raison de la concentration des activités humaines et industrielles.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 3. Évolution du PIB mondial à partir de 1850
    if show_all or st.checkbox("Évolution du PIB mondial à partir de 1850"):
        df_gdp = pd.read_csv(csv_paths["Global GDP"], header=0)
        df_gdp_filtered = df_gdp[df_gdp['Year'] >= 1850]
        df_gdp_filtered['GDP'] = df_gdp_filtered['GDP'].astype(float) / 1e9  # Conversion en milliards

        plt.figure(figsize=(10, 5))
        plt.plot(df_gdp_filtered['Year'], df_gdp_filtered['GDP'], color='purple', marker='.', linestyle='-')
        plt.title('Évolution du PIB mondial à partir de 1850 (en milliards de dollars)')
        plt.xlabel('Année')
        plt.ylabel('PIB mondial (Md)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
            """
            <div class="justified-text">
                L'évolution du PIB mondial met en évidence une croissance continue, particulièrement après la Seconde Guerre mondiale, avec l'essor industriel et l'expansion économique globale.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Total des émissions de CO2 mondiales
    if show_all or st.checkbox("Total des émissions de CO² mondiales"):
        df_co2 = pd.read_csv(csv_paths["OWID CO2 data"], header=0)
        df_co2_world = df_co2[df_co2['country'] == 'World']

        plt.figure(figsize=(10, 5))
        plt.plot(df_co2_world['year'], df_co2_world['co2_including_luc'], color='red', marker='.', linestyle='-')
        plt.title("Émissions mondiales de CO² (y compris l'utilisation des terres)")
        plt.xlabel('Année')
        plt.ylabel('Émissions de CO2 (Gt)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
            """
            <div class="justified-text">
                Les émissions de CO2 ont augmenté de façon marquée depuis 1850, avec une accélération notable après 1950, coïncidant avec l'industrialisation et la croissance économique mondiale.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 5. Émissions mondiales de CO² par source (à partir de 1900)
    if show_all or st.checkbox("Émissions mondiales de CO² par source (à partir de 1900)"):
        columns_of_interest = ['year', 'cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 
                               'land_use_change_co2', 'oil_co2', 'other_industry_co2']
        rename_dict = {
            'year': 'Année',
            'cement_co2': 'CO2 Ciment',
            'coal_co2': 'CO2 Charbon',
            'flaring_co2': 'CO2 Torchage',
            'gas_co2': 'CO2 Gaz',
            'land_use_change_co2': 'CO2 Changement d\'utilisation des terres',
            'oil_co2': 'CO2 Pétrole',
            'other_industry_co2': 'CO2 Autres Industries'
        }
        df_co2_sources = df_co2_world[columns_of_interest].rename(columns=rename_dict)
        df_co2_sources = df_co2_sources[df_co2_sources['Année'] >= 1900]

        plt.figure(figsize=(12, 6))
        for column in list(rename_dict.values())[1:]:  # Exclure 'Année'
            plt.plot(df_co2_sources['Année'], df_co2_sources[column], marker='.', linestyle='-', label=column)

        plt.title('Émissions mondiales de CO² par source (à partir de 1900)')
        plt.xlabel('Année')
        plt.ylabel('Émissions de CO² (Gt)')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
            """
            <div class="justified-text">
                Cette analyse montre la contribution de différentes sources d'émissions de CO2 au fil du temps, avec une prédominance des émissions provenant du charbon, du pétrole et des changements d'utilisation des terres.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 6. Évolution de la population mondiale par année
    if show_all or st.checkbox("Évolution de la population mondiale par année"):
        df_co2_world['population'] = df_co2_world['population'] / 1e9  # Conversion en milliards

        plt.figure(figsize=(10, 5))
        plt.plot(df_co2_world['year'], df_co2_world['population'], color='green', marker='.', linestyle='-')
        plt.title('Évolution de la population mondiale par année')
        plt.xlabel('Année')
        plt.ylabel('Population mondiale (en milliards)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
            """
            <div class="justified-text">
                La courbe montre une augmentation rapide de la population mondiale depuis le début du 20ème siècle, particulièrement marquée après 1950, en raison des progrès en médecine et des politiques de santé publique.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 7. Corrélation entre la population et les émissions de CO2
    if show_all or st.checkbox("Corrélation entre population et émissions de CO²"):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_co2_world['year'], df_co2_world['population'], color='blue', label='Population mondiale (en milliards)', marker='.', linestyle='-')
        ax1.set_xlabel('Année')
        ax1.set_ylabel('Population mondiale (en milliards)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(df_co2_world['year'], df_co2_world['co2_including_luc'], color='red', label='Émissions de CO2 (Gt)', marker='.', linestyle='-')
        ax2.set_ylabel('Émissions de CO2 (Gt)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Corrélation entre la population mondiale et les émissions de CO²')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown(
            """
            <div class="justified-text">
                Cette visualisation met en lumière la corrélation entre la croissance de la population et l'augmentation des émissions de CO2. Cette relation est fortement influencée par l'industrialisation et l'urbanisation accrues.
            </div>
            """,
            unsafe_allow_html=True
        )


# PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[3]: # PAGE 3 *** Préparation des données ***
    st.write("## Préparation des données")

    st.markdown(
        """
        <div class="justified-text">
            Afin de préparer notre travail de modélisation, trois fichiers sont retenus :
            <br><br>
            <ol>
                <li>
                    Le fichier <span style="color: blue; font-style: italic; text-decoration: underline;">Zonal annual means.csv</span>, dont seules les colonnes Year, Glob, NHem, SHem sont conservées et renommées respectivement en Année, Var. Temp., Hém. Nord, et Hém. Sud.
                </li>
                <li>
                    Le fichier <span style="color: blue; font-style: italic; text-decoration: underline;">global-gdp-over-the-long-run.csv</span>, dont seules les colonnes Year et GDP sont conservées et renommées en Année et PIB (Md). La colonne PIB est arrondie à zéro chiffre après la virgule et exprimée en milliards (Md).
                </li>
                <li>
                    Le fichier <span style="color: blue; font-style: italic; text-decoration: underline;">owid-co2-data.csv</span>, scindé en deux fichiers :
                    <ul>
                        <li>Un fichier contenant les données de la zone mondiale pour les détails des émissions de CO2 par source.</li>
                        <li>Un fichier contenant les données de la zone mondiale pour les colonnes Population et Total CO2.</li>
                    </ul>
                </li>
            </ol>
            Les graphiques ci-dessous comparent les variations de température observées aux données projetées.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chargement des fichiers CSV
    csv_paths = {
        "owid_co2": BASE_DIR / 'owid-co2-data.csv',
        "zonal_annual_means": BASE_DIR / 'Zonal annual means.csv'
    }
    
    df_co2 = load_csv(csv_paths["owid_co2"], header=0)
    df_zonal = load_csv(csv_paths["zonal_annual_means"], header=0)

    # Filtrer pour ne garder que les données où 'country' est "World"
    df_co2 = df_co2[df_co2['country'] == 'World']

    # Renommer la colonne 'year' en 'Year' dans df_co2
    df_co2 = df_co2.rename(columns={'year': 'Year'})

    # Extraction des colonnes nécessaires
    df_co2 = df_co2[['Year', 'temperature_change_from_co2']]
    df_zonal = df_zonal[['Year', 'Glob']]

    # Filtrer les années de 1850 à 2023 dans chaque DataFrame
    df_co2 = df_co2[(df_co2['Year'] >= 1850) & (df_co2['Year'] <= 2023)]
    df_zonal = df_zonal[(df_zonal['Year'] >= 1850) & (df_zonal['Year'] <= 2023)]

    # Fusion des deux DataFrames sur la colonne 'Year'
    df_merged = pd.merge(df_co2, df_zonal, on='Year')

    # Ajout de la colonne pour la différence entre 'temperature_change_from_co2' et 'Glob'
    df_merged['difference'] = df_merged['temperature_change_from_co2'] - df_merged['Glob']

    # Tracé du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(df_merged['Year'], df_merged['temperature_change_from_co2'], label='owid-co2-data.csv')
    plt.plot(df_merged['Year'], df_merged['Glob'], label='Zonal annual means.csv')

    # Ajouter les labels et le titre
    plt.xlabel('Année')
    plt.ylabel('Variation de température')
    plt.title('Variation de température due au CO2 et autres facteurs (1900-2023)')
    plt.legend()

    # Afficher le graphique
    st.pyplot(plt)

   
# PAGE 4 *** PAGE 4  *** PAGE 4 *** PAGE 4  *** PAGE 4 *** PAGE 4  *** PAGE 4 *** PAGE 4  *** PAGE 4 *** PAGE 4  *** PAGE 4 *** PAGE 4  *** PAGE 4 *** PAGE 4  *** PAGE 4 *** PAGE 4  *** PAGE 4
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[4]: # PAGE 4 *** Dataset final & DataVizualization ***
    st.write("## Dataset final & DataVizualization")

    st.markdown(
        """
        <div class="justified-text">
            Après la fusion des différents jeux de données sur l'année et l'ajout des données pour l'année 2023, le dataset final se présente comme suit :
        </div>
        """,
        unsafe_allow_html=True
    )

    # Charger le fichier CSV du dataset final
    csv_path = BASE_DIR / 'final_df.csv'
    final_df = load_csv(csv_path, header=0)

    # Création des options de visualisation pour le dataset
    option = st.selectbox(
        "Choisissez l'option d'affichage des données :",
        ("Aucune sélection", "En-tête du data", "Fin du data", "Informations", "Valeurs manquantes", "Doublons")
    )

    # Affichage conditionnel selon l'option choisie
    if option == "Aucune sélection":
        st.write("Veuillez choisir une option pour afficher le dataset.")

    elif option == "En-tête du data":
        styled_df = final_df.head().style.format({
            final_df.columns[0]: "{:.0f}",
            final_df.columns[1]: "{:.2f}",
            final_df.columns[2]: "{:.2f}",
            final_df.columns[3]: "{:.2f}",
            final_df.columns[4]: "{:.0f}",
            final_df.columns[5]: "{:.0f}",
            final_df.columns[6]: "{:.0f}",
        })
        st.dataframe(styled_df)

    elif option == "Fin du data":
        styled_df = final_df.tail().style.format({
            final_df.columns[0]: "{:.0f}",
            final_df.columns[1]: "{:.2f}",
            final_df.columns[2]: "{:.2f}",
            final_df.columns[3]: "{:.2f}",
            final_df.columns[4]: "{:.0f}",
            final_df.columns[5]: "{:.0f}",
            final_df.columns[6]: "{:.0f}",
        })
        st.dataframe(styled_df)

    elif option == "Informations":
        st.write("### Informations sur le dataset")
        df_info = pd.DataFrame({
            "Type de données": final_df.dtypes,
            "Nombre de non nuls": final_df.count(),
            "Pourcentage de non nuls (%)": ((final_df.count() / len(final_df)) * 100).round(2),
            "Nombre de valeurs uniques": final_df.nunique()
        })
        st.table(df_info)
        rows, cols = final_df.shape
        st.write(f"Le dataset contient **{rows} lignes** et **{cols} colonnes**.")

    elif option == "Valeurs manquantes":
        st.write("### Valeurs manquantes")
        missing_values = final_df.isna().sum().to_frame().T
        st.dataframe(missing_values)

    elif option == "Doublons":
        num_duplicates = final_df.duplicated().sum()
        st.write(f"Le nombre de doublons est de : **{num_duplicates}**")

    # Ajout d'observations sur les données
    st.markdown(
        """
        <span style="color: blue; font-weight: bold; text-decoration: underline;">Observations</span>
        <div class="justified-text">
            Le graphique suivant montre l'absence de saisonnalité évidente et la corrélation apparente entre la population, le PIB, et les émissions de CO2, en particulier après 1980. 
            Depuis 1900, la population mondiale et les émissions de CO2 semblent suivre des trajectoires parallèles, tandis que le PIB montre une relation plus étroite avec les émissions à partir des années 1980.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Filtrer les données pour les années >= 1900
    filtered_final_df = final_df[final_df['Année'] >= 1900]

    # Création d'un graphique combiné pour les variables pertinentes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Tracé des courbes des différentes variables
    ax.plot(filtered_final_df['Année'], filtered_final_df['Var. Temp.'], label='Variation Température Globale', color='green', linestyle='-', marker='.')
    ax.plot(filtered_final_df['Année'], filtered_final_df['PIB (Md)'], label='PIB (Md)', color='blue', linestyle='-', marker='.')
    ax.plot(filtered_final_df['Année'], filtered_final_df['Population (m)'], label='Population (m)', color='purple', linestyle='--', marker='.')
    ax.plot(filtered_final_df['Année'], filtered_final_df['Total CO2 (mT)'], label='Total CO2 (mT)', color='red', linestyle='-.', marker='.')

    # Configurer les axes et le titre
    ax.set_title("Évolution des Données Globales depuis 1900")
    ax.set_xlabel("Année")
    ax.set_ylabel("Valeurs (log)")
    ax.set_yscale('log')  # Échelle logarithmique pour une meilleure visualisation
    ax.grid(True)
    ax.legend(loc='best')

    st.pyplot(plt)

    # Matrice de corrélation
    st.markdown(
        """
        <span style="color: blue; font-weight: bold; text-decoration: underline;">Matrice de corrélation</span>
        """,
        unsafe_allow_html=True
    )

    correlation_matrix = final_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de corrélation des variables')
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt)

    st.markdown(
        """
        <div class="justified-text">
            L'analyse de la matrice de corrélation révèle des liens significatifs entre certaines variables. Par exemple, une corrélation forte entre la population mondiale et les émissions de CO2, ainsi qu'entre le PIB et la température globale, soulignant l'impact économique et démographique sur l'environnement.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Affichage des métriques de performance du modèle linéaire
    st.markdown(
        """
        <span style="color: blue; font-weight: bold; text-decoration: underline;">Métriques de performance du modèle</span>
        """,
        unsafe_allow_html=True
    )

    # Création et évaluation du modèle
    X = final_df[['PIB (Md)', 'Population (m)', 'Total CO2 (mT)']]
    y = final_df['Var. Temp.']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calcul des métriques de performance
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Afficher les métriques de performance
    performance_metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'R² (Coefficient de détermination)': r2
    }
    performance_df = pd.DataFrame(performance_metrics.items(), columns=["Métrique", "Valeur"])
    performance_df['Valeur'] = performance_df['Valeur'].map("{:.3f}".format)
    st.table(performance_df)

    st.markdown(
        """
        <div class="justified-text">
            Ces métriques illustrent la capacité du modèle linéaire à prédire la variable de température à partir des variables explicatives choisies. Un coefficient R² élevé indique que le modèle explique une part importante de la variance observée dans les données.
        </div>
        """,
        unsafe_allow_html=True
    )


# PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[5]: # PAGE 5 *** Modélisation ***
    st.write("## Modélisations")

    # Charger le fichier CSV avec chemin relatif
    csv_path = BASE_DIR / 'final_df10.csv'
    data = load_csv(csv_path, header=0)

    # Description
    st.markdown(
        """
        <p class="justified-text">
        Sélectionnez les variables explicatives (y compris l'année) et l'année de départ pour entraîner un modèle de régression pour prédire 'Var. Temp.' en utilisant différents types de régression.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Liste des variables explicatives sans le PIB, la variable cible, et Hém. Nord/Sud
    variables_exclues = ['PIB (Md)', 'Var. Temp.', 'Hém. Nord', 'Hém. Sud']
    variables_disponibles = [col for col in data.columns if col not in variables_exclues]
    variables_choisies = st.multiselect("Choisissez les variables explicatives :", variables_disponibles)

    # Sélecteur d'année de départ
    annee_min, annee_max = data['Année'].min(), data['Année'].max()
    annee_depart = st.slider("Sélectionnez l'année de départ :", int(annee_min), int(annee_max), int(annee_min))

    # Menu de sélection du type de régression
    modele_selectionne = st.selectbox(
        "Sélectionnez le modèle de régression :", 
        ["Régression Linéaire", "Lasso", "Ridge", "Régression Polynomiale", "Forêt Aléatoire"]
    )

    # Filtrer les données en fonction de l'année de départ
    data_filtre = data[data['Année'] >= annee_depart]

    if variables_choisies:
        # Préparation des données
        X = data_filtre[variables_choisies]
        y = data_filtre['Var. Temp.']

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialisation du modèle
        if modele_selectionne == "Régression Linéaire":
            modele = LinearRegression()
        elif modele_selectionne == "Lasso":
            modele = Lasso(alpha=0.1)
        elif modele_selectionne == "Ridge":
            modele = Ridge(alpha=1.0)
        elif modele_selectionne == "Régression Polynomiale":
            modele = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
        elif modele_selectionne == "Forêt Aléatoire":
            modele = RandomForestRegressor(n_estimators=100, random_state=42)

        # Ajout des commentaires en fonction des choix de l'utilisateur
        if 'Population (m)' in variables_choisies:
            st.markdown(
                """
                <p class="justified-text">
                L'inclusion de la population comme variable explicative permet d'analyser son impact direct sur la variation de température, ce qui est pertinent compte tenu des études sur la corrélation entre la croissance démographique et les émissions de gaz à effet de serre.
                </p>
                """,
                unsafe_allow_html=True
            )

        if 'Total CO2 (mT)' in variables_choisies:
            st.markdown(
                """
                <p class="justified-text">
                En intégrant les émissions totales de CO2, l'analyse met l'accent sur la relation entre la hausse des émissions et les changements de température, essentielle pour comprendre l'impact des activités humaines sur le climat.
                </p>
                """,
                unsafe_allow_html=True
            )

        # Commentaires en fonction du modèle sélectionné
        st.markdown(f"<p><strong>Vous avez sélectionné le modèle {modele_selectionne} :</strong></p>", unsafe_allow_html=True)
        if modele_selectionne == "Régression Linéaire":
            st.markdown("<p class='justified-text'>La régression linéaire est appropriée pour modéliser les relations linéaires et permet une interprétation facile grâce à ses coefficients.</p>", unsafe_allow_html=True)
        elif modele_selectionne == "Lasso":
            st.markdown("<p class='justified-text'>Le modèle Lasso introduit une régularisation, ce qui est utile pour réduire les coefficients non significatifs et peut conduire à un modèle plus simple.</p>", unsafe_allow_html=True)
        elif modele_selectionne == "Ridge":
            st.markdown("<p class='justified-text'>Le modèle Ridge applique une régularisation pour limiter la complexité des coefficients et ainsi réduire le risque de surapprentissage.</p>", unsafe_allow_html=True)
        elif modele_selectionne == "Régression Polynomiale":
            st.markdown("<p class='justified-text'>La régression polynomiale permet de capturer des relations non linéaires plus complexes.</p>", unsafe_allow_html=True)
        elif modele_selectionne == "Forêt Aléatoire":
            st.markdown("<p class='justified-text'>Le modèle de Forêt Aléatoire est un puissant algorithme non linéaire qui combine plusieurs arbres de décision pour améliorer la précision.</p>", unsafe_allow_html=True)

        # Commentaires en fonction de l'année de départ
        if annee_depart < 1900:
            st.markdown("<p class='justified-text'>L'utilisation d'une année de départ antérieure à 1900 peut intégrer des tendances historiques à long terme.</p>", unsafe_allow_html=True)
        elif 1950 <= annee_depart < 2000:
            st.markdown("<p class='justified-text'>L'analyse à partir des années 1950 inclut l'ère moderne où l'industrialisation a entraîné des changements significatifs.</p>", unsafe_allow_html=True)
        elif annee_depart >= 2000:
            st.markdown("<p class='justified-text'>En commençant l'analyse à partir des années 2000, l'étude se concentre sur les changements climatiques récents.</p>", unsafe_allow_html=True)

        # Entraînement du modèle
        modele.fit(X_train, y_train)

        # Prédictions et évaluation du modèle
        y_pred = modele.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.markdown(f"<p><strong>Performance du modèle ({modele_selectionne}) :</strong></p>", unsafe_allow_html=True)
        st.write(f"Mean Squared Error (MSE) : {mse:.2f}")
        st.write(f"R² Score : {r2:.2f}")

        # Affichage des coefficients pour les modèles linéaires
        if modele_selectionne in ["Régression Linéaire", "Lasso", "Ridge"]:
            st.write("**Coefficients des variables explicatives :**")
            coeff_df = pd.DataFrame({'Variable': variables_choisies, 'Coefficient': modele.coef_})
            st.dataframe(coeff_df)

        # Affichage des prédictions par rapport aux années avec matplotlib
        X_test['Année'] = data_filtre.loc[X_test.index, 'Année']
        result_df = pd.DataFrame({
            'Année': X_test['Année'],
            'Valeurs Réelles': y_test,
            'Prédictions': y_pred
        }).sort_values(by='Année')

        # Création du graphique avec matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(result_df['Année'], result_df['Valeurs Réelles'], label='Valeurs Réelles', color='lightblue', linewidth=2)
        plt.plot(result_df['Année'], result_df['Prédictions'], label='Prédictions', color='blue', linewidth=2)
        plt.xlabel("Année")
        plt.ylabel("Température")
        plt.title(f"Comparaison des Prédictions et des Valeurs Réelles par Année ({modele_selectionne})")
        plt.legend()

        # Espacer les étiquettes de l'axe des années et afficher une année sur cinq, en commençant par l'année de départ
        annees_affichees = np.arange(annee_depart, annee_max + 1, 5)
        plt.xticks(annees_affichees, rotation=45)

        # Supprimer le séparateur des milliers
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

        st.pyplot(plt)
    else:
        st.markdown("<p class='justified-text'>Veuillez choisir des variables explicatives pour entraîner le modèle.</p>", unsafe_allow_html=True)


# PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6  *** PAGE 6 *** PAGE 6
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[6]: # PAGE 6 *** Prédictions ***
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.write("## Prédictions")

    # Charger le fichier CSV avec chemin relatif
    csv_path = BASE_DIR / 'final_df10.csv'
    data = load_csv(csv_path, header=0)

    # Description
    st.markdown("""
        <div class="justified-text">
        Sélectionnez les variables explicatives et l'année de départ pour entraîner un modèle de régression 
        et visualiser les prédictions de la variable 'Var. Temp.' sur le graphique, avec des projections jusqu'en 2100.
        </div>
    """, unsafe_allow_html=True)

    # Liste des variables explicatives sans le PIB ni la variable cible
    variables_disponibles = [col for col in data.columns if col not in ['PIB (Md)', 'Var. Temp.']]
    variables_choisies = st.multiselect("Choisissez les variables explicatives :", variables_disponibles, format_func=lambda x: x)

    # Sélecteur d'année de départ
    annee_min, annee_max = data['Année'].min(), data['Année'].max()
    annee_depart = st.slider("Sélectionnez l'année de départ :", int(annee_min), int(annee_max), int(annee_min))

    # Menu de sélection du type de régression
    modele_selectionne = st.selectbox("Sélectionnez le modèle de régression :", 
                                      ["Régression Linéaire", "Lasso", "Ridge", "Régression Polynomiale", "Forêt Aléatoire"])

    # Filtrer les données en fonction de l'année de départ
    data_filtre = data[data['Année'] >= annee_depart]

    # Vérifier si des variables explicatives ont été choisies
    if variables_choisies:
        # Préparation des données
        X = data_filtre[variables_choisies]
        y = data_filtre['Var. Temp.']

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sélection du modèle de régression
        if modele_selectionne == "Régression Linéaire":
            modele = LinearRegression()
        elif modele_selectionne == "Lasso":
            modele = Lasso(alpha=0.1)
        elif modele_selectionne == "Ridge":
            modele = Ridge(alpha=1.0)
        elif modele_selectionne == "Régression Polynomiale":
            modele = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
        elif modele_selectionne == "Forêt Aléatoire":
            modele = RandomForestRegressor(n_estimators=100, random_state=42)

        # Ajout des commentaires en fonction des choix de l'utilisateur
        if 'Population (m)' in variables_choisies:
            st.markdown("""
            <div class="justified-text">
            L'inclusion de la population comme variable explicative permet d'analyser son impact direct sur la variation de température, ce qui est pertinent compte tenu des études sur la corrélation entre la croissance démographique et les émissions de gaz à effet de serre.
            </div>
            """, unsafe_allow_html=True)

        if 'Total CO2 (mT)' in variables_choisies:
            st.markdown("""
            <div class="justified-text">
            En intégrant les émissions totales de CO2, l'analyse met l'accent sur la relation entre la hausse des émissions et les changements de température, essentielle pour comprendre l'impact des activités humaines sur le climat.
            </div>
            """, unsafe_allow_html=True)

        if 'Année' in variables_choisies:
            st.markdown("""
            <div class="justified-text">
            L'utilisation de l'année comme variable explicative permet de capturer des tendances temporelles, ce qui peut être important pour modéliser des évolutions chronologiques.
            </div>
            """, unsafe_allow_html=True)

        # Commentaires en fonction du modèle sélectionné
        st.write(f"**Vous avez sélectionné le modèle {modele_selectionne} :**")
        if modele_selectionne == "Régression Linéaire":
            st.markdown("""
            <div class="justified-text">
            La régression linéaire est appropriée pour modéliser les relations linéaires et permet une interprétation facile grâce à ses coefficients.
            </div>
            """, unsafe_allow_html=True)
        elif modele_selectionne == "Lasso":
            st.markdown("""
            <div class="justified-text">
            Le modèle Lasso introduit une régularisation, ce qui est utile pour réduire les coefficients non significatifs et peut conduire à un modèle plus simple.
            </div>
            """, unsafe_allow_html=True)
        elif modele_selectionne == "Ridge":
            st.markdown("""
            <div class="justified-text">
            Le modèle Ridge applique une régularisation pour limiter la complexité des coefficients et ainsi réduire le risque de surapprentissage.
            </div>
            """, unsafe_allow_html=True)
        elif modele_selectionne == "Régression Polynomiale":
            st.markdown("""
            <div class="justified-text">
            La régression polynomiale permet de capturer des relations non linéaires plus complexes.
            </div>
            """, unsafe_allow_html=True)
        elif modele_selectionne == "Forêt Aléatoire":
            st.markdown("""
            <div class="justified-text">
            Le modèle de Forêt Aléatoire est un puissant algorithme non linéaire qui combine plusieurs arbres de décision pour améliorer la précision.
            </div>
            """, unsafe_allow_html=True)

        # Commentaires en fonction de l'année de départ
        if annee_depart < 1900:
            st.markdown("""
            <div class="justified-text">
            L'utilisation d'une année de départ antérieure à 1900 peut intégrer des tendances historiques à long terme.
            </div>
            """, unsafe_allow_html=True)
        elif 1950 <= annee_depart < 2000:
            st.markdown("""
            <div class="justified-text">
            L'analyse à partir des années 1950 inclut l'ère moderne où l'industrialisation a entraîné des changements significatifs.
            </div>
            """, unsafe_allow_html=True)
        elif annee_depart >= 2000:
            st.markdown("""
            <div class="justified-text">
            En commençant l'analyse à partir des années 2000, l'étude se concentre sur les changements climatiques récents.
            </div>
            """, unsafe_allow_html=True)

        # Entraînement du modèle
        modele.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred_test = modele.predict(X_test)

        # Prédictions sur l'ensemble complet pour visualisation
        y_pred_all = modele.predict(X)

        # Générer des années futures jusqu'en 2100
        annees_futures = pd.DataFrame({'Année': np.arange(annee_max + 1, 2101)})

        # Simuler des valeurs futures pour les variables explicatives
        for var in variables_choisies:
            if var != 'Année':
                coef_tendance = (data_filtre[var].iloc[-1] - data_filtre[var].iloc[0]) / (data_filtre['Année'].iloc[-1] - data_filtre['Année'].iloc[0])
                annees_futures[var] = data_filtre[var].iloc[-1] + coef_tendance * (annees_futures['Année'] - data_filtre['Année'].iloc[-1])

        # Prédictions sur les années futures
        y_pred_futures = modele.predict(annees_futures[variables_choisies])
        annees_futures['Prédictions'] = y_pred_futures

        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)

        st.write(f"**Performance du modèle ({modele_selectionne}) :**")
        st.write(f"Mean Squared Error (MSE) : {mse:.2f}")
        st.write(f"R² Score : {r2:.2f}")

        # Affichage des coefficients pour les modèles linéaires
        if modele_selectionne in ["Régression Linéaire", "Lasso", "Ridge"]:
            st.write("**Coefficients des variables explicatives :**")
            coeff_df = pd.DataFrame({'Variable': variables_choisies, 'Coefficient':            modele.coef_})
            st.dataframe(coeff_df)

        # Sauvegarder les variables explicatives choisies
        with open(BASE_DIR / 'variables_explicatives.pkl', 'wb') as file:
            pickle.dump(variables_choisies, file)

        # Sauvegarder le modèle entraîné
        with open(BASE_DIR / 'modele_entraine.pkl', 'wb') as file:
            pickle.dump(modele, file)

        # Affichage des prédictions par rapport aux années avec matplotlib
        data_filtre['Prédictions'] = y_pred_all
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        plt.figure(figsize=(14, 7))
        plt.plot(data_filtre['Année'], data_filtre['Var. Temp.'], label='Valeurs Réelles', color='lightblue', linewidth=2)
        plt.plot(data_filtre['Année'], data_filtre['Prédictions'], label='Prédictions', color='blue', linewidth=2)
        plt.plot(annees_futures['Année'], annees_futures['Prédictions'], label='Prédictions Futures', color='red', linewidth=2, linestyle='--')
        plt.xlabel("Année")
        plt.ylabel("Température")
        plt.title(f"Comparaison des Prédictions et des Valeurs Réelles par Année ({modele_selectionne})")
        plt.legend()

        # Espacer les étiquettes de l'axe des années et afficher une année sur cinq, en commençant par l'année de départ
        annees_affichees = np.arange(annee_depart, 2101, 5)
        plt.xticks(annees_affichees, rotation=45)
        st.pyplot(plt)
        st.markdown('</div>', unsafe_allow_html=True)

        # Courbes d'évolution de la population et du CO2
        if 'Population (m)' in variables_choisies:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            plt.figure(figsize=(14, 5))
            plt.plot(data_filtre['Année'], data_filtre['Population (m)'], label='Population (m) Réelle', color='green', linewidth=2)
            plt.plot(annees_futures['Année'], annees_futures['Population (m)'], label='Population (m) Projetée', color='darkgreen', linewidth=2, linestyle='--')
            plt.xlabel("Année")
            plt.ylabel("Population (m)")
            plt.title("Évolution de la Population de l'Année Sélectionnée à 2100")
            plt.legend()
            st.pyplot(plt)
            st.markdown('</div>', unsafe_allow_html=True)

        if 'Total CO2 (mT)' in variables_choisies:
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            plt.figure(figsize=(14, 5))
            plt.plot(data_filtre['Année'], data_filtre['Total CO2 (mT)'], label='Total CO2 (mT) Réel', color='purple', linewidth=2)
            plt.plot(annees_futures['Année'], annees_futures['Total CO2 (mT)'], label='Total CO2 (mT) Projeté', color='darkviolet', linewidth=2, linestyle='--')
            plt.xlabel("Année")
            plt.ylabel("Total CO2 (mT)")
            plt.title("Évolution du CO2 de l'Année Sélectionnée à 2100")
            plt.legend()
            st.pyplot(plt)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("Veuillez choisir des variables explicatives pour entraîner le modèle.")
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[7]: # PAGE 7 *** Limites ***
    st.write("## Limites")

    st.markdown(
        """
        <div class="justified-text">
            Il existe plusieurs limites à ces prédictions, et pour n'en citer que trois :
            <br><br>
            1- Selon <a href="https://population.un.org/wpp/Download/Standard/MostUsed/" style="color: blue;">l'ONU</a>, les projections de population mondiale ne peuvent excéder 10 milliards 291 millions d'individus, et le pic sera atteint en 2084.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Charger les données ONU depuis le fichier CSV
    onu_population = load_csv('onu_population.csv', sep=';', encoding='ISO-8859-1')

    # Calcul du total de la population par année
    total_population_by_year = onu_population.groupby('Année')['Population'].sum()

    # Création du graphique pour le total par année avec annotations
    plt.figure(figsize=(6, 3))
    plt.plot(total_population_by_year.index, total_population_by_year, marker=' ', linewidth=1, color='green')

    # Personnalisation du graphique
    plt.title('Évolution de la Population mondiale de 2024 à 2100', fontsize=12)
    plt.ylabel('Total Population (millions)', fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    # Annotation des valeurs tous les 20 ans et la dernière année
    for year in list(range(2044, 2101, 20)) + [2100]:
        population = total_population_by_year.get(year, 0)
        plt.text(year, population - 250, f'{population:,.0f}', fontsize=10, ha='center', va='bottom', color='blue')

    plt.tight_layout()
    st.pyplot(plt)

    st.markdown(
        """
        <div class="justified-text">
            <br>
            2- Selon <a href="https://fr.statista.com/statistiques/559789/reserves-mondiales-de-petrole-brut-1990/#:~:text=En%202023%2C%20les%20r%C3%A9serves%20mondiales,de%201.569%20milliards%20de%20barils." style="color: blue;">Statista</a>, les réserves mondiales de pétrole sont estimées à 1570 milliards de barils. Avec une consommation actuelle de 100 millions de barils par jour, il reste environ 40 ans de stocks, à consommation constante. Cependant, en pratique, les producteurs n'exporteront plus bien avant l'épuisement des réserves.
            <br><br>
            Selon <a href="https://www.planete-energies.com/fr/media/chiffres/consommation-mondiale-gaz-naturel." style="color: blue;">Planète énergie (Total Énergies)</a>, les réserves mondiales de gaz s'élèvent à 188 000 milliards de m³. Avec une consommation actuelle de 3940 milliards de m³ par an, il reste environ 47 ans de stocks, à consommation constante. Il est à noter que la Russie et l'Iran possèdent à elles deux 37% des réserves mondiales.
            <br><br>
            Toujours selon <a href="https://www.planete-energies.com/fr/media/chiffres/reserves-mondiales-charbon" style="color: blue;">Planète énergie (Total Énergies)</a>, les réserves mondiales de charbon couvrent 200 ans de consommation actuelle. Les États-Unis, la Russie, l'Australie, la Chine et l'Inde détiennent 76% des réserves mondiales.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="justified-text">
            <br>
            3- Le site <a href="https://www.cea.fr/presse/Pages/actualites-communiques/environnement/bilan-mondial-methane-record-emission.aspx#:~:text=%E2%80%8BLe%20bilan%20mondial%20de,des%20%C3%A9missions%20mondiales%20de%20m%C3%A9thane." style="color: blue;">CEA</a> précise que les activités humaines ont émis un record de 400 millions de tonnes métriques de méthane en 2020. Or, le méthane (CH₄) a un potentiel de réchauffement global (PRG) 85 fois supérieur à celui du CO₂ sur 20 ans, qui diminue à 30 fois au bout de 100 ans selon la formule PRG(t) = 84.e^(-0.05776t), où t désigne le temps.
            <br><br>
            Pour information, ce méthane provient de l'agriculture, de l'élevage, de la gestion des déchets, de l'industrie énergétique (fuites lors d'extractions de pétrole et de charbon), et de la combustion de la biomasse.
            <br><br>
            La courbe ci-dessous donne un aperçu des stocks de méthane en équivalent CO₂ dans l'atmosphère en fonction du calcul de son PRG, pour une émission constante annuelle de 400 millions de tonnes métriques. On observe qu'au bout de 15 ans, 350 gigatonnes d'équivalent CO₂ ont été ajoutées à l'atmosphère.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Calcul des émissions de méthane en équivalent CO2 sur 75 ans
    PRG_initial = 84
    lambda_decay = 0.05776
    emissions_per_year = 0.400  # émissions en milliards de tonnes de méthane par an
    years = np.arange(0, 76)  # simulation sur 75 ans

    PRG_values = PRG_initial * np.exp(-lambda_decay * years)
    annual_emissions_CO2eq = emissions_per_year * PRG_values  # émissions annuelles en équivalent CO2
    cumulative_emissions_CO2eq = np.cumsum(annual_emissions_CO2eq)  # cumul des émissions

    # Création du DataFrame pour visualisation
    df_75_years = pd.DataFrame({
        'Année': years,
        'PRG annuel équivalent CO2 (millions de tonnes)': annual_emissions_CO2eq,
        'Cumul CO2 équivalent (millions de tonnes)': cumulative_emissions_CO2eq
    })

    # Création de l'histogramme et de la courbe cumulée
    plt.figure(figsize=(14, 7))
    plt.bar(df_75_years['Année'], df_75_years['Cumul CO2 équivalent (millions de tonnes)'], color='lightblue', label='Cumul CO2 équivalent')
    plt.plot(df_75_years['Année'], df_75_years['Cumul CO2 équivalent (millions de tonnes)'], color='blue', marker='o', label='Courbe Cumul CO2 équivalent')

    plt.title('Histogramme et courbe du cumul CO2 équivalent sur 75 ans', fontsize=12)
    plt.xlabel('Année')
    plt.ylabel('Cumul CO2 équivalent (Gigatonnes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    st.pyplot(plt)


# PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
elif page == pages[8]: # PAGE 8 *** Conclusions ***
    st.write("## Conclusions")

    st.markdown(
        """
        <div class="justified-text">
            Cette analyse a permis de mettre en pratique des techniques avancées de data analyse appliquées au domaine complexe et essentiel qu'est le réchauffement climatique. En explorant des jeux de données variés, allant des températures globales aux émissions de gaz à effet de serre et à la croissance économique, nous avons pu comprendre l'interdépendance entre ces facteurs. 
            <br><br>
            L'apprentissage de la data analyse a été au cœur de la démarche, intégrant des étapes de préparation de données, de visualisation et de modélisation pour extraire des informations significatives. 
            <br><br>
            Les compétences acquises en manipulation de données, en création de visualisations pertinentes et en application de modèles de régression ont permis de révéler des corrélations importantes, telles que le lien entre le PIB mondial, la croissance de la population, et l'augmentation des températures. L'utilisation de métriques de performance et de l'évaluation des modèles a renforcé la capacité à valider des hypothèses sur la base de données réelles.
            <br><br>
            Cette application pratique a également mis en évidence les limites des prédictions, rappelant que l'analyse des données historiques, bien qu'informative, doit être complétée par une compréhension approfondie des facteurs extérieurs et des incertitudes futures. La modélisation nous a montré que si la data analyse est puissante pour établir des relations et prévoir des tendances, elle nécessite une interprétation prudente dans le contexte de la complexité environnementale.
            <br><br>
            L'étude des variations de température et de leurs causes a souligné l'importance d'une approche multidimensionnelle, mêlant des données économiques, démographiques et climatiques. Elle a démontré comment l'apprentissage de la data analyse peut être un outil indispensable pour sensibiliser, prévoir, et guider les politiques en matière de lutte contre le changement climatique.
            <br><br>
            En conclusion, cette expérience souligne la pertinence de l'intégration des techniques de data analyse dans la recherche de solutions aux défis environnementaux actuels et futurs.
        </div>
        """,
        unsafe_allow_html=True
    )


# PAGE 9  ***  PAGE 9  *** PAGE 9 *** PAGE 9 *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9 *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9  *** PAGE 9 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------