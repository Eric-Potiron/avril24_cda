# -*- coding: utf-8 -*-

import io
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from pathlib import Path

# Définir le répertoire de base (répertoire du script courant)
BASE_DIR = Path(__file__).resolve().parent

# Charger les fichiers CSV
def load_csv(filename, sep=',', encoding='utf-8', header=0):
    return pd.read_csv(BASE_DIR / filename, sep=sep, encoding=encoding, header=header)

st.sidebar.title("Sommaire")

pages = ["Projet", "Jeux de données sources", "Pertinence des données", "Préparation des données",
         "Dataset final & DataVizualization", "Modélisation", "Prédictions", "Limites", "Conclusions"]


page = st.sidebar.radio("", pages)

# Ajouter des commentaires en dessous du sommaire
st.sidebar.write("---")  # Ligne de séparation facultative
st.sidebar.write("Cohorte avril 2024 / DA")
st.sidebar.write("Sujet : Températures Terrestres")
st.sidebar.write("Eric Potiron")

# Ajout du CSS pour la justification
st.markdown(
    """
    <style>
    .justified-text {
        text-align: justify;
        line-height: 1.5; /* Ajustez la valeur pour l'espacement souhaité entre les lignes */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0 *** PAGE 0  *** PAGE 0
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if page == pages[0]:
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
        L'objectif de ce projet est de démontrer comment l'apprentissage des techniques de data analyse peut être appliqué pour étudier et comprendre les dynamiques du réchauffement climatique. En utilisant des jeux de données historiques sur les variations de température, les émissions de gaz à effet de serre, et des indicateurs économiques et démographiques, ce projet vise à :<br><br>
            <span style="margin-left:20px; display: inline-block;">
            Explorer et visualiser les tendances mondiales liées au réchauffement climatique et à l'industrialisation.<br><br>
            Évaluer les corrélations entre différents facteurs, tels que le PIB, la croissance démographique et les émissions de CO₂, pour identifier les relations clés influençant le climat.<br><br>
            Utiliser des modèles de régression et d'autres techniques analytiques pour prédire les variations futures des températures mondiales en fonction des données actuelles et des scénarios projetés.<br><br>
            Mettre en avant les limites et les défis de l'analyse de données environnementales, en insistant sur l'importance de la contextualisation des résultats et des modèles dans le cadre d'une compréhension plus large des phénomènes climatiques.<br><br>
            Ce projet cherche à illustrer l'application concrète des compétences en data analyse pour contribuer à une meilleure compréhension des problématiques environnementales, tout en servant de base de réflexion pour développer des stratégies d'adaptation face au changement climatique.<br><br>
            </span>
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

elif page == pages[1]:
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
            # Définir le header en fonction du fichier (ici, certains fichiers ont un header différent)
            header = 0 if 'gdp' in csv_name or 'co2' in csv_name else 1
            df = load_csv(csv_path, header=header)

            # Calcul des informations
            rows, cols = df.shape  # Lignes et colonnes
            num_duplicates = df.duplicated().sum()  # Doublons
            manquantes = df.isna().sum().to_frame().T  # Valeurs manquantes
            info = pd.DataFrame({
                'Column': df.columns,
                'Non-Null Count': [df[col].notnull().sum() for col in df.columns],
                'Dtype': [df[col].dtype for col in df.columns]
            })
            info = info.T  # Transpose le DataFrame pour un affichage horizontal
            info.columns = info.iloc[0]  # Définit les noms de colonnes
            info = info[1:]  # Supprime la première ligne redondante

            # Affichage des données dans Streamlit
            st.write(f"**Le dataframe contient** {rows} lignes et {cols} colonnes.")
            st.write(f"Le nombre de **doublons** est de : {num_duplicates}")
            st.write(f"**Valeurs manquantes :**")
            st.dataframe(manquantes)
            st.write(f"**Informations :**")
            st.dataframe(info)
            st.write(f"**En tête :**")  
            st.write(df.head())

            # Ajouter des descriptions spécifiques pour certains fichiers
            if 'Zonal' in csv_name:
                st.write("Source : NASA")
                st.write("Accès libre : (https://data.giss.nasa.gov/gistemp/)")
                st.markdown(
                    """
                    <p class="justified-text">
                    Le fichier contient des données annuelles moyennes de variations de température pour différentes régions du globe, de 1880 à une date récente.
                    <br><br>
                    La NASA se source via différents moyens :
                    <br><strong><u>Stations météorologiques de surface</strong></u>, <strong><u>Bouées océaniques</strong></u>,
                    <strong><u>Navires et plates-formes pétrolières</strong></u>, et <strong><u>Satellites</strong></u>.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
            elif 'Southern Hemisphere' in csv_name:
                st.write("Source : NASA")
                st.write("Accès libre : (https://data.giss.nasa.gov/gistemp/)")
                st.markdown(
                    """
                    <p class="justified-text">
                    Le fichier contient des données de variations de température moyennes pour l'hémisphère sud, avec des valeurs mensuelles, saisonnières et annuelles.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
            elif 'Northern Hemisphere' in csv_name:
                st.write("Source : NASA")
                st.write("Accès libre : (https://data.giss.nasa.gov/gistemp/)")
                st.markdown(
                    """
                    <p class="justified-text">
                    Le fichier contient des données de variations de température moyennes pour l'hémisphère nord, avec des valeurs mensuelles, saisonnières et annuelles.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
            elif 'gdp' in csv_name:
                st.write("Source : OCDE")
                st.write("Accès libre : (https://ourworldindata.org/)")
                st.markdown(
                    """
                    <p class="justified-text">
                    Ce fichier donne une vision de l’évolution du PIB mondial depuis l’an 1 jusqu’à 2022. Ces estimations historiques du PIB sont ajustées en fonction de l'inflation.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
            elif 'co2' in csv_name:
                st.write("Source : OWID")
                st.write("Accès libre : (https://github.com/owid/co2-data)")
                st.markdown(
                    """
                    <p class="justified-text">
                    Le fichier contient des données sur les émissions de CO₂ et d'autres gaz à effet de serre par pays, couvrant plusieurs indicateurs environnementaux et économiques.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )


# PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2 *** PAGE 2  *** PAGE 2
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

elif page == pages[2]:

    st.write("## Pertinence des données")

    # Disposition des boutons "Tout afficher" et "Tout masquer" côte à côte
    col1, col2 = st.columns([1, 1])
    with col1:
        show_all = st.button("Tout afficher")
    with col2:
        hide_all = st.button("Tout masquer")

    # Gestion de l'affichage global des sections
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

    if show_all or st.checkbox("Variations mondiales des températures globales par année"):
        df1 = pd.read_csv(csv_paths["Zonal annual means"], header=0)
        plt.figure(figsize=(8, 4))
        plt.plot(df1['Year'], df1['Glob'], marker='.', linestyle='-', color='blue')
        plt.title('Variations mondiales des températures globales par année')
        plt.xlabel('Année')
        plt.ylabel('Température Globale (°C)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
        """
        <p class="justified-text">
        La courbe montre une tendance globale des températures au fil des années. Il est possible de distinguer une augmentation progressive des températures globales sur la période couverte par les données. Cette tendance indique un réchauffement climatique évident, avec des fluctuations annuelles, mais une augmentation générale sur le long terme. Cela suggère un effet des activités humaines et des changements environnementaux sur les températures moyennes globales, notamment depuis le début de l'ère industrielle.
        <br>
        On peut d’ores et déjà constater une augmentation significative à partir de 1960.
        </p>
        """, 
        unsafe_allow_html=True)

    if show_all or st.checkbox("Variations mondiales des températures pour les hémisphères Nord et Sud"):
        df1 = pd.read_csv(csv_paths["Zonal annual means"], header=0)
        plt.figure(figsize=(8, 4))
        plt.plot(df1['Year'], df1['NHem'], marker='.', linestyle='-', color='green', label='Hémisphère Nord')
        plt.plot(df1['Year'], df1['SHem'], marker='.', linestyle='-', color='orange', label='Hémisphère Sud')
        plt.title('Variations mondiales des températures pour les hémisphères Nord et Sud')
        plt.xlabel('Année')
        plt.ylabel('Température (°C)')
        plt.legend(loc='best')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
        """
        <p class="justified-text">
        La courbe montre que les températures des deux hémisphères ont une tendance similaire avec une augmentation progressive sur la période étudiée. Cependant, il y a quelques différences notables : L'hémisphère nord montre une tendance légèrement plus marquée à la hausse par rapport à l'hémisphère sud, ce qui pourrait être lié à une concentration plus importante d'activités humaines et industrielles dans cette région.
        <br>
        Ces résultats renforcent l'hypothèse d'un réchauffement climatique global, bien que les impacts spécifiques puissent varier légèrement entre les hémisphères.
        </p>
        """, 
        unsafe_allow_html=True)

    if show_all or st.checkbox("Évolution du PIB mondial à partir de 1850"):
        df_pib = pd.read_csv(csv_paths["Global GDP"], header=0)
        df_pib_1850 = df_pib[df_pib['Year'] >= 1850]
        df_pib_1850['GDP'] = df_pib_1850['GDP'].astype('float64') / 1e9
        plt.figure(figsize=(8, 4))
        plt.plot(df_pib_1850['Year'], df_pib_1850['GDP'], marker='.', linestyle='-', color='purple')
        plt.title('Évolution du PIB mondial à partir de 1850 en milliards de dollars')
        plt.xlabel('Année')
        plt.ylabel('PIB mondial (en milliards de dollars)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
        """
        <p class="justified-text">
        La tendance générale est donc celle d'une augmentation continue du PIB mondial, traduisant une expansion économique globale, malgré certaines périodes d'instabilité. Cela reflète une augmentation significative de la production, de la technologie et des capacités économiques à l’échelle mondiale.
        </p>
        """, 
        unsafe_allow_html=True)

    if show_all or st.checkbox("Total des émissions de CO² mondiales"):
        df4 = pd.read_csv(csv_paths["OWID CO2 data"], header=0)
        df4_world = df4[df4['country'] == 'World']
        plt.figure(figsize=(8, 4))
        plt.plot(df4_world['year'], df4_world['co2_including_luc'], marker='.', linestyle='-', color='brown')
        plt.title("Émissions mondiales de CO² (y compris l'utilisation des terres)")
        plt.xlabel('Année')
        plt.ylabel('Émissions de CO2 (Gt)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
        """
        <p class="justified-text">
        La courbe montre une augmentation significative des émissions mondiales de CO₂ depuis 1850. Voici quelques observations clés :
        <br>
        Tendance générale : Les émissions de CO₂ ont suivi une tendance à la hausse continue, en particulier après la Seconde Guerre mondiale, marquant l'ère de la forte industrialisation et de l'augmentation de la consommation énergétique mondiale.
        <br>
        Périodes de hausse rapide : Certaines périodes montrent des hausses plus brusques, liées à l'expansion industrielle, au développement économique rapide des pays, et à l'utilisation accumulée des combustibles fossiles.
        <br>
        Changements d'utilisation des terres : L'inclusion des changements d'utilisation des terres (comme la déforestation) dans ces données montre l'impact significatif des pratiques agricoles et forestières sur les émissions totales.
        <br>
        Ces résultats mettent en évidence la contribution humaine croissante aux émissions de CO₂ et soulignent l'importance de stratégies d'atténuation pour gérer et réduire ces émissions à l'échelle mondiale.
        </p>
        """, 
        unsafe_allow_html=True)

    if show_all or st.checkbox("Émissions mondiales de CO² par source (à partir de 1900)"):
        df4_world = df4[df4['country'] == 'World']
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
        df4_world = df4_world[columns_of_interest].rename(columns=rename_dict)
        df4_detail_1900 = df4_world[df4_world['Année'] >= 1900]
        renamed_columns = list(rename_dict.values())[1:]  # Exclure 'Année'
        plt.figure(figsize=(8, 4))
        for column in renamed_columns:
            plt.plot(df4_detail_1900['Année'], df4_detail_1900[column], marker='.', linestyle='-', label=column)
        plt.title('Émissions mondiales de CO² par source (à partir de 1900)')
        plt.xlabel('Année')
        plt.ylabel('Émissions de CO² (Gt)')
        plt.legend(loc='best')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
        """
        <p class="justified-text">
        Cette analyse souligne les changements majeurs dans les sources d'émission au 20ème siècle, avec un passage progressif du charbon vers le pétrole et le gaz naturel, tout en mettant en lumière la diversification des sources industrielles de CO₂. La prise en compte de ces divers facteurs est essentielle pour une stratégie de réduction des émissions efficace.
        </p>
        """, 
        unsafe_allow_html=True)

    if show_all or st.checkbox("Évolution de la population mondiale par année"):
        df4_pop = df4[df4['country'] == 'World']
        df4_pop['population'] = df4_pop['population'] / 1e9
        plt.figure(figsize=(8, 4))
        plt.plot(df4_pop['year'], df4_pop['population'], marker='.', linestyle='-', color='blue')
        plt.title('Évolution de la population mondiale par année')
        plt.xlabel('Année')
        plt.ylabel('Population mondiale (en milliards)')
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
        """
        <p class="justified-text">
        Le graphique montre une augmentation rapide de la population mondiale à partir du 20ème siècle :
        </p>
        <div style="margin-left: 30px;">
            <p class="justified-text">Stabilité relative avant le 19ème siècle : La population mondiale était relativement stable jusqu'à la fin du 19ème siècle, avec une légère croissance régulière.</p>
            <p class="justified-text">Croissance rapide au 20ème siècle : Une augmentation significative se produit au cours du 20ème siècle, avec des taux de croissance plus élevés après la Seconde Guerre mondiale. Cette période marque l'amélioration des soins de santé, une diminution de la mortalité infantile et une augmentation de la longévité.</p>
            <p class="justified-text">Tendance vers la surpopulation : La courbe indique une tendance vers une population mondiale qui dépasse les 8 milliards au 21ème siècle, mettant en évidence les défis liés à la gestion des ressources, du climat et des infrastructures pour soutenir la croissance.</p>
        </div>
        <p class="justified-text">
        Cette tendance à la croissance rapide montre l’impact humain croissant sur la planète et souligne l’importance des stratégies de développement durable pour répondre aux défis démographiques.
        </p>
        """, 
        unsafe_allow_html=True)

    if show_all or st.checkbox("Corrélation entre population et émission de CO²"):
        world_data2 = df4[df4['country'] == 'World']
        world_data2['population'] = world_data2['population'] / 1e9
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(world_data2['year'], world_data2['population'], marker='.', linestyle='-', color='blue', label='Population')
        ax1.set_xlabel('Année')
        ax1.set_ylabel('Population mondiale', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2 = ax1.twinx()
        ax2.plot(world_data2['year'], world_data2['co2_including_luc'], marker='.', linestyle='-', color='red', label='CO² (y compris LUC)')
        ax2.set_ylabel('Émissions de CO² (Gt)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.title('Évolution de la population mondiale & des émissions de CO² (y compris LUC)')
        ax1.grid(True)
        fig.tight_layout()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
        plt.grid(True)
        st.pyplot(plt)
        st.markdown(
        """
        <p class="justified-text">
        La courbe montre une augmentation parallèle de la population mondiale et des émissions de CO₂. Cela indique que la croissance démographique et l'augmentation des émissions de CO₂ sont intimement liées.
        <br>
        L'industrialisation et l'urbanisation, surtout à partir du 20ème siècle, semblent avoir contribué à la fois à l'augmentation rapide de la population et à la croissance des émissions.
        <br>
        Après 1950, il y a une accélération marquée de la croissance de la population, accompagnée d'une augmentation rapide des émissions de CO₂. Cela correspond à la période de forte industrialisation mondiale et d'expansion des énergies fossiles.
        <br>
        Cette analyse met en évidence la relation étroite entre la croissance démographique et les émissions de CO₂. Les défis environnementaux liés aux émissions de gaz à effet de serre ne peuvent être séparés des dynamiques démographiques.
        </p>
        """, 
        unsafe_allow_html=True)


# PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3 *** PAGE 3  *** PAGE 3
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

elif page == pages[3]:

    st.write("## Préparation des données")

    st.markdown(
        """
        <div style="text-align: justify;">
            Afin de préparer notre travail de modélisation, 3 fichiers sont retenus.
            <br><br>
            1- Le fichier <span style="color: blue; font-style: italic; text-decoration: underline;">Zonal annual means.csv</span> et dont seules les colonnes Year, Glob, NHem, SHem sont conservées et renommées respectivement Année, Var. Temp., Hém. Nord et Hém. Sud.
            <br><br>
            2- Le fichier <span style="color: blue; font-style: italic; text-decoration: underline;">global-gdp-over-the-long-run.csv</span> et dont seules les colonnes Year et GDP sont conservées et renommées respectivement Année et PIB (Md). A cela s'ajoute :
                <br>
                <span style="margin-left:20px; display: inline-block;">
                    ✔ Division des valeurs de la colonne PIB (Md) par 10^9 pour les exprimer en milliard (Md).
                    <br>
                    ✔ Arrondi de la colonne PIB à 0 chiffres après la virgule.
                    <br>
                    ✔ Ajout des années 1820 à 1823.
                    <br>
                    ✔ Interpolation linéaire pour les valeurs manquantes de PIB.
                </span>
            <br><br>
            3- Le fichier <span style="color: blue; font-style: italic; text-decoration: underline;">owid-co2-data.csv</span> est scindé en 2 fichiers : 1 contenant les données de la zone monde pour les détails des émissions de CO² par source et 1 contenant les données de la zone monde pour les colonnes Population et Total CO².
                <span style="margin-left:20px; display: inline-block;">
                    <br>
                    📁 Zone monde pour les détails des émissions de CO² par source
                    <br>
                    <span style="margin-left:20px; display: inline-block;">
                        ✔ On ne conserve que les colonnes year, population, cement_co2, coal_co2, flaring_co2, gas_co2, land_use_change_co2, oil_co2, other_industry_co2, co2_including_luc renommées respectivements Année, Population (m), Ciment, Charbon, Torchage, Gaz, Usage des sols, Pétrole, Industrie et Total CO² (GT).
                        <br>
                        ✔ On ne garde que les années > 1880.
                        <br>
                        ✔ Division des valeurs de Population par 10^6 pour les exprimer en millions.
                    </span>
                    <br>
                    <br>
                    📁 Zone monde pour les émissions de CO²
                    <br>
                    <span style="margin-left:20px; display: inline-block;">
                        ✔ On ne conserve que les colonnes year, population et co2_including_luc renommées respectivements Année, Population (m) et Total CO² (GT).
                        <br>
                        ✔ On ne garde que les années > 1880.
                        <br>
                        ✔ Division des valeurs de Population par 10^6 pour les exprimer en millions.
                    </span>
                    <br><br>
                </span>
                    4- Choix retenu concernant les Variations de températures et hypothèse de travail.
                    <br><br>
                    Comme l'indique le graphique ci-dessous, les données de température du fichier Zonal annual means.csv ressemble plus à des données constatées plutôt qu'à des données issues d'une régression (owid-co2-data.csv). C'est pour cette raison que ces données serviront de base de travail.
                    <br><br>
                    Compte tenu de l'absence de données concernant les émissions de méthane (CH4) et de protoxyde d’azote (N²O) avant 1990, l'hypothèse de travail ne retiendra que le CO².<br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chargement des fichiers CSV avec chemins relatifs
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

    # Suppression des doublons éventuels de l'année 'Year'
    df_merged = df_merged.drop_duplicates(subset='Year')

    # Ajout de la colonne pour la différence entre 'temperature_change_from_co2' et 'Glob'
    df_merged['difference'] = df_merged['temperature_change_from_co2'] - df_merged['Glob']

    # Filtrer pour ne garder que les années à partir de 1900
    df_merged = df_merged[df_merged['Year'] >= 1900]

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

elif page == pages[4]:

    # Lire le fichier CSV avec chemin relatif
    csv_path = BASE_DIR / 'final_df.csv'

    st.write("## Dataset final & DataVizualization")
    st.markdown(
        """
        Après fusion sur l'année des 3 fichiers cités précédemment et ajout des données pour l'année 2023, le fichier de travail se présente comme suit :
        """,
        unsafe_allow_html=True)

    # Crée des boutons radio pour sélectionner l'option d'affichage
    option = st.selectbox(
        "",
        ("Aucune sélection", "En-tête du data", "Fin du data", "Informations", "Valeurs manquantes", "Doublons")
    )

    # Charger le DataFrame pour chaque option sélectionnée
    final_df = load_csv(csv_path, header=0)

    if option == "Aucune sélection":
        st.write(f"")

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
        df_info = pd.DataFrame({
            "Type de données": final_df.dtypes,
            "Nombre de non nuls": final_df.count(),
            "Pourcentage de non nuls (%)": ((final_df.count() / len(final_df)) * 100).round(2),  # Pourcentage arrondi à 2 chiffres
            "Nombre de valeurs uniques": final_df.nunique()
        })
        st.table(df_info)
        rows, cols = final_df.shape
        st.write(f"Le dataframe contient {rows} lignes et {cols} colonnes.")

    elif option == "Valeurs manquantes":
        manquantes = final_df.isna().sum().to_frame().T
        st.dataframe(manquantes)

    elif option == "Doublons":
        num_duplicates = final_df.duplicated().sum()
        st.write(f"Le nombre de doublons est de : {num_duplicates}")

    st.markdown(
        """
        <span style="color: blue; font-weight: bold; text-decoration: underline;">Observations</span><br>
            <p class="justified-text">
            Le graphique ci-après indique une absence de saisonnalité, et à partir des années 80 une corrélation multiple qui se dessine. Ainsi depuis 1900, la population et les émissions de CO² sont étroitement corrélées, et il semble que le PIB et les émissions de CO² amorcent une corrélation à compter des années 1980.
            </p>
        
        Les variables étant indiquées au fil du temps, c’est donc une série temporelle.
        """,
        unsafe_allow_html=True)

    # Filtrage des données pour ne prendre en compte que celles à partir de l'année 1900
    filtered_final_df = final_df[final_df['Année'] >= 1900]

    # Création d'un graphique unique pour toutes les données demandées
    fig, ax = plt.subplots(figsize=(8, 5))

    # Tracé des différentes courbes
    ax.plot(filtered_final_df['Année'], filtered_final_df['Var. Temp.'], marker='.', linestyle='-', label='Variation Température Globale', color='green')
    ax.plot(filtered_final_df['Année'], filtered_final_df['PIB (Md)'], marker='.', linestyle='-', label='PIB (Md)', color='blue')
    ax.plot(filtered_final_df['Année'], filtered_final_df['Population (m)'], marker='.', linestyle='--', label='Population (m)', color='purple')
    ax.plot(filtered_final_df['Année'], filtered_final_df['Total CO2 (mT)'], marker='.', linestyle='-.', label='Total CO2 (mT)', color='red')

    # Configuration de l'affichage du graphique
    ax.set_title('Évolution des Données Globales (depuis 1900)')
    ax.set_xlabel('Année')
    ax.set_ylabel('Valeurs')
    ax.grid(True)
    ax.legend(loc='center right')
    ax.set_yscale('log')
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown(
        """
        <span style="color: blue; font-weight: bold; text-decoration: underline;">Matrice de corrélation</span><br>
        """,
        unsafe_allow_html=True
    )

    correlation_matrix = final_df.corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de corrélation des variables (Fichier Modifié)')
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt)

    st.markdown(
        """
        L’analyse de la matrice de corrélation montre que :
            <br><br>
            <span style="margin-left:20px; display: inline-block;">
                <p class="justified-text">
                <span style="font-weight: bold; text-decoration: underline;">Var. Temp. (Variable cible)</span> :
                Corrélation forte avec le PIB (0.94) : Cela montre une relation positive importante, indiquant que les augmentations du PIB sont fortement associées aux variations de la température. Corrélation forte avec la Population (0.94) : La croissance de la population mondiale est étroitement liée aux variations de température, ce qui peut être lié à des impacts environnementaux plus larges.
                <br><br>
                <span style="font-weight: bold; text-decoration: underline;">CO² total</span> : Corrélation très forte avec la Population (0.997) : Cela indique que l’augmentation de la population est un facteur majeur des émissions de CO². Corrélation très forte avec le PIB (0,96) : L’augmentation économique s’accompagne souvent d’une augmentation des émissions de CO², probablement due à une intensification de l’industrialisation.
                <br><br>
                <span style="font-weight: bold; text-decoration: underline;">Année</span> : Corrélation élevée avec le Var. Temp. (0.87), Population (0.94), et Total CO² (0.95) : Cela indique que sur la période étudiée, la population, les émissions de CO², et la température ont toutes montré une augmentation continue au fil du temps.
                </p>
            </span>
        <br><br>
        <span style="color: blue; font-weight: bold; text-decoration: underline;">Métriques</span><br>
        """,
        unsafe_allow_html=True
    )

    # Exécution du modèle
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

    # Résumé des métriques de performance
    performance_metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'R² (Coefficient de détermination)': r2
    }

    performance_df = pd.DataFrame(performance_metrics.items(), columns=["Métrique", "Valeur"])
    performance_df['Valeur'] = performance_df['Valeur'].map("{:.3f}".format)

    # Afficher avec Streamlit
    st.table(performance_df)

    st.markdown(
        """
        Les métriques de performance du modèle de régression linéaire sont :
            <br><br>
            <span style="margin-left:20px; display: inline-block;">
                <p class="justified-text">
                <span style="font-weight: bold; text-decoration: underline;">MAE</span> : Indique une erreur moyenne absolue entre les prédictions et les valeurs réelles.
                <br><br>
                <span style="font-weight: bold; text-decoration: underline;">MSE</span> : Représente la moyenne des erreurs quadratiques. Plus cette valeur est proche de zéro, mieux le modèle est ajusté.
                <br><br>
                <span style="font-weight: bold; text-decoration: underline;">RMSE</span> : Donne une idée de la taille des erreurs en utilisant la même unité que la variable cible (degrés Celsius).
                <br><br>
                <span style="font-weight: bold; text-decoration: underline;">R²</span> : Indique que le modèle explique une part significative de la variance des données.
                </p>
            </span>
        """,
        unsafe_allow_html=True)


# PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5 *** PAGE 5  *** PAGE 5
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Votre code Streamlit
elif page == pages[5]:

    st.write("## Modélisations")

    # Charger le fichier CSV avec chemin relatif
    csv_path = BASE_DIR / 'final_df10.csv'
    data = load_csv(csv_path, header=0)

    # Description
    st.write("""
        Sélectionnez les variables explicatives (y compris l'année) et l'année de départ 
        pour entraîner un modèle de régression pour prédire 'Var. Temp.' en utilisant différents types de régression.
    """)

    # Liste des variables explicatives sans le PIB et la variable cible
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
            # Pipeline pour la régression polynomiale
            modele = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
        elif modele_selectionne == "Forêt Aléatoire":
            modele = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Entraînement du modèle
        modele.fit(X_train, y_train)
        
        # Prédictions et évaluation du modèle
        y_pred = modele.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"**Performance du modèle ({modele_selectionne}) :**")
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
        st.write("Veuillez choisir des variables explicatives pour entraîner le modèle.")


# PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6 *** PAGE 6  *** PAGE 6  *** PAGE 6 *** PAGE 6
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

elif page == pages[6]:

    st.write("## Prédictions")

    # Charger le fichier CSV avec chemin relatif
    csv_path = BASE_DIR / 'final_df10.csv'
    data = load_csv(csv_path, header=0)

    # Description
    st.write("""
        Sélectionnez les variables explicatives et l'année de départ pour entraîner un modèle de régression 
        et visualiser les prédictions de la variable 'Var. Temp.' sur le graphique, avec des projections jusqu'en 2100.
    """)

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
            # Pipeline pour la régression polynomiale
            modele = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
        elif modele_selectionne == "Forêt Aléatoire":
            modele = RandomForestRegressor(n_estimators=100, random_state=42)
        
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
            coeff_df = pd.DataFrame({'Variable': variables_choisies, 'Coefficient': modele.coef_})
            st.dataframe(coeff_df)

        # Sauvegarder les variables explicatives choisies
        with open(BASE_DIR / 'variables_explicatives.pkl', 'wb') as file:
            pickle.dump(variables_choisies, file)

        # Sauvegarder le modèle entraîné
        with open(BASE_DIR / 'modele_entraine.pkl', 'wb') as file:
            pickle.dump(modele, file)
        
        # Affichage des prédictions par rapport aux années avec matplotlib
        data_filtre['Prédictions'] = y_pred_all
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

        # Supprimer le séparateur des milliers
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

        st.pyplot(plt)
        
        # Courbes d'évolution de la population et du CO2
        if 'Population (m)' in variables_choisies:
            plt.figure(figsize=(14, 5))
            plt.plot(data_filtre['Année'], data_filtre['Population (m)'], label='Population (m) Réelle', color='green', linewidth=2)
            plt.plot(annees_futures['Année'], annees_futures['Population (m)'], label='Population (m) Projetée', color='darkgreen', linewidth=2, linestyle='--')
            plt.xlabel("Année")
            plt.ylabel("Population (m)")
            plt.title("Évolution de la Population de l'Année Sélectionnée à 2100")
            plt.legend()
            st.pyplot(plt)

        if 'Total CO2 (mT)' in variables_choisies:
            plt.figure(figsize=(14, 5))
            plt.plot(data_filtre['Année'], data_filtre['Total CO2 (mT)'], label='Total CO2 (mT) Réel', color='purple', linewidth=2)
            plt.plot(annees_futures['Année'], annees_futures['Total CO2 (mT)'], label='Total CO2 (mT) Projeté', color='darkviolet', linewidth=2, linestyle='--')
            plt.xlabel("Année")
            plt.ylabel("Total CO2 (mT)")
            plt.title("Évolution du CO2 de l'Année Sélectionnée à 2100")
            plt.legend()
            st.pyplot(plt)
    else:
        st.write("Veuillez choisir des variables explicatives pour entraîner le modèle.")


# PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7 *** PAGE 7  *** PAGE 7
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

elif page == pages[7]:

    st.write("## Limites")

    st.markdown(
        """
        <div style="text-align: justify;">
            Il existe plusieurs limites à ces prédictions, et pour n'en citer que 3 :
            <br><br>
            1- Selon <a href="https://population.un.org/wpp/Download/Standard/MostUsed/" style="color: blue;">l'ONU</a> les projections de population mondiale ne peuvent excéder 10 milliards 291 millions d'individus, et le pic sera atteint en 2084.
            <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Charger les données ONU depuis le fichier CSV
    onu_population = load_csv('onu_population.csv', sep=';', encoding='ISO-8859-1')

    # Calcul du total de la population par année
    total_population_by_year = onu_population.groupby('Année')['Population'].sum()

    # Création du graphique pour le total par année avec annotations à côté, sans la première année
    plt.figure(figsize=(6, 3))
    plt.plot(total_population_by_year.index, total_population_by_year, marker=' ', linewidth=1, color='green')

    # Personnalisation du graphique
    plt.title('Évolution de la Population mondiale de 2024 à 2100', fontsize=12)
    plt.ylabel('Total Population (millions)', fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    # Annotation du nombre tous les 20 ans, y compris la dernière année, sans la première valeur
    for year in list(range(2044, 2101, 20)) + [2100]:  # Commence à 2044 pour exclure 2024
        population = total_population_by_year.get(year, 0)
        plt.text(year, population - 250, f'{population:,.0f}', fontsize=10, ha='center', va='bottom', color='blue')

    # Ajustement du layout
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown(
        """
        <div style="text-align: justify;">
            <br><br>
            2- Selon <a href="https://fr.statista.com/statistiques/559789/reserves-mondiales-de-petrole-brut-1990/#:~:text=En%202023%2C%20les%20r%C3%A9serves%20mondiales,de%201.569%20milliards%20de%20barils." style="color: blue;">statista</a> les réserves mondiales de pétrole sont 1570 milliards de barils.
            Avec une consommation actuelle de 100 millions de barils / jour, il reste environ 40 ans de stocks, à consommation fixe. En théorie, car les producteurs n'exporteront plus bien avant.<br>
            Selon <a href="https://www.planete-energies.com/fr/media/chiffres/consommation-mondiale-gaz-naturel." style="color: blue;">Planète énergie (Total énergie)</a> les réserves mondiales de Gaz sont 188.000 milliards de m3.
            Avec une consommation actuelle de 3940 milliards de m3 / an, il reste environ 47 ans de stocks, à consommation fixe. En théorie, car les producteurs n'exporteront plus bien avant.
            En en précisant que la Russie et l'Iran possèdent à elles deux 37% des réserves mondiales.<br>
            Toujours selon <a href="https://www.planete-energies.com/fr/media/chiffres/reserves-mondiales-charbon" style="color: blue;">Planète énergie (Total énergie)</a> les réserves mondiales de charbon couvrent 200 ans de consommation actuelle. Étant précisé que ces 5 pays : Etats-Unis, Russie, Australie, Chine et Inde possèdent 76% des réserves mondiales.
            <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align: justify;">
            <br><br>
            3- Le site <a href="https://www.cea.fr/presse/Pages/actualites-communiques/environnement/bilan-mondial-methane-record-emission.aspx#:~:text=%E2%80%8BLe%20bilan%20mondial%20de,des%20%C3%A9missions%20mondiales%20de%20m%C3%A9thane." style="color: blue;">statista</a> précise que les activités humaines ont émis un record de 400 millions de tonnes métriques de méthane en 2020.
            Or le méthane (CH4) à un PRG (potentiel de réchauffement global) 85 fois supérieur au CO² à 20 ans, qui diminue à 30 fois au bout de 100 ans selon la formule PRG(t) = 84.e^(-0.05776t), t désigne le temps.
            Pour information, ce méthane provient de l'agriculture, l'élevage, la gestion des déchets, l'industrie énergétique (fuites lors d'extractions de pétrole et charbon) et la combustion de la biomasse.
            <br><br>
            La courbe ci-dessous donne un aperçu du stocks de méthane en équivalent CO² dans l'atmosphère en fonction du calcul de son PRG, pour une constante annuelle de 400 millions de tonnes métriques.
            On observe qu'au bout de 15 ans, ont été ajoutées 350 gigatonnes d'équivalent CO² dans l'atmosphère.
            <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Définition des paramètres
    PRG_initial = 84
    lambda_decay = 0.05776
    emissions_per_year = 0.400  # émissions en millions de tonnes de méthane par an
    years = np.arange(0, 76)  # simulation sur 75 ans

    # Calcul des PRG pour chaque année
    PRG_values = PRG_initial * np.exp(-lambda_decay * years)
    annual_emissions_CO2eq = emissions_per_year * PRG_values  # calcul des émissions annuelles en CO2 équivalent
    cumulative_emissions_CO2eq = np.cumsum(annual_emissions_CO2eq)  # calcul cumulatif

    # Création du DataFrame
    df_75_years = pd.DataFrame({
        'Année': years,
        'Émissions CH4 (millions de tonnes)': emissions_per_year,
        'PRG annuel équivalent CO2 (millions de tonnes)': annual_emissions_CO2eq,
        'Cumul CO2 équivalent (millions de tonnes)': cumulative_emissions_CO2eq
    })

    # Création de l'histogramme et de la courbe de la colonne cumulée
    plt.figure(figsize=(14, 7))
    plt.bar(df_75_years['Année'], df_75_years['Cumul CO2 équivalent (millions de tonnes)'], color='lightblue', label='Cumul CO2 équivalent')
    plt.plot(df_75_years['Année'], df_75_years['Cumul CO2 équivalent (millions de tonnes)'], color='blue', marker='o', label='Courbe Cumul CO2 équivalent')

    plt.title('Histogramme et courbe du cumul CO² équivalent sur 75 ans', fontsize=12)
    plt.xlabel('Année')
    plt.ylabel('Cumul CO2 équivalent (Gigatonnes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    st.pyplot(plt)

    # Affichage du début du DataFrame
    # st.write("Aperçu du DataFrame :")
    # st.dataframe(df_75_years.head())


# PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8 *** PAGE 8  *** PAGE 8
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

elif page == pages[8]:

    st.write("## Conclusions")

    st.markdown(
        """
        <div style="text-align: justify;">
            <br><br>
            Cette analyse a permis de mettre en pratique des techniques avancées de data analyse appliquées au domaine complexe et essentiel qu'est le réchauffement climatique. En explorant des jeux de données variés, allant des températures globales aux émissions de gaz à effet de serre et à la croissance économique, nous avons pu comprendre l'interdépendance entre ces facteurs. L'apprentissage de la data analyse a été au cœur de la démarche, intégrant des étapes de préparation de données, de visualisation et de modélisation pour extraire des informations significatives.<br><br>
            Les compétences acquises en manipulation de données, en création de visualisations pertinentes et en application de modèles de régression ont permis de révéler des corrélations importantes, telles que le lien entre le PIB mondial, la croissance de la population, et l'augmentation des températures. L'utilisation de métriques de performance et de l'évaluation des modèles a renforcé la capacité à valider des hypothèses sur la base de données réelles.<br><br>
            Cette application pratique a également mis en évidence les limites des prédictions, rappelant que l'analyse des données historiques, bien qu'informative, doit être complétée par une compréhension approfondie des facteurs extérieurs et des incertitudes futures. La modélisation nous a montré que si la data analyse est puissante pour établir des relations et prévoir des tendances, elle nécessite une interprétation prudente dans le contexte de la complexité environnementale.<br><br>
            L'étude des variations de température et de leurs causes a souligné l'importance d'une approche multidimensionnelle, mêlant des données économiques, démographiques et climatiques. Elle a démontré comment l'apprentissage de la data analyse peut être un outil indispensable pour sensibiliser, prévoir, et guider les politiques en matière de lutte contre le changement climatique. En conclusion, cette expérience souligne la pertinence de l'intégration des techniques de data analyse dans la recherche de solutions aux défis environnementaux actuels et futurs.<br><br>
            <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )


# PAGE 9  ***  PAGE 9  *** PAGE 9 *** PAGE 9 *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9 *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9  *** PAGE 9 *** PAGE 9  *** PAGE 9 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
