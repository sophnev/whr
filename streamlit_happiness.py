import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import pycountry
import geopandas

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.linear_model import LinearRegression    
from sklearn.tree import DecisionTreeRegressor


whr=pd.read_csv("world-happiness-report.csv")


    
whr2021=pd.read_csv("world-happiness-report-2021.csv")

#Traitement de whr2023
whr2023=pd.read_csv('whr2023.csv',sep=';')

# Nous choisissons de récupérer la notion de "région" dans le dataset annuel WHR2021
# Pour cela il faut d'abord procéder à l'homogénéisation des noms de pays dans les deux sources

corr_nom_pays = {
    "Turkiye": "Turkey",
    "State of Palestine": "Palestinian Territories",
    "Czechia": "Czech Republic",
    "Eswatini": "Swaziland"
}

whr2023["Country name"].replace(to_replace=corr_nom_pays, inplace=True)

whr2023=whr2023.merge(right=whr2021[["Country name", "Regional indicator"]] , on = "Country name", how = "left")

# identification des pays pour lesquels la région n'est pas renseignée 
whr2023[whr2023["Regional indicator"].isna()]["Country name"].value_counts()

# utilisation d'un dictionnaire pour renseigner leurs régions 
region_mapping = {
    "Angola": "Middle East and North Africa",
    "Belize": "Latin America and Caribbean",
    "Bhutan": "South Asia",
    "Central African Republic": "Middle East and North Africa",
    "Syria": "Middle East and North Africa",
    "Qatar": "Middle East and North Africa",
    "Sudan": "Middle East and North Africa",
    "Trinidad and Tobago": "Latin America and Caribbean",
    "Djibouti": "Middle East and North Africa",
    "Somalia": "Middle East and North Africa",
    "Somaliland region": "Middle East and North Africa",
    "Cuba": "Latin America and Caribbean",
    "Guyana": "Middle East and North Africa",
    "Oman": "Middle East and North Africa",
    "Suriname": "Latin America and Caribbean",
    "South Sudan": "Middle East and North Africa",
    "Congo (Kinshasa)": "Middle East and North Africa"
}

for x in range(len(whr2023)):
    country = whr2023.loc[x, "Country name"]
    if country in region_mapping:
        whr2023.loc[x, "Regional indicator"] = region_mapping[country]

#Ajustement du format des variables
variables = ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', \
             'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect']

for col in variables:
    whr2023[col] = whr2023[col].str.replace(',', '.')
    whr2023[col] = whr2023[col].astype(float)


num_data_2021=whr2021.select_dtypes(include='float')
matrice_cor = num_data_2021.iloc[:,4:10].corr()


st.title("Projet World Happiness - Datascientest 2024")

st.sidebar.title("Sommaire")

pages=["Le Projet", 
       "Présentation des jeux de données",
       "Exploration et analyse des données",
       "Nettoyage et pré-processing",
       "Modélisation globale",
       "Modélisation régionale",
       "Pour aller plus loin..."]

page=st.sidebar.radio("Aller vers",pages)

st.sidebar.title('Auteurs :')
st.sidebar.write('[Marie ENAULT](http://linkedin.com/in/mchavoutier)')
st.sidebar.write('[Joseph GICQUEL](https://www.linkedin.com/in/jgicquel)')
st.sidebar.write('[Sophie NEVANEN](http://www.linkedin.com/in/sophienevanen)')

st.sidebar.image('logo_datascientest.png', width=100)

if page == pages[0] :
    st.title('Introduction au projet')
    
    st.image('bonheur.jpeg')
    st.markdown('Dans ce projet nous allons effectuer une analyse approfondie des données collectées par le World Happiness Report (WHR).Cette enquête a pour objectif d’estimer le bonheur des pays autour de la planète à l’aide de mesures socio-économiques autour de la santé, l’éducation, la corruption, l’économie, l’espérance de vie, etc.') 
    st.markdown('La mesure du bonheur utilisée dans les WHR repose sur l’échelle du bonheur, ou échelle de Cantril : il s’agit de l’un des premiers instruments introduits dans la quête métrique de la satisfaction de vie. On mesure le bonheur d’une population à partir de la réponse à la question suivante : « Voici une échelle de 0 à 10 qui représente l’échelle de la vie. Supposons que le sommet de l’échelle représente la vie la meilleure pour vous, et le bas de l’échelle la vie la pire pour vous. Où vous situez-vous personnellement sur cette échelle en ce moment ? ».')
    st.markdown('L’objectif de notre projet est de présenter ces données à l’aide de visualisations interactives bien pensées et de déterminer les combinaisons de facteurs permettant d’expliquer pourquoi certains pays sont mieux classés que les autres.')

if page == pages[1] :
    st.title("Présentation des jeux de données")
    
    st.header('Les jeux fournis dans kaggle')
    
    st.write('Le jeu de données initial comporte 2 fichiers :')
    st.subheader('world_happiness_report_2021.csv :')
    
    st.write('Taille du dataframe :', whr2021.shape)

    st.write('Ce fichier nous fournit le score de bonheur 2021 par pays ainsi que diverses variables potentiellement explicatives')
    # Affichage de la carte de bonheur
    
    def alpha3code(column):
        CODE=[]
        for country in column:
            try:
                code=pycountry.countries.get(name=country).alpha_3
                CODE.append(code)
            except:
                    CODE.append('None')
        return CODE
    
    # create a column for code
    whr2021['CODE']=alpha3code(whr2021['Country name'])
    whr2021.head()
    
    #Correction des codes manquants
    whr2021.loc[17,'CODE']='CZE'
    whr2021.loc[23,'CODE']='TWN'
    whr2021.loc[32,'CODE']='-99'
    whr2021.loc[61,'CODE']='KOR'
    whr2021.loc[64,'CODE']='MDA'
    whr2021.loc[68,'CODE']='BOL'
    whr2021.loc[73,'CODE']='CYP'
    whr2021.loc[75,'CODE']='RUS'
    #whr2021.loc[76,'CODE']=''#HONGKONG
    whr2021.loc[78,'CODE']='VNM'
    whr2021.loc[82,'CODE']='COG'
    whr2021.loc[84,'CODE']='CIV'
    whr2021.loc[99,'CODE']='LAO'
    whr2021.loc[106,'CODE']='VEN'
    whr2021.loc[117,'CODE']='IRN'
    #whr2021.loc[124,'CODE']=''#Territoire palestinien
    #whr2021.loc[129,'CODE']=''#Swaziland
    whr2021.loc[141,'CODE']='TZA'
    #whr2021.loc[17]

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world.columns=['pop_est', 'continent', 'name', 'CODE', 'gdp_md_est', 'geometry']
    merge=pd.merge(world,whr2021, on='CODE', how='outer')
    location=pd.read_csv('https://raw.githubusercontent.com/melanieshi0120/COVID-19_global_time_series_panel_data/master/data/countries_latitude_longitude.csv')
    merge=merge.merge(location,how='left',on='name').sort_values(by='Ladder score',ascending=False).reset_index()
    
    fig = px.choropleth(merge, 
                    locations="CODE", 
                    color="Ladder score",
                    hover_name="name",
                    hover_data=["Ladder score"],
                    projection="equirectangular",
                    color_continuous_scale='rdylgn',  # Change colorscale here
                    title='WHR 2021 Score de bonheur par pays')
    
    # Make the plot larger
    fig.update_layout(height=500, width=1000)
    
    st.plotly_chart(fig)
    
    st.write('**Description des variables :**')
    
    variable_options = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    variable = st.selectbox('Selectionnez une variable', variable_options)
    
    if variable == "Logged GDP per capita" :
        st.write('Le produit intérieur brut par habitant est un indicateur du niveau d’activité économique, il correspond au produit intérieur brut du pays rapporté au nombre d’habitants.')
    elif variable == "Social support" :
        st.write('Proportion d’individus répondant oui à la question suivante “Si vous étiez en difficulté, avez-vous des parents ou des amis sur lesquels vous pouvez compter pour vous aider chaque fois que vous en avez besoin ?”')
    elif variable == "Healthy life expectancy" :
        st.write("Espérance de vie en bonne santé, c'est-à-dire sans limitation irréversible d'activité dans la vie quotidienne ni incapacités.")
    elif variable == "Freedom to make life choices" :
        st.write('Proportion d’individus répondant oui à la question suivante “Dans votre pays, êtes-vous satisfait de votre liberté de choisir ce que vous faites de votre vie?”')
    elif variable == "Generosity" :
        st.write("proportion d’individus répondant oui à la question suivante “Avez-vous donné de l'argent à un organisme de bienfaisance au cours du mois dernier ?”")
    elif variable == "Perceptions of corruption" :
        st.write("Proportion moyenne d’individus répondant oui aux 2 questions suivantes : “La corruption est-elle répandue au sein des entreprises situées dans votre pays ?” et “La corruption est-elle répandue au sein du gouvernement de votre pays ?”")

    st.write('**Carte des scores pour ', variable, ':**')

    
    merge2=merge.merge(location,how='left',on='name').sort_values(by=variable,ascending=False).reset_index()
    
    if variable not in merge.columns:
        st.error(f"The selected variable '{variable}' is not available in the dataset.")
    else:
        merge = merge.sort_values(by=variable, ascending=False).reset_index()

        fig = px.choropleth(merge,
                            locations="CODE",
                            color=variable,
                            hover_name="name",
                            hover_data=[variable],
                            projection="equirectangular",
                            color_continuous_scale='rdylgn')

        fig.update_layout(height=500, width=1000)
        
        st.plotly_chart(fig)

    if st.checkbox("Afficher les 10 premières lignes du dataframe whr 2021") :
        st.dataframe(whr2021.head(10))

    if st.checkbox("Afficher la description du dataframe whr 2021") :        
        st.dataframe(whr2021.describe())

    if st.checkbox("Afficher les NA de whr2021") :
        st.dataframe(whr2021.isna().sum())

    st.subheader("world-happiness-report.csv : ")

    st.write("Ce fichier nous fournit un historique de ce score de bonheur entre 2005 et 2020 par pays ainsi que les historiques des mêmes variables explicatives")
    
    st.write('Taille du dataframe :', whr.shape)
    
    if st.checkbox("Afficher les 10 premières lignes du dataframe") :
        st.dataframe(whr.head(10))

    if st.checkbox("Afficher la description du dataframe whr") :  
        st.dataframe(whr.describe())
    
    if st.checkbox("Afficher les NA de whr") :
        st.dataframe(whr.isna().sum())
    
    st.header('Limites')
    st.write('**Certaines variables sont subjectives :**')
    lst = ['Soutien social', 'Liberté de choix', 'Générosité', 'Perception de corruption']

    s = ''

    for i in lst:
        s += "- " + i + "\n"

    st.markdown(s)
    
    st.write("**L'historique est incomplet**")
    st.write('Beaucoup trop peu de pays sont disponibles en 2005 (15% par rapport aux années les plus complètes).')
    
    fig = plt.figure(figsize=(10,6))
    sns.countplot(x=whr.year)
    st.pyplot(fig)
    
    col = ["NA_total"] + list(range(2005, 2023))
    NB_NA = pd.DataFrame(index=whr2023.columns, columns=col)

    for j in whr2023.columns:
        for i in range(2005, 2023):
            NB_NA.at[j, i] = whr2023.loc[whr2023["year"] == i, j].isnull().sum()

    NB_NA["NA_total"] = whr2023.isnull().sum()
    
    st.write('**Nombre de valeurs manquantes par variable par année :**')
    st.table(NB_NA)
    
    

if page == pages[2] :
    st.title('Exploration et analyse des données')
    
    st.header('Etude des corrélations')
    # Sélection des variables de type Float
    num_data_2021=whr2021.select_dtypes(include='float')

    st.write('matrice des corrélations')
    matrice_cor = num_data_2021.iloc[:,[0] + list(range(4, 10))].corr()

    fig, ax = plt.subplots()
    sns.heatmap(matrice_cor, annot = True, ax = ax, cmap = "vlag")
    st.pyplot(fig)

    st.subheader('Distribution des variables sur le jeu whr2021') 
    
    variable_options = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    variable = st.selectbox('Selectionnez une variable', variable_options)

    # suppression années 2005 trop peu représentée
    whr2023 = whr2023[whr2023["year"] != 2005]

    fig = plt.figure()
    sns.histplot(num_data_2021[variable])
    plt.title(variable)
    st.pyplot(fig)
    
    # Calculating average indicators per year (worldwide)
    colors = ["k", "m", "y", "r", "g", "c", "b", "m", "y"]

    whr_moy_mondiale = whr2023.groupby("year").agg({
         'Life Ladder': 'mean',
         'Log GDP per capita': 'mean',
         'Social support': 'mean',
         'Healthy life expectancy at birth': 'mean',
         'Freedom to make life choices': 'mean',
         'Generosity': 'mean',
         'Perceptions of corruption': 'mean',
         'Positive affect': 'mean',
         'Negative affect': 'mean',
         'Country name': lambda x: len(x.unique())
         }).reset_index()
    
    st.subheader('Evolution base 100 des différents indicateurs dans le temps (monde entier) hors "Generosity"')
    whr_moy_mondiale2=whr_moy_mondiale.drop(columns='Generosity')

    colors = ["k", "m", "y", "r", "g", "b", "m", "y"]

    fig, ax = plt.subplots(figsize=[15, 8])

    ax.plot(whr_moy_mondiale2["year"], whr_moy_mondiale2.iloc[:, 1] / whr_moy_mondiale2.iloc[0, 1], \
            c="k", label=whr_moy_mondiale2.columns[1])

    for i in range(2, 9):
        ax.plot(whr_moy_mondiale2["year"], whr_moy_mondiale2.iloc[:, i] / whr_moy_mondiale2.iloc[0, i], \
                c=colors[i - 1], linestyle="--", label=whr_moy_mondiale2.columns[i])

    ax.set_xlabel("Année")
    ax.set_xlim([2006, 2022])
    ax.set_xticks(np.arange(2006, 2023, 2))
    ax.legend()

    st.pyplot(fig)
    
    st.subheader('Evolution base 100 des variables par région')
    
    variable2_options = ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Perceptions of corruption']
    variable2 = st.selectbox('Selectionnez une variable', variable2_options)

    # Map selected variable to column name
    variable_column_mapping = {
        'Life Ladder': 'Life Ladder',
        'Log GDP per capita': 'Log GDP per capita',
        'Social support': 'Social support',
        'Healthy life expectancy at birth': 'Healthy life expectancy at birth',
        'Freedom to make life choices': 'Freedom to make life choices',
        'Perceptions of corruption': 'Perceptions of corruption'
        }

    # Calcul des indicateurs moyen par région et par année
    whr_moy_region = whr2023.groupby(["year","Regional indicator"]).agg({
        'Life Ladder': 'mean',
        'Log GDP per capita': 'mean',
        'Social support': 'mean',
        'Healthy life expectancy at birth': 'mean',
        'Freedom to make life choices': 'mean',
        'Perceptions of corruption': 'mean',
        'Positive affect': 'mean',
        'Negative affect': 'mean',
        'Country name': lambda x: len(x.unique())
        }).reset_index()

    # Illustration des variations par région : cas de l'évolution base 100 par région de la liberté de choix

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "yellow"]

    fig, ax = plt.subplots(figsize=[15, 8])

    for i, region in enumerate(whr_moy_region["Regional indicator"].unique()):
        region_data = whr_moy_region[whr_moy_region["Regional indicator"] == region]
        ax.plot(region_data["year"], region_data[variable_column_mapping[variable2]] / region_data.iloc[0][variable_column_mapping[variable2]], c=colors[i], label=region)

    ax.set_xlabel("Année")
    ax.set_xlim([2006, 2022])
    plt.title('évolution base 100 par région')
    ax.set_xticks(np.arange(2006, 2022, 2))
    ax.legend()

    st.pyplot(fig)
    
    st.write('**Quelques remarques :**')
    st.write('PIB par habitant ainsi que l’espérance de vie progressent de façon sensiblement linéaire quelque soit la région')
    st.write('Le soutien social et la liberté de choix ont des évolutions plus variées dans le temps, au niveau mondial comme au niveau régional')
    st.write('Le score du bonheur semble suivre les tendances de ces deux indicateurs, mais pas forcément de la même façon en fonction des régions.')
    st.write("D'où la question de la pertinence d'un modèle mondial ?")

if page == pages[3] :
    st.title('Nettoyage et pré-processing')
    
    st.subheader('Gestion des NaNs')
    
    st.write('Nous faisons le choix pour l’analyse des évolutions de ne pas supprimer de Pays de l’analyse même lorsque très peu d’années sont disponibles.')
    st.write("En effet, nous constatons qu'aucun pays n’est présent sur chacun des 18 ans d’historique total.")

    df = pd.DataFrame(columns=["Regional indicator", "Pays", "nb_année_dispo"])

    for i in whr2023["Country name"].unique() :
        Région = whr2023["Regional indicator"].loc[whr2023["Country name"] == i].mode()[0]
        nb_annees_pays = whr2023[whr2023["Country name"] == i].shape[0]
        df1 = pd.DataFrame(np.array([[Région,i,nb_annees_pays]]),columns=["Regional indicator","Pays","nb_année_dispo"])
        df = pd.concat([df,df1],axis = 0)
    
    df['nb_année_dispo']=df['nb_année_dispo'].astype("int")

    nb_pays_par_année = pd.DataFrame(df['nb_année_dispo'].value_counts().sort_index(ascending = False))
    nb_pays_par_année["cumul"] = nb_pays_par_année["count"].cumsum()/165

    fig, ax = plt.subplots()
    plt.plot(nb_pays_par_année.index,nb_pays_par_année["cumul"]*100,"b:+")
    plt.title("% Pays conservés en fonction du minimum d'années accepté")
    plt.ylim([0,110]);

    st.pyplot(fig)
    
    st.write('Conserver uniquement les pays avec plus de 6 années disponibles revient à supprimer 12% des pays (20 pays), avec une surreprésentation sur certaines régions mondiales :')

    region1_table = ['Latin America and Caribbean', 24, '4 soit 17%']
    region2_table = ['Middle East and North Africa', 29, '10 soit 34%']
    region3_table = ['South Asia', 8, '2 soit 25%']
    region4_table = ['Sub-Saharan Africa', 36, '4 soit 11%']
    
    table = pd.DataFrame([region1_table, region2_table, region3_table, region4_table], columns=['Région', 'nb total pays', '% pays avec peu d’historique (<7)'])
    
    st.table(table)
    
    st.write('Nous reconstitutions donc un historique de la façon suivante :')
    st.write('Une seule année disponible : nous répliquons les données sur toutes les années du périmètre')
    st.write('Deux ou plus de deux années disponibles : nous faisons l’hypothèse d’une évolution linéaire entre les deux dates disponibles.')
    
    st.image('gestion nan.png')
    
    st.subheader('Ajout de variables')
    st.write('Pour compléter le jeu de données, nous choisissons d''ajouter des variables provenant de ONU et de la Banque Mondiale')
    lst = ['Labour force participation - Total : Taux de participation au marché du travail, c’est à dire le pourcentage de personnes agées de 16 ans et plus qui travaillent ou cherchent du travail',
           'Unemployment rate - Total : Taux de chômage',
           'Intentional homicide rates per 100,000 : nombre d’homicides volontaires pour 100 000 habitants',
           'Ratio of girls to boys in primary education : ratio du nombre de filles comparé au nombre de garçons à l’école primaire',
           'Seats held by women in national parliament : Pourcentage de sièges occupés par des femmes au Parlement du pays',
           'Percentage of individuals using the internet : Pourcentage de personnes utilisant Internet', 
           'Students enrolled in upper secondary education : Nombre d’étudiants en éducation supérieure en milliers',
           'Total fertility rate : Nombre d’enfants par femme',
           'Infant mortality for both sexes : Mortalité infantile, comptée pour 1000 naissances',
           'Health personnel : Nombre de médecins pour 1000 habitants',
           'Forest cover : Pourcentage de la surface du pays couverte par de la forêt',
           'Tourist/visitor arrivals : nombre de touristes (en millier)',
           'Émissions de CO2 (tonnes métriques par habitant)',
           'Émissions totales de GES (kt d’équivalent CO2)',
           'Population urbaine (% du total)',
           'Densité de la population : Nb de personnes par km²',
           'Temps de travail',
           '% Energies renouvelables',
           '% Agriculture dans le PIB',
           'Appauvrissement si recours santé',
           'Concentration richesse (Part des revenus globaux du pays détenus par les 20% de revenus les plus élevés)',
           'Expo particules fines - PM2,5 (% de la population exposée à des niveaux supérieurs à la valeur de référence de l’OMS)',
           'Pop urbaine sup 1M',
           'Inflation']
           
    s = ''

    for i in lst:
        s += "- " + i + "\n"

    st.markdown(s)

    st.write('Nous les traiterons de la même façon que nous avons traité les variables de notre jeu de données initial')

#suppression des variables Positive affect et Negative affect
whr2023 = whr2023.drop(["Positive affect","Negative affect"],axis = 1)

 #identification des pays pour lesquelles l'année est disponible
whr2023["Année_connue"] = 1

# Gestion des NaNs via méthode KNN (car corrélations) pour les variables corrélées
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=4)
val_num = whr2023[['Life Ladder','Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']]
val_num = imputer.fit_transform(val_num)
val_num = pd.DataFrame(val_num)

whr2023['Life Ladder'].fillna(val_num[0], inplace=True)
whr2023['Log GDP per capita'].fillna(val_num[1], inplace=True)
whr2023['Social support'].fillna(val_num[2], inplace=True)
whr2023['Healthy life expectancy at birth'].fillna(val_num[3], inplace=True)
whr2023['Freedom to make life choices'].fillna(val_num[4], inplace=True)

 # Créer un DataFrame contenant toutes les combinaisons possibles de "Country name" et d'années
all_combinations = pd.MultiIndex.from_product([whr2023["Country name"].unique(), range(2006, 2023)], names=["Country name", "year"])
all_combinations_df = pd.DataFrame(index=all_combinations).reset_index()

# Fusionner avec le DataFrame whr2023 initial en utilisant une jointure externe pour ajouter les lignes manquantes
whr2023 = pd.merge(all_combinations_df, whr2023, on=["Country name", "year"], how="left")

# Remplacer les valeurs manquantes par NaN
whr2023 = whr2023.fillna(float("nan"))
whr2023['Année_connue']=whr2023['Année_connue'].fillna(0)

#Interpolation pour les variables Generosity et Perceptions of corruption

whr_remp= whr2023[["Country name","year","Generosity","Perceptions of corruption"]]

for Pays in whr2023["Country name"].unique():
    whr_remp.loc[whr_remp["Country name"] == Pays] = whr_remp.loc[whr_remp["Country name"] == Pays].interpolate()

whr2023[["Generosity","Perceptions of corruption"]] = whr_remp[["Generosity","Perceptions of corruption"]]


#Compléter le Regional indicator pour les lignes où la région est absente
#remplissage vers l'avant puis vers l'arrière dans le même groupe de Country_name
whr2023['Regional indicator'] = whr2023.groupby('Country name')['Regional indicator'].ffill()
whr2023['Regional indicator'] = whr2023.groupby('Country name')['Regional indicator'].bfill()

#Compléter les valeurs sur les années "aux extrémités"

def fill_missing_values(df):
    # Sélectionner les colonnes contenant des NaN
    columns_with_missing_values = df.columns[df.isna().any()].tolist()
        
    # Remplir les valeurs manquantes avec ffill et bfill pour chaque colonne
    for column in columns_with_missing_values:
        df[column] = df.groupby('Country name')[column].ffill()
        df[column] = df.groupby('Country name')[column].bfill()
    
    return df

# Appeler la fonction pour remplir les valeurs manquantes dans le DataFrame whr2023
whr2023 = fill_missing_values(whr2023)

# Pour les quelques pays sans données sur générosité ou corruption, on remplace par la médiane
whr2023["Generosity"].fillna(whr2023["Generosity"].median(),inplace = True)
whr2023["Perceptions of corruption"].fillna(whr2023["Perceptions of corruption"].median(),inplace = True)

emploi=pd.read_csv('SYB66_329_202310_Labour Force and Unemployment.csv', encoding='latin-1',skiprows=1)
crime=pd.read_csv('SYB66_328_202310_Intentional homicides and other crimes.csv', encoding='latin-1',skiprows=1)
sexratio_edu=pd.read_csv('SYB66_319_202310_Ratio of girls to boys in education.csv', encoding='latin-1',skiprows=1)
sexratio_parlement=pd.read_csv('SYB66_317_202310_Seats held by women in Parliament.csv', encoding='latin-1',skiprows=1)
internet=pd.read_csv('SYB66_314_202310_Internet Usage.csv', encoding='latin-1',skiprows=1)
education=pd.read_csv('SYB66_309_202310_Education.csv', encoding='latin-1',skiprows=1)
fertilite=pd.read_csv('SYB66_246_202310_Population Growth, Fertility and Mortality Indicators.csv', encoding='latin-1',skiprows=1)
sante=pd.read_csv('SYB66_154_202310_Health Personnel.csv', encoding='latin-1',skiprows=1)
environnement=pd.read_csv('SYB66_145_202310_Land.csv', encoding='latin-1',skiprows=1)
tourisme=pd.read_csv('SYB66_176_202310_Tourist-Visitors Arrival and Expenditure.csv', encoding='latin-1',skiprows=1)

# Appliquer le même traitement aux datasets provenant de UNdata formattés de la même façon après avoir sélectionné
# les indicateurs qui nous intéressent à l'intérieur

def transformer_dataset(df):
    # Renommer les colonnes
    df = df.rename(columns={'Unnamed: 1': 'Country name', 'Year': 'year'})
    
    # Supprimer certaines colonnes
    colonnes_a_supprimer = ['Region/Country/Area', 'Footnotes', 'Source']
    df = df.drop(columns=colonnes_a_supprimer)
    
    # Convertir la colonne 'Value' en format numérique
    df['Value'] = df['Value'].astype(str).str.replace(',', '.').astype(float)
    
    # Transposer le dataframe
    df_pivot = df.pivot_table(index=['Country name', 'year'], columns='Series', values='Value').reset_index()
    
    return df_pivot

#traitement du dataset emploi
emploi=emploi[(emploi['Series']=='Labour force participation - Total')|(emploi['Series']=='Unemployment rate - Total')]
emploi=transformer_dataset(emploi)

#traitement du dataset crime
crime=crime[(crime['Series']=='Intentional homicide rates per 100,000')]
crime=transformer_dataset(crime)

#traitement du dataset sexratio_edu
sexratio_edu=sexratio_edu[(sexratio_edu['Series']=='Ratio of girls to boys in primary education')]
sexratio_edu=transformer_dataset(sexratio_edu)

#traitement du dataset sexratio_parlement
sexratio_parlement=sexratio_parlement[(sexratio_parlement['Series']=='Seats held by women in national parliament, as of February (%)')]
sexratio_parlement=transformer_dataset(sexratio_parlement)

#traitement du dataset internet
internet=internet[(internet['Series']=='Percentage of individuals using the internet')]
internet=transformer_dataset(internet)

#traitement du dataset education
education=education[(education['Series']=='Students enrolled in upper secondary education (thousands)')]
education=transformer_dataset(education)

#traitement du dataset fertilite
fertilite1=fertilite[(fertilite['Series']=='Total fertility rate (children per women)')]
fertilite1=transformer_dataset(fertilite1)
fertilite2=fertilite[(fertilite['Series']=='Infant mortality for both sexes (per 1,000 live births)')]
fertilite2=transformer_dataset(fertilite2)

#traitement du dataset santé
sante=sante[(sante['Series']=='Health personnel: Physicians (per 1000 population)')]
sante=transformer_dataset(sante)

#traitement du dataset environnement
environnement=environnement[(environnement['Series']=='Forest cover (% of total land area)')]
environnement=transformer_dataset(environnement)

#traitement du dataset tourisme
tourisme=tourisme[(tourisme['Series']=='Tourist/visitor arrivals (thousands)')]
tourisme=transformer_dataset(tourisme)



# Effectuer toutes les jointures à la fois
whr2023_enrichi = whr2023  # Initialiser le dataframe avec le dataframe contenant l'ensemble des données du WHR

# Liste des dataframes à fusionner
dataframes = [emploi, crime, sexratio_edu, sexratio_parlement, internet, education, fertilite1, fertilite2, sante, environnement, tourisme]

for df in dataframes:
    whr2023_enrichi = pd.merge(whr2023_enrichi, df, on=['Country name', 'year'], how='left')
    

# Ajout des variables issues de l'OCDE
data_ocde= pd.read_excel('Data2.xlsx', sheet_name = "Data")
data_ocde.rename(columns={'Country_name': 'Country name'}, inplace=True)

whr2023_enrichi = pd.merge(whr2023_enrichi, data_ocde, on=['Country name', 'year'], how='left')

# Interpolation
# Remplacement des NaN Pays via la moyenne de l'évolution

for Pays in whr2023_enrichi["Country name"].unique() :
    whr2023_enrichi[whr2023_enrichi["Country name"]==Pays] = whr2023_enrichi[whr2023_enrichi["Country name"]==Pays].interpolate()

whr2023_enrichi = whr2023_enrichi.reset_index().drop(["index"],axis=1)

whr2023_enrichi.info()

# Suppression des colonnes avec plus de 10% de valeurs manquantes
def drop_columns_with_high_nan(df, threshold=10):
    # Initialiser une liste pour stocker les colonnes à supprimer
    cols_to_drop = []
    
    # Parcourir les colonnes du DataFrame
    for column in df.columns:
        # Calculer le pourcentage de NaN pour chaque colonne
        nan_percentage = df[column].isna().mean() * 100
        
        # Si le pourcentage de NaN dépasse le seuil spécifié, ajouter la colonne à la liste des colonnes à supprimer
        if nan_percentage > threshold:
            cols_to_drop.append(column)
    
    # Supprimer les colonnes de la DataFrame
    df_cleaned = df.drop(cols_to_drop, axis=1)
    
    return df_cleaned

# Appeler la fonction pour supprimer les colonnes avec plus de 10% de NaN
df_mondial = drop_columns_with_high_nan(whr2023_enrichi, threshold=10)

# Remplacement des données manquantes sur tout un pays
for i in range(11,19) :
    df_mondial[df_mondial.columns[i]]= df_mondial[df_mondial.columns[i]].fillna(df_mondial[df_mondial.columns[i]].median())

# Etude des corrélations pour supprimer les variables non corrélées

matrice_cor = df_mondial.drop(["Regional indicator"],axis=1).iloc[:,2:].corr()

fig, ax = plt.subplots(figsize = (15,15))
sns.heatmap(matrice_cor, annot = True, ax = ax, cmap = "vlag");

# Liste des variables très faiblement corrélées avec "Life Ladder"
liste_corr = [i for i in matrice_cor.columns if abs(matrice_cor.at["Life Ladder", i]) < 0.3]

# Suppression des variables très faiblement corrélées dans df_mondial
df_mondial = df_mondial.drop(liste_corr, axis=1)

# Matrice de corrélation mise à jour
matrice_cor2 = df_mondial.drop(["Regional indicator"], axis=1).iloc[:, 2:].corr()

    
if page == pages[4] :
    
    df_mondial_final = pd.read_csv('df_mondial.csv')

    st.title('Modélisation globale')

    st.header('choix du modèle :')

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestRegressor(max_depth=5)
        elif classifier == 'Decision Tree':
            clf = DecisionTreeRegressor(max_depth = 5)
        elif classifier == 'Linear Regression':
            clf = LinearRegression()
        return clf

    choix_modele = ['Linear Regression', 'Random Forest', 'Decision Tree']
    modele_choisi = st.selectbox('Choix du modèle', choix_modele)

    choix_dataset = ['Complet', 'sans interpolation', 'sans région', 'sans année']
    dataset_choisi = st.selectbox('Choix du dataset', choix_dataset)

    if dataset_choisi == 'Complet':
        X = pd.read_csv('X_modele1.csv')
        y = pd.read_csv('y_modele1.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        X_train = pd.concat([X_train,pd.get_dummies(X_train["year"], prefix='Année')],axis = 1)
        X_test = pd.concat([X_test,pd.get_dummies(X_test["year"], prefix='Année')],axis = 1)
        X_train = pd.concat([X_train,pd.get_dummies(X_train["Regional indicator"], prefix='Région')],axis = 1)
        X_test = pd.concat([X_test,pd.get_dummies(X_test["Regional indicator"], prefix='Région')],axis = 1)

        X_train = X_train.drop(['year','Regional indicator'], axis = 1)
        X_test = X_test.drop(['year','Regional indicator'], axis = 1)
        
        scaler = StandardScaler()

        X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.fit_transform(X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])
        X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.transform(X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])
        
    elif dataset_choisi == 'sans interpolation':
        X = pd.read_csv('X_modele2.csv')
        y = pd.read_csv('y_modele2.csv')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        X_train = pd.concat([X_train,pd.get_dummies(X_train["year"], prefix='Année')],axis = 1)
        X_test = pd.concat([X_test,pd.get_dummies(X_test["year"], prefix='Année')],axis = 1)
        X_train = pd.concat([X_train,pd.get_dummies(X_train["Regional indicator"], prefix='Région')],axis = 1)
        X_test = pd.concat([X_test,pd.get_dummies(X_test["Regional indicator"], prefix='Région')],axis = 1)

        X_train = X_train.drop(['year','Regional indicator'], axis = 1)
        X_test = X_test.drop(['year','Regional indicator'], axis = 1)

        scaler = StandardScaler()

        X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.fit_transform(X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])
        X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.transform(X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])

        
    elif dataset_choisi == 'sans région':
        X = pd.read_csv('X_modele3.csv')
        y = pd.read_csv('y_modele3.csv')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        scaler = StandardScaler()

        X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.fit_transform(X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])
        X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.transform(X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])

    elif dataset_choisi == 'sans année':
        X = pd.read_csv('X_modele4.csv')
        y = pd.read_csv('y_modele4.csv')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        scaler = StandardScaler()

        X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.fit_transform(X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])
        X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.transform(X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])

    clf = prediction(modele_choisi)
    clf.fit(X_train, y_train)

    st.write("score train : ", clf.score(X_train, y_train))
    st.write("score test : ", clf.score(X_test, y_test))

    
    st.write('Nous utiliserons le modèle Random Forest Regressor')
    
    st.header("premières constatations")
    st.subheader('Impact de l’interpolation des données')
    
    st.write('Contre-intuitivement, les résultats (score de tests) sont meilleurs sur le Dataset complet c’est-à-dire avec reconstitution de l’historique.')
    st.write('Nous expliquons cela par le fait que reconstituer un historique identique sur l’ensemble des pays supprime un biais dans les données : les données sont probablement davantage disponibles sur une typologie de pays précis.')
    st.write('Cela se confirme par la visualisation de la répartition des scores de bonheur qui, avec le dataset complet se rapproche d’une loi normale')
    
    from scipy.stats import norm
    
    if st.checkbox("Afficher le graphique") :
        domain = np.linspace(2,8)
        pdf_norm = norm.pdf(domain,loc = 5.3, scale = 1.2)
    
        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Plot the normal distribution
        ax.plot(domain, pdf_norm, color="black", label="loi normale (5.3,1.2)")

        # Plot the distributions for interpolated and original data using Seaborn
        sns.distplot(df_mondial_final["Life Ladder"], bins=10, kde=True, rug=True, color='#FFD700', label="Data interpolé", ax=ax)
        sns.distplot(whr["Life Ladder"], bins=10, kde=True, rug=True, color='#7EC0EE', label="Data origine", ax=ax)

        # Display legend
        plt.legend()

        # Display the plot using Streamlit
        st.pyplot(fig)
    
    st.subheader('Importance de la Région')
    st.write('Nous constatons par ailleurs une diminution des résultats avec la suppression de l’information “Région” mais, par contre, pas de changement avec la suppression de l’information de l’année.')
    st.write('Comme nous le pressentions dans la phase d’analyse, les hommes ne sont pas toujours sensibles aux mêmes composantes du bien-être. Le temps ne semble pas avoir d’impact mais les régions ont leur importance.')
    st.write('Néanmoins, l’étude des composantes principales nous alerte sur l’impact de la donnée Région prise en l’état.')
    
    X = df_mondial_final[['Log GDP per capita','Healthy life expectancy at birth','Freedom to make life choices','Social support'\
                ,'year','Regional indicator']]
    y = df_mondial_final['Life Ladder']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    X_train = pd.concat([X_train,pd.get_dummies(X_train["year"], prefix='Année')],axis = 1)
    X_test = pd.concat([X_test,pd.get_dummies(X_test["year"], prefix='Année')],axis = 1)
    X_train = pd.concat([X_train,pd.get_dummies(X_train["Regional indicator"], prefix='Région')],axis = 1)
    X_test = pd.concat([X_test,pd.get_dummies(X_test["Regional indicator"], prefix='Région')],axis = 1)

    X_train = X_train.drop(['year','Regional indicator'], axis = 1)
    X_test = X_test.drop(['year','Regional indicator'], axis = 1)

    scaler = StandardScaler()

    X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.fit_transform(X_train[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])
    X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']] = scaler.transform(X_test[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices']])

    dt_reg = DecisionTreeRegressor(max_depth = 5)
    dt_reg.fit(X_train, y_train)

      
    
    feats = {}
    for feature, importance in zip(X_train.columns, dt_reg.feature_importances_):
        feats[feature] = importance 
    
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importances= importances.sort_values(by='Importance', ascending=False).head(8)

    fig = plt.figure(figsize = (20, 10))

    plt.bar(importances.head(5).index,importances.head(5)["Importance"])
    
    st.pyplot(fig)
    
    if st.checkbox("Afficher la table des variables les plus importantes") :
        st.table(importances.head(5))

    st.subheader('Importance du PIB')    
    st.write('Dernière constatation : le PIB a un impact extrêmement important sur les modèles. En l’état, il rend les autres données presque inutiles.')
    st.write('Nous tenterons de supprimer la donnée afin de dégager si possible d’autres axes prédictifs sans perdre en performance de modèle.')

    st.header("Choix des paramétrages")
    st.write('Hypothèses retenues')
    lst = ['Modèle retenu : RandomForestRegressor',
           'Dataset interpolé',
           'Abandon des variables année et Région']
           
    s = ''

    for i in lst:
        s += "- " + i + "\n"

    st.markdown(s)
    
    st.subheader('Etudes ajouts des variables')     
    
    choix_dataset2 = ['Initial, avec PIB', 'Initial, sans PIB', 'Complété, avec PIB', 'Complété, sans PIB']
    dataset2_choisi = st.selectbox('Choix du dataset', choix_dataset2)
    
    if dataset2_choisi == 'Initial, avec PIB':
        X = df_mondial_final[['Log GDP per capita','Healthy life expectancy at birth','Freedom to make life choices',\
                'Social support','Generosity','Perceptions of corruption']]
        y = df_mondial_final['Life Ladder']

    elif dataset2_choisi == 'Initial, sans PIB':
        X = df_mondial_final[['Healthy life expectancy at birth','Freedom to make life choices',\
                'Social support','Generosity','Perceptions of corruption']]
        y = df_mondial_final['Life Ladder']
        
    elif dataset2_choisi == 'Complété, avec PIB':
        X = df_mondial_final.drop(['Life Ladder','Country name',"Regional indicator","year"], axis = 1)
        y = df_mondial_final['Life Ladder']
    
    elif dataset2_choisi == 'Complété, sans PIB':
        X = df_mondial_final.drop(['Life Ladder','Country name',"Regional indicator","year",'Log GDP per capita'], axis = 1)
        y = df_mondial_final['Life Ladder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_reg = RandomForestRegressor(max_depth = 4)
    rf_reg.fit(X_train, y_train)

    st.write("score train : " , rf_reg.score(X_train, y_train))
    st.write("score test : ", rf_reg.score(X_test,y_test))
    
    X_train = pd.DataFrame(X_train,columns = X.columns)

    feats = {}
    for feature, importance in zip(X_train.columns, rf_reg.feature_importances_):
        feats[feature] = importance 
    
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importances= importances.sort_values(by='Importance', ascending=False).head(8)

    fig = plt.figure(figsize = (20, 10))

    plt.bar(importances.head(5).index,importances.head(5)["Importance"])
    
    st.pyplot(fig)

    if st.checkbox("Afficher la table des variables les plus importantes pour le dataset choisi") :
        st.table(importances.head(5))
        
    st.subheader('Conclusion au niveau mondial')
    
    df_mondial_ccl = pd.read_csv('df_mondial.csv')
    
    st.write('Avec le modèle retenu, les 5 composantes les plus importantes du bonheur sont les suivantes :')

    if st.checkbox('supprimer les données subjectives') :
        X = df_mondial_ccl.drop(['Life Ladder','Country name',"Regional indicator","year","Log GDP per capita",\
                            "Social support","Freedom to make life choices","Generosity","Perceptions of corruption"], axis=1)
        y = df_mondial_ccl['Life Ladder']
    else :
        X = df_mondial_ccl.drop(['Life Ladder', 'Country name', "year", "Regional indicator", "Log GDP per capita"], axis=1)
        y = df_mondial_ccl['Life Ladder']
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_reg = RandomForestRegressor(max_depth=5)
    rf_reg.fit(X_train_scaled, y_train)
            
    X_train = pd.DataFrame(X_train,columns = X.columns)

    feats = {}
    for feature, importance in zip(X_train.columns, rf_reg.feature_importances_):
        feats[feature] = importance 
            
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importances= importances.sort_values(by='Importance', ascending=False).head(8)

    fig = plt.figure(figsize = (20, 10))
    plt.bar(importances.head(5).index,importances.head(5)["Importance"])
    st.pyplot(fig)
            
    st.table(importances)
    

if page == pages[5] :
    st.title('Modélisation régionale')
    
    st.write('Nous allons analyser les variables les plus importantes région par région.')
    
    # Création du fichier de base Dataset_région
    st.write("Aucune variable n'est supprimée en amont : nous sélectionnerons les variables représentatives (ie Nans < 10%) après filtre sur la région étudiée")


    df_tot_region=pd.read_csv('df_tot_region.csv')
    
    st.write(df_tot_region['Regional indicator'].value_counts())
    
    regions_options = ['Sub-Saharan Africa',
                       'Middle East and North Africa',
                       'Latin America and Caribbean',
                       'Western Europe',
                       'Central and Eastern Europe',
                       'Commonwealth of Independent States',
                       'Southeast Asia',
                       'South Asia',
                       'East Asia',
                       'North America and ANZ']
    region_choisie = st.multiselect('Selectionnez une ou plusieurs régions', regions_options, default='Western Europe')

    # Composition des dataset régionaux

    def preparation_df_region (df,regions) :
           
        # Filtre sur les pays des régions à étudier
        data_region = df[df["Regional indicator"].isin(regions)]
    
        # Suppression des indicateurs trop peu présents sur la région (seuil choisi à 10%)
        valeurs_manquantes = data_region.isnull().sum(axis=0) / len(data_region) * 100
        colonnes_conservees = valeurs_manquantes[valeurs_manquantes <= 10].index
        data_region = data_region[colonnes_conservees]

        # Filtrer les colonnes de type float
        colonnes_float = data_region.select_dtypes(include='float').columns
    
        # Calcul de la médiane pour les colonnes de type float
        medianes_par_colonne = data_region[colonnes_float].median()

        # Remplacement des valeurs manquantes par la médiane de la région sur chaque variable
        for colonne in colonnes_float:
            data_region[colonne].fillna(medianes_par_colonne[colonne], inplace=True)

        return data_region
    
    choix_données = ['Ensemble des données objectives et subjectives', 'Données objectives uniquement']
    données_choisies = st.selectbox('Choix des données à prendre en compte', choix_données)


    def application_df_region(regions):
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        
        df_region = preparation_df_region(df_tot_region, regions)
        st.write(df_region.info())
        
        if données_choisies == 'Ensemble des données objectives et subjectives' :
        
            # Préparation à la modélisation "ensemble des données objectives et subjectives"
            X = df_region.drop(['Life Ladder', 'Country name', "year", "Regional indicator", "Log GDP per capita"], axis=1)
            y = df_region['Life Ladder']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf_reg = RandomForestRegressor(max_depth=5)
            rf_reg.fit(X_train_scaled, y_train)
            st.write("\n \n")
            st.write("Performance du modèle incluant l'ensemble des données objectives et subjectives")
            st.write("\n")
            st.write("Score d'entraînement :", rf_reg.score(X_train_scaled, y_train))
            st.write("Score de test :", rf_reg.score(X_test_scaled, y_test))


            # Extraction des variables les plus importantes
            st.write("\n \n")
            st.write("Importance des variables du modèle incluant les données objectives et subjectives")
            st.write("\n")
            
            X_train = pd.DataFrame(X_train,columns = X.columns)

            feats = {}
            for feature, importance in zip(X_train.columns, rf_reg.feature_importances_):
                feats[feature] = importance 
            
            importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
            importances= importances.sort_values(by='Importance', ascending=False).head(8)

            fig = plt.figure(figsize = (20, 10))
            plt.bar(importances.head(5).index,importances.head(5)["Importance"])
            st.pyplot(fig)
            
            st.table(importances)
  
        if données_choisies == 'Données objectives uniquement' :
            # Préparation à la modélisation "sélection des données objectives uniquement"
            X = df_region.drop(['Life Ladder','Country name',"Regional indicator","year","Log GDP per capita",\
                            "Social support","Freedom to make life choices","Generosity","Perceptions of corruption"], axis=1)
            y = df_region['Life Ladder']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf_reg = RandomForestRegressor(max_depth=5)
            rf_reg.fit(X_train_scaled, y_train)
            st.write("\n \n")
            st.write("Performance du modèle incluant les données objectives uniquement")
            st.write("\n")
            st.write("Score d'entraînement :", rf_reg.score(X_train_scaled, y_train))
            st.write("Score de test :", rf_reg.score(X_test_scaled, y_test))

            # Extraction des variables les plus importantes
            st.write("\n \n")
            st.write("Importance des variables du modèle incluant les données objectives uniquement")
            st.write("\n")
            
            X_train = pd.DataFrame(X_train,columns = X.columns)

            feats = {}
            for feature, importance in zip(X_train.columns, rf_reg.feature_importances_):
                feats[feature] = importance 
            
            importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
            importances= importances.sort_values(by='Importance', ascending=False).head(8)

            fig = plt.figure(figsize = (20, 10))
            plt.bar(importances.head(5).index,importances.head(5)["Importance"])
            st.pyplot(fig)
            
            st.table(importances)
        
    application_df_region(region_choisie)
    
if page == pages[6] :
    st.title('Pour aller plus loin')
    
    st.write('Limites de notre analyse :')
    st.write('Malheureusement, ces premiers résultats sont à relativiser compte tenu des données et de leur disponibilité très disparate d’une Région à l’autre :')
    st.write('Les scores obtenus montrent notamment que sur certaines Régions les modèles ne sont pas suffisamment robustes en l’état (trop peu de pays disponibles et/ou nombres de données insuffisants ou non pertinentes)')
    st.write('Les comparaisons entre les Régions restent limitées car trop peu de Régions ont suffisamment de données communes')
 
    st.write('Il serait intéressant d’affiner cette analyse avec :')
    st.write('La constitution d’une base de données plus homogène')
    st.write('L’étude d’une autre répartition des pays en fonction de la sensibilité des populations à certains facteurs, et non plus uniquement à l’aspect géographique')
    st.write('L’essai d’autres types de modélisation permettant peut-être d’optimiser la robustesse des modèles régionaux')