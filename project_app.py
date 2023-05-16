#Data munging

import pandas as pd

url = "https://raw.githubusercontent.com/pekpekkis/Data-Management-Group-Project/main/annual_co2_emissions.csv"
df_co2 = pd.read_csv(url)

url2 = "https://raw.githubusercontent.com/pekpekkis/Data-Management-Group-Project/main/annual_renewable_energy.csv"
df_green_energy = pd.read_csv(url2)

url3 = "https://raw.githubusercontent.com/pekpekkis/Data-Management-Group-Project/main/annual_population.csv"
df_population = pd.read_csv(url3, on_bad_lines='skip')
df_population = df_population.loc[:, ['Country name', 'Year', 'Population']]
df_population = df_population.rename(columns={'Country name': 'Entity'})

url4 = "https://raw.githubusercontent.com/pekpekkis/Data-Management-Group-Project/main/GDP.csv"
df_GDP = pd.read_csv(url4)

url5 = "https://raw.githubusercontent.com/pekpekkis/Data-Management-Group-Project/main/gini_coefficient.csv"
df_gini = pd.read_csv(url5)

url6 = "https://raw.githubusercontent.com/pekpekkis/Data-Management-Group-Project/main/continents.csv"
df_continents = pd.read_csv(url6)
df_continents = df_continents.drop('Year', axis='columns')

df_combined = df_co2.merge(df_green_energy, how='left', on=['Entity', 'Code', 'Year']).merge(df_population, how='left', on=['Entity', 'Year']).merge(df_GDP, how='left', on=['Entity', 'Code', 'Year']).merge(df_gini, how='left', on=['Entity', 'Code', 'Year']).merge(df_continents, how='left', on=['Entity', 'Code'])

df_combined['GDP per capita'] = df_combined['GDP (constant 2015 US$)'] / df_combined['Population']

def income_dummies(row):
    if row['GDP per capita'] < 1036:
        return 1, 0, 0, 0
    elif row['GDP per capita'] >= 1036 and row['GDP per capita'] < 4045:
        return 0, 1, 0, 0
    elif row['GDP per capita'] >= 4045 and row['GDP per capita'] < 12535:
        return 0, 0, 1, 0
    elif row['GDP per capita'] >= 12535:
        return 0, 0, 0, 1
    else:
        return 0, 0, 0, 0

df_combined[['Low income', 'Lower-middle income', 'Upper-middle income', 'High income']] = df_combined.apply(income_dummies, axis=1, result_type='expand')

def geo_dummies(row):
    if row['Continent'] == 'Africa':
        return 1, 0, 0, 0, 0, 0
    elif row['Continent'] == 'Asia':
        return 0, 1, 0, 0, 0, 0
    elif row['Continent'] == 'Europe':
        return 0, 0, 1, 0, 0, 0
    elif row['Continent'] == 'North America':
        return 0, 0, 0, 1, 0, 0
    elif row['Continent'] == 'South America':
        return 0, 0, 0, 0, 1, 0
    else:
        return 0, 0, 0, 0, 0, 1

df_combined[['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']] = df_combined.apply(geo_dummies, axis=1, result_type='expand')

df_combined = df_combined.rename(columns={'Entity': 'Country', 'Annual CO₂ emissions': 'CO2 emissions', 'Renewables (% equivalent primary energy)': 'Renewable energy %', 'GDP (constant 2015 US$)': 'GDP', 'Gini coefficient': 'Gini'})
df_combined.loc[df_combined['Country'] == 'United States', 'Country'] = 'United States of America'
df_combined.loc[df_combined['Country'] == "Cote d'Ivoire", 'Country'] = "Côte d'Ivoire"
df_combined['CO2 emissions'] = df_combined['CO2 emissions'] / 1000000
df_combined = df_combined[df_combined['Code'].notna()]
df_combined = df_combined.drop('Code', axis='columns')
df_combined = df_combined.drop(df_combined[df_combined['Country'] == 'World'].index, axis=0)

df_2020 = df_combined.loc[df_combined.Year == 2020]

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import folium
import contextily as ctx
import branca.colormap as cm
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import seaborn as sns

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.sort_values('name')
world.loc[world['name'] == 'Dem. Rep. Congo', 'name'] = 'Democratic Republic of Congo'
world.loc[world['name'] == 'Bosnia and Herz.', 'name'] = 'Bosnia and Herzegovina'
world.loc[world['name'] == 'Dominican Rep.', 'name'] = 'Dominican Republic'
world.loc[world['name'] == 'Central African Rep.', 'name'] = 'Central African Republic'
world.loc[world['name'] == 'Eq. Guinea', 'name'] = 'Equatorial Guinea'
world.loc[world['name'] == 'S. Sudan', 'name'] = 'South Sudan'
world.loc[world['name'] == 'Solomon Is.', 'name'] = 'Solomon Islands'
world.loc[world['name'] == 'Timor-Leste', 'name'] = 'Timor'


df_map_2020 = world.merge(df_2020, left_on='name', right_on='Country')

map_2020 = folium.Map(location=[0, 0], zoom_start=2)

bins_1 = [0, 1, 5, 20, 50, 100, 300, 1000, 4000, 10000, df_map_2020['CO2 emissions'].max()]
bins_2 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, df_map_2020['Renewable energy %'].max()]

df_clean = df_combined.dropna()
df_clean['CO2 emissions_log'] = np.log(df_clean['CO2 emissions'])

#Data visualization
def page1():
    st.title('Carbon Dioxide Emissions and Renewable Energy by Country')
    st.subheader('Examine the pollution level and green energy usage in a global comparison')
    st.caption('Source: Our World in Data')
    st.write('Hover above a country to see its carbon dioxide emissions level or renewable energy usage')
    st.sidebar.title('Variables')
    layer = st.sidebar.radio('Select a variable', ('CO2 emissions', 'Renewable energy %'))
    
    if layer == 'CO2 emissions':
        color = 'YlOrRd'
    else:
        color = 'YlGn'
    
    if layer == 'CO2 emissions':
        name = 'CO2 Emissions, million tonnes'
    else:
        name = 'Renewable energy, % of primary energy'
    
    if layer == 'CO2 emissions':
        scale = bins_1
    else:
        scale = bins_2
    
    def choose_map(variable):
        choropleth_map = folium.Choropleth(
            geo_data=df_map_2020,
            name=variable,
            data=df_map_2020,
            columns=['Country', variable],
            key_on='feature.properties.name',
            fill_color= color,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name= name,
            bins= scale,
            highlight=True,
            reset=True,
            overlay=True,
            control=True,
            show=True
        ).add_to(map_2020)
    
        choropleth_map.geojson.add_child(
            folium.features.GeoJsonTooltip(
                fields=['Country', variable],
                aliases=['Country:', variable + ':'],
                localize=True,
                sticky=True
                )
            )
    
    choose_map(layer)
    
    folium_static(map_2020)
    
    import matplotlib.pyplot as plt
    
    # Group the data by country and sum the emissions for each country
    total_co2_by_country = df_2020.groupby('Country')['CO2 emissions'].sum()
    
    # Sort the data in descending order and display the top 5 countries
    top_5_countries = total_co2_by_country.sort_values(ascending=False).head(5)
    
    renewable_by_country = df_2020.groupby('Country')['Renewable energy %'].sum()
    top_5_countries2 = renewable_by_country.sort_values(ascending=False).head(5)
    
    df_grouped = df_combined.groupby(['Continent', 'Year'])['CO2 emissions'].sum()
    
    avg_renewable_by_continent = df_2020.groupby('Continent')['Renewable energy %'].mean()
    sorted_values = avg_renewable_by_continent.sort_values(ascending=False)
    
    fig1, ax1 = plt.subplots(figsize=(8,6))
    fig2, ax2 = plt.subplots(figsize=(12,8))
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    def choose_graphs(variable):
        if variable == 'CO2 emissions':
            
            for country in df_grouped.index.levels[0]:
                ax2.plot(df_grouped.loc[country].index, df_grouped.loc[country].values, label=country)
                ax2.fill_between(df_grouped.loc[country].index, df_grouped.loc[country].values, 0, alpha=0.2)
            ax2.legend(loc='upper left')
            ax2.set_xlabel('Year', fontsize=14)
            ax2.set_ylabel('CO2 Emissions, Million Tonnes', fontsize=14)
            ax2.set_title('Carbon Dioxide Emissions Over Time by Continent', fontsize=18)
            st.pyplot(fig2)
            
            top_5_countries.plot.bar(ax=ax1)
            ax1.set_xlabel('Country')
            ax1.set_ylabel('Carbon Dioxide Emissions, Million Tonnes')
            ax1.set_title('Top 5 Most Emitting Countries in 2020')
        else:
            top_5_countries2.plot.bar(ax=ax1)
            ax1.set_xlabel('Country')
            ax1.set_ylabel('Renewable Energy, % of Primary Energy')
            ax1.set_title('Top 5 Countries with the Highest Renewable Energy Share in 2020')
            
            sorted_values.plot.bar(ax=ax3)
            ax3.set_xlabel('Continent')
            ax3.set_ylabel('Average Renewable Energy %')
            ax3.set_title('Average Renewable Energy % by Continent in 2020')
            st.pyplot(fig3)
    
    choose_graphs(layer)
    
    st.pyplot(fig1)
    
def page2():
    st.title("Log-linear -regression Explaining Carbon Dioxide Emissions with Renewable Energy and Other Controlling Variables")
    st.caption('Source: Our World in Data')
    regression_page = st.sidebar.radio('Select regression to examine', ('Linear regression for whole dataset', 'Linear regression with outliers removed'))
    
    def choose_regression_page(reg):
        global df_clean  # Use the global keyword to modify the global df_clean dataframe
        if reg == 'Linear regression for whole dataset':
            pass
            
        else:
            #Data modeling, regression without outliers
            threshold = 1

            z_scores = np.abs((df_clean['CO2 emissions_log'] - df_clean['CO2 emissions_log'].mean()) / df_clean['CO2 emissions_log'].std())

            outliers = df_clean[z_scores > threshold]

            df_clean = df_clean[z_scores <= threshold]
            

    choose_regression_page(regression_page)
    
    summary_stats = df_clean.describe()
    summary_stats = summary_stats.drop(['Year', 'CO2 emissions_log', 'Low income', 'Lower-middle income', 'Upper-middle income', 'High income', 'Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania'], axis=1)

    df_corr = df_clean.drop(['Year', 'CO2 emissions_log', 'GDP per capita', 'Low income', 'Lower-middle income', 'Upper-middle income', 'High income', 'Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania'], axis=1)
    corr_matrix = df_corr.corr()
    
    features = df_clean.drop(['Country', 'Continent', 'CO2 emissions', 'CO2 emissions_log', 'Year', 'GDP per capita', 'Europe', 'High income'], axis=1)
    y_full_2 = df_clean['CO2 emissions']
    y_full = df_clean['CO2 emissions_log']
    X_full = features

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full,test_size=0.2, random_state=1)

    numeric_columns=list(X_train.select_dtypes('float64').columns)
    dummy_columns=list(X_train.select_dtypes('int64').columns)
    categorical_columns = list(X_train.select_dtypes('int64').columns)

    categorical_columns_cleaned = [col for col in categorical_columns if col not in dummy_columns]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler,OneHotEncoder

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, dummy_columns)
        ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()

    X_train_num=X_train[numeric_columns + categorical_columns]
    
    lin_reg.fit(X_train_num, y_train)
    r_squared = lin_reg.score(X_train_num, y_train)
    
    lin_reg.fit(X_train_preprocessed, y_train)
    r_squared = lin_reg.score(X_train_preprocessed, y_train)

    lin_reg.fit(X_train_preprocessed, y_train)
    y_train_pred = lin_reg.predict(X_train_preprocessed)

    from sklearn.metrics import mean_squared_error
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    y_test_pred = lin_reg.predict(X_test_preprocessed)    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    st.subheader('Summary statistics')
    st.dataframe(summary_stats)
    
    st.subheader('Correlation matrix')
    st.dataframe(corr_matrix)
    st.caption('No multicollinearity')
    
    fig4, ax4 = plt.subplots(figsize=(11.7, 8.27))
    sns.histplot(y_full_2, bins=30, ax=ax4)
    ax4.set_xlabel("CO2 emissions", size=15)
    ax4.set_ylabel('count', size=15)
    ax4.set_title('Distribution of CO2 emissions', size=20)
    st.pyplot(plt)
    st.caption('Carbon dioxide emissions seem to be log-normally distributed')
    
    fig5, ax5 = plt.subplots(figsize=(11.7, 8.27))
    sns.histplot(y_full, bins=30, ax=ax5)
    ax5.set_xlabel("CO2 emissions_log", size=15)
    ax5.set_ylabel('count', size=15)
    ax5.set_title('Distribution of logarithmic CO2 emissions', size=20)
    st.pyplot(plt)
    st.caption('Carbon dioxide emissions are almost log-normally distributed')
    
    import statsmodels.api as sm

    X = df_clean[["Renewable energy %", "Population", "GDP", "Gini",
                  "Low income", "Lower-middle income", "Upper-middle income",
                  "Africa", "Asia", "North America", "South America", "Oceania"]]
    y = df_clean["CO2 emissions_log"]
    
    lin_reg.fit(X, y)
    
    X_with_constant = sm.add_constant(X)
    
    ols_model = sm.OLS(y, X_with_constant)
    
    results = ols_model.fit()
    coefficients = lin_reg.coef_
    
    p_values = results.pvalues[1:]
    
    coefficients_with_pvalues = pd.DataFrame({"Coefficient": coefficients, "P-value": p_values}, index=X.columns)
    
    coefficients_with_pvalues = coefficients_with_pvalues.style.format({
        "Coefficient": "{:.15f}",
        "P-value": "{:.15f}"
    })
    
    st.subheader('Coefficients')
    st.dataframe(coefficients_with_pvalues)
    st.caption('Baseline for income dummies: High income')
    st.caption('Baseline for geograpical dummies: Europe')
    st.caption('Statistically significant coefficients with 95 % confidence level:')
    
    significant_coefficients = []
    for index, row in coefficients_with_pvalues.data.iterrows():
        if row["P-value"] < 0.05:
            significant_coefficients.append(index)

    st.caption(', '.join(significant_coefficients))
    
    goodness_of_fit = pd.DataFrame({"Goodness of fit":[r_squared,
                                                     r_squared,
                                                     train_rmse,
                                                     test_rmse]}, index=["R-squared for training dataset:",
                                                                         "R-squared for training dataset & preprocessed features",
                                                                         "Trainingset RMSE",
                                                                        "Testset RMSE"])
    
    st.subheader('Model fit')                                                                    
    st.dataframe(goodness_of_fit)
    if regression_page == 'Linear regression for whole dataset':
        st.caption('Model has OK R-squared')
        st.caption('RMSE for the model is subpar, which indicates poor predictability')
        st.caption('Thus, see model without outliers')
    else:
        st.caption('Model R-squared improved compared to the model including the whole dataset')
        st.caption('RMSE also improved and now the model has also good predictability')
    
st.sidebar.title('Select page')
nav_selection = st.sidebar.radio("Go to", ["Page 1 - Data Visualization", "Page 2 - Data Modeling"])


if nav_selection == "Page 1 - Data Visualization":
    page1()
elif nav_selection == "Page 2 - Data Modeling":
    page2()