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

df_combined = df_co2.merge(df_green_energy, how='left', on=['Entity', 'Code', 'Year']).merge(df_population, how='left', on=['Entity', 'Year']).merge(df_GDP, how='left', on=['Entity', 'Code', 'Year']).merge(df_gini, how='left', on=['Entity', 'Code', 'Year'])

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

df_combined = df_combined.rename(columns={'Entity': 'Country', 'Annual CO₂ emissions': 'CO2 emissions', 'Renewables (% equivalent primary energy)': 'Renewable energy %', 'GDP (constant 2015 US$)': 'GDP', 'Gini coefficient': 'Gini'})
df_combined = df_combined[df_combined['Code'].notna()]
df_combined = df_combined.drop('Code', axis='columns')

df_2020 = df_combined.loc[df_combined.Year == 2020]
print(df_2020)
#In summary, this code merges four DataFrames, calculates GDP per capita, creates dummy variables for income categories, and cleans up the resulting DataFrame. It then selects only the rows with data from the year 2020 and prints them to the console.


