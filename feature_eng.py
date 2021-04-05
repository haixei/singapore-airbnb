from eda_data_transformations import data
import plotly.express as pltx
import plotly.graph_objects as go
import pandas as pd

# Making sure that the data got imported correctly, and importing original data to use the names
print(data.head())
data_og = pd.read_csv('data/listings.csv')

# Saving version of the data without keyword kolumns to compare the model results later
data_without_keywords = data

# Splitting data and creating new features
data['luxury'] = 0
data['dorm'] = 0
data['spacious'] = 0

for i in data.index:
    if ('penthouse' in data_og['name'][i]) or ('luxury' in data_og['name'][i]):
        data['luxury'][i] = 1
    elif 'dorm' in data_og['name'][i]:
        data['dorm'][i] = 1
    elif ('2br' in data_og['name'][i]) or ('3br' in data_og['name'][i] or ('spacious' in data_og['name'][i])):
        data['spacious'][i] = 1

print('Amount of filled rows:\n', data.spacious.value_counts())

# Removing outliers
data.drop(data[(data['number_of_reviews'] > 2) & (data['price'] > 7)].index, inplace=True)
data.drop(data[(data['minimum_nights'] > 2) & (data['minimum_nights'] < 5) & (data['price'] > 7)].index,
          inplace=True)
data.drop(data[(data['availability_365'] > 1) & (data['availability_365'] < 4) & (data['price'] > 6)].index,
          inplace=True)

# Removing deeply correlated features
data.drop(['reviews_per_month'], axis=1)
data.reset_index(drop=True, inplace=True)

print('Data size: ', data.shape)
