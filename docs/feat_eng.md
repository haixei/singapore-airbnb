# 2. Feature Engineering
Fairly short section in this case, I'm going to go trough the features, add some new ones and clean these who do not contribute to the result a lot. Keep in mind that I did some minor engineering in the EDA to display the data properly already.

2.1 Creating new features
Since we gathered some interesting information from the names, we can use it to create some new features that might help us achieve more accurate result. The original data set has only 17 columns from which the placement of a listing (neighbourhood, lat/long) and the room type have the most influence on our target. It makes a lot of sense and it's probably the most of where the price is coming from but in some specific cases we're dealing with listings where the way they're framed or their specific characteristics could greatly influence the price. Two room bedroom apartment with a luxury finish will probably cost more than one with a very basic finish.

Before doing anythig I will save the data without keywords so we can compare the results later and load in original data to retreive the names column.
```python
# Making sure that the data got imported correctly, and importing original data to use the names
print(data.head())
data_og = pd.read_csv('data/listings.csv')

# Saving version of the data without keyword kolumns to compare the model results later
data_without_keywords = data
```

Now I'm going to create columns for our new features and fill them with data. As in the case of one hot encoding, I'm simply creating them filled with 0 if a keyword is not included in the name, and 1's if it is.

```python
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
```

One problem we might face is that the amount of listings that are influenced by my observation is so small, it won't help the model a lot. Let's look at the amount of filled rows in the "spacious" keyword.
```
>> Amount of filled rows:
   0    7839
   1      66
```

I'm worried it might not have much influence and even steer the model into a wrong direction. On the other side if we face a listing that is influenced by it, that information could be very useful for our predictions. I will test that hypothesis later when I train the model.

# 2.2 Removing the outliers
Removing the data that could potentially distrupt our model is important but in this case if we look back at our EDA, we can see that there's not that many outliers. Most features follow some kind of trajectory. There's only a few that have a few random points worth cleaning up.

```python
# Removing outliers
data.drop(data[(data['number_of_reviews'] > 2) & (data['price'] > 7)].index, inplace=True)
data.drop(data[(data['minimum_nights'] > 2) & (data['minimum_nights'] < 5) & (data['price'] > 7)].index,
          inplace=True)
data.drop(data[(data['availability_365'] > 1) & (data['availability_365'] < 4) & (data['price'] > 6)].index,
          inplace=True)
```

# 2.3 Removing deeply correlated features
Some features could have a big correlation to the point that it might as well look like a column repeated two times. Looking back at the correlation heatmap we can tell that it's the case for reviews and the location of a listing. Number of reviews and reviews per month tell the same story in different ways, so we can freely remove the latter. When it comes to the location, neighbourhood group has a big relation to the longitude so there's no use in keeping it either but this one was dropped before already since it had a lot of values that could easily polute the data set after encoding with features that are not very meaningful to us because of the previously mentioned relation to another characteristic.

```python
# Removing deeply correlated features
data.drop(['reviews_per_month', 'neighbourhood_group'], axis=1)
data.reset_index(drop=True, inplace=True)
```

Since all the features got checked and look right, we can move into [preparing the model for training.](about_model.md)
