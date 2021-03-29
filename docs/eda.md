# 1. Explorative Data Analysis
This is our first stop. In this section I will dive deep into the features and get as much information from them as I can. The variable we're trying to predict is the price, so I will mostly try to explore relation of the features to the target. At the end I will also take a little bit of time to showcase the relationships between the features which is going to be very important to the feature engineering later.

### Our starting point
- Every row describes characteristics of a listing
- We don't know specific information about the listings but we have some description in their names
- Our target is the price (per night) of a listing

The first thing I think about when starting an EDA is what kind of information I'm actually dealing with and how to approach it. I'm frequently looking at airbnb listings and understand how they try to appeal to people and what to look for when it comes to getting higher or lower prices on similar type of place. Since I have some knowledge about the topic already, I will set a hypothesis that I will try to explore later. I think the way a listing is framed is as important to the price as its caterogical features, that's why I decided to at the end explore the descriptions of the listings, in particular - their names.

From personal experience I also know that the listings tend to be more expensive if they're close to public transport or in the center of the city. The placement of a property does make a huge difference, so I will take a bit more time exploring the neighbourhoods. Hopefully we will also discover some interesting information while going trough the features and see how we can use them to our advantage. 

### Basics information about the dataset
- Sample column:

| id | name | host_id | host_name | neighbourhood_group | neighbourhood | latitude | longitude | room_type | price | minimum_nights | number_of_reviews | last_review | reviews_per_month | calculated_host_listings_count | availability_365 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 71609 | Ensuite Room (Room 1 & 2) near EXPO | 367042 | Belinda | East Region | Tampines | 1.34541 | 103.95712 | Private room | 206 | 1 | 14 | 2019-08-11 | 0.15 | 9 | 353 |

- Shape
```python
# Removing the id columns
data = pd.read_csv('data/listings.csv')
data = data.drop(['id', 'host_id'], axis=1)

>> (7907, 14)
```

## 1.1 Learning about the target
