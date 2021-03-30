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
Price per night is the variable we're trying to predict so first I'm going to visualize its distribution and check the skewness.
![Price distribution](/plots/price-dist-01.png)
```
Skewness:  19.09278291149684
Kurtosis:  464.432795706112
```
We can clearly see that it does not follow a normal distribution so I'm going to use a log function to fix that. As a side note I will add that I did try other methods like Yeo Johnson but they didn't provide a more satisfying output. Sometimes simple is better.
![Price distribution normalized](/plots/price-dist-02.png)

## 1.2 Visualising features
Now when we took care of our target it's time to explore the features. My first objective is visualizing the numerical features against our target in hopes to learn more about them. Before I do so I will also check for missing values and try to handle them.
```
Missing values:
name                                 2
host_name                            0
neighbourhood_group                  0
neighbourhood                        0
latitude                             0
longitude                            0
room_type                            0
price                                0
minimum_nights                       0
number_of_reviews                    0
last_review                       2758
reviews_per_month                 2758
calculated_host_listings_count       0
availability_365                     0

```
It looks like there's a lot of missing values in last_review and reviews_per_month columns and they are connected to each other.. Which actually makes sense. Let's look at it that way, if there's the number of reviews for a listing is equal for 0 then there's simply no reviews per month. We also won't find the last review because it simply doesn't exist. In this case, I will fill in missing values with a 0.

There's also two missing names, since it's a very small number compared to the scale of the data set, I'm going to drop them. After the cleaning It's finally time to make some graphs. I'm using the plotly library, it's especially useful for looking at the graphs in the browser since it offers a lot of cool features.

![Features against price](/plots/price-num-features-01.png)
These graphs displayed some interesting information. We can clearly notice that some of the features are prone to having outliers, and we should note that for later when we're going to do engineering on them. The availability feature looks very intruiging, it has some spikes in price in certain points so it might be useful for our predictions. When it comes to the number of reviews, many hosts have between 0 - 50 and the more reviews the less likely we are to see expensive listings. The number of reviews per month and all reviews seem to have a lot in common, it's also something we should keep in mind, since we might just need one of them for our final model.

I also want to explore some caterogical features on their own, I picked two: room type and neighbourhood. Both could have an impact on our target.
![Room type against price](/plots/ftest-room-type.png)
![Neighbourhood type against price](/plots/ftest-neighbourhood.png)
Just like I predicted, there is in fact some correlation between them. Shared rooms are obviously cheaper and hardly ever go high in price, the entire home/apartment listings are the opposite. There's also a few neighbourhoods with visibly higher and lower prices like Southern Islands and Buklt Panjang. Our second graph is quite interesting and since we know exactly where the listings are on the map we can visualize it easily with a heat map.
![Price map](/plots/price-map.png)

Another thing I'm curious about is how popular are certain room types and neighbourhoods.
![Price map](/plots/room-type-histogram.png)
![Price map](/plots/neighbourhood-histogram.png)
It seems like people who travel to Singapore are not very likely to stay in shared rooms, otherwise they would have more popularity. That makes me wonder if there's a better alternative around for people who want to stay somewhere for cheap or if it's because people who travel there stay for a vacation so they need a better standard (what would make a lot of sense since tourism is a very important factor in Singapore's economy). It's not related to our model but still an interesting thought.

## 1.3 Transforming features
We already took care of our target but since we learned a lot about our features in the previous section, it's time to transform them as well. I will start with exploring the skewness and normalizing the data.
