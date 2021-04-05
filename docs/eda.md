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

## 1.3 Skewness and normalization of the features
We already took care of our target but since we learned a lot about our features in the previous section, it's time to transform them as well. I will start with exploring the skewness and normalizing the data.
![Skewness](/plots/skewness-01.png)
We can see that some features have skewness way over the line of a correct one, so I'm going to apply the log function over them. Normalizing data will let us look at it in a different, better way. You can see the result on the graph below.
```
Skewness after normalization:
latitude                          1.661790
longitude                        -0.740894
price                             0.309884
minimum_nights                    1.209086
number_of_reviews                 0.799579
reviews_per_month                 1.393804
calculated_host_listings_count    0.372717
availability_365                 -1.313386
```
![Features against price (normalized)](/plots/price-num-features-02.png)

## 1.4 Diving deeper into the names
This part of the exploration is I'd say quite experimental since I'm not sure what kind of results it could bring. What I wish to see is what kind of information we can extract from the names since they might hide something interesting. My bet is on the way renters frame the listing, the size of a place and additional features like being close to public transport or the airport. I'm going to start with creating a word cloud to see what kind of keywords are used most often.
![Listing name word map](/plots/name-word-map.png)
We can see from the image that the most used words are related to the size of a place, where it's places and the feel it should give off. Knowing that I want to pick some of the more popular words and do a small experiment. I will take rentals with the keyword and ones of the same kind but without it, get the average prices and compare. I'm really curious to see by how much they differ.
![Price difference](/plots/keyword-price-diff.png)
There really seem to be a notable difference in average price between some of them, especially "penthouse", "luxury, "2br|3br| and "dorm". This information might be very useful for our model but I'm going to be careful with that. Some of these might be related to already existing features in our data set, I'm going to look more into this in the next step which is feature engineering.

## 1.4 Encoding caterogical features
I will be using one hot encoding since it's probably the best way for these features. Label encoding simply doesn't fit them and could lead the model to thinking that labels of a higher number are related to something, when they are just simply different things. With too many unique values, that kind of approach could be a big issue so I will be dropping the name and host name features. The name can be useful but we already saw how we can extract important information from it, so we'll add some columns representing that later. Host name is just their name and won't be useful. There might be some relation between price and the host but that can be represented by their id.
```
Data after encoding:  (7905, 59)
```
## 1.5 Correlation heatmap
The last step in this EDA will be displaying the correlation between features using a heatmap.
![Heatmap graph](/plots/corr-01.png)
There's not too many visible relationships, which is pretty good. I wish we had more strong relationships related to price, but we will do fine without them too. Most of these correlations makes sense, like room type being related to a neightbourhood group, so at least we don't have to worry about some weird links between the features that we'd have to explore more in depth. The only thing that I need to keep in mind is, once again, the relationship between review features that are closely linked.

## Next step
Since we analyzed the data set in depth already, it's time to move to working on our data for a little bit. Feel free to check the next step which is [feature engineering.](docs/feat-eng.md)
