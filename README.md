![](https://i.imgur.com/UJeNsLq.png)
# Singapore's Airbnb Data Analysis
Singapore, a sovereign island city-state in South East Asia, brings attention for many rasons. It's a popular travel destination as well as the center of a lot of businesses. Naturally, it brings a lot of tourism which is a major contributor to its economy and people who need to stay for work related reasons. Airbnb in such place has a lot of popularity. The reasonable prices it offers attract a lot of customers and the variety of options is only a plus. From shared rooms to entire mansions, hosts created a catalog of offers that is impossible to ignore. In this data analysis I will go trough the information collected from the company and try to learn more about the market there.

## Legend
1. [Exploratory data analysis](docs/eda.md)
	 - [1.1 Learning about the target](docs/eda.md#11-learning-about-the-target)
	 - [1.2 Visualising features](docs/eda.md#12-visualising-features)
	 - [1.3 Skewness and normalization of the features](docs/eda.md#13-skewness-and-normalization-of-the-features)
	 - [1.4 Diving deeper into the names](docs/eda.md#14-diving-deeper-into-the-names)
	 - [1.5 Encoding caterogical features](docs/eda.md#15-encoding-caterogical-features)
	 - [1.6 Correlation heatmap](docs/eda.md#16-correlation-heatmap)
2. [Feature engineering](docs/feat_eng.md)
   - [2.1 Adding new features](docs/feat_eng.md#21-adding-new-features)
   - [2.2 Removing outliers](docs/feat_eng.md#22-removing-the-outliers)
   - [2.3 Removing deeply correlated features](docs/feat_eng.md#23-removing-deeply-correlated-features)
3. [About the model](docs/about_model.md)
   - [3.1 Explaining the approach](docs/about_model.md#31-explaining-the-approach)
	 - [3.1.1 Cross validation](docs/about_model.md#311-setting-up-cross-validation)
	 - [3.1.2 Adding models](docs/about_model.md#312-adding-models)
   - [3.2 Stacking & blending](docs/about_model.md#32-stacking-and-blending)
4. [Result analysis](docs/result_analysis.md)
   - [4.1 Individual model scores](docs/result_analysis.md#41-individual-model-scores)
   - [4.2 Blended models score](docs/result_analysis.md#42-nlended-model-score)
   - [4.3 Comparison of two data sets](docs/result_analysis.md#43-comparison-of-two-data-sets)

## The Goal
- [x] Exploring the relationships between features
- [x] Learning about the connection between language used and prices
- [x] Predicting listing prices

## The Data
You can find the data under this link. The csv file is made of 16 following columns:
```
id
name
host_id
host_name
neighbourhood_group
neighbourhood
latitude
longitude
room_type
price
minimum_nights
number_of_reviews
last_review
reviews_per_month
calculated_host_listings_count
availability_365
```
