# Singapore's Airbnb Data Analysis
Singapore, a sovereign island city-state in South East Asia, brings attention for many rasons. It's a popular travel destination as well as the center of a lot of businesses. Naturally, it brings a lot of tourism which is a major contributor to its economy and people who need to stay for work related reasons. Airbnb in such place has a lot of popularity. The reasonable prices it offers attract a lot of customers and the variety of options is only a plus. From shared rooms to entire mansions, hosts created a catalog of offers that is impossible to ignore. In this data analysis I will go trough the information collected from the company and try to learn more about the market there.

## Legend
1. [Exploratory data analysis](docs/EDA.md)
	 - [1.1 Learning about the target](docs/EDA.md#section)
	 - [1.2 Visualising features](docs/EDA.md#section)
	 - [1.3 Handling missing values](docs/EDA.md#section)
	 - [1.4 Exploring singular features](docs/EDA.md#section)
	 - [1.5 Transforming features](docs/EDA.md#section)
	 	 - [Skewness](docs/EDA.md##section)
		 - [Normalization](docs/EDA.md##section)
		 - [Encoding](docs/EDA.md##section)
	 - [1.6 Correlation heatmap](docs/EDA.md#section)
	 - [1.7 Exploring neighbourhoods](docs/EDA.md#section)
	 - [1.8 Diving deep into the names](docs/EDA.md#section)
2. [Feature engineering](docs/feat_eng.md)
   - [1.1 Creating new features](docs/feat_eng.md#section)
	 - [1.2 Removing outliers](docs/feat_eng.md#section)
	 - [1.3 Removing deeply correlated features](docs/feat_eng.md#section)
3. [Creating the model](docs/about_model.md)
   - [1.1 Explaining the approach](docs/about_model.md#section)
	 - [1.2 Cross validation](docs/about_model.md#section)
	 - [1.3 Adding models](docs/about_model.md#section)
	 - [1.4 Stacking & blending](docs/about_model.md#section)
4. [Training the model and result analysis](docs/result_analysis.md)

## The Goal
- [x] Exploring the relationships between features
- [ ] Learning about the connection between language used and prices
- [ ] Predicting listing prices

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
