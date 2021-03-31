import pandas as pd
import plotly as plt
import plotly.figure_factory as ff
import plotly.express as pltx
from plotly.subplots import make_subplots
import matplotlib as mpl
import plotly.graph_objects as go
import numpy as np
from scipy.stats import yeojohnson
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# EDA
# - Each column describes a different feature of the listing
# - The goal is to predict the price of the listing based of the features

# Load the data and show information about it
data = pd.read_csv('data/listings.csv')
data = data.drop(['id', 'host_id'], axis=1)
print(data.head())
print(data.shape)

# Exploring the target (price)
price = [data.price]
fig_price = ff.create_distplot(price, ['Price'])
fig_price.update_layout(title_text='Original Price Data')
# >> fig_price.show()

# Exploring skewness
print('Skewness: ', data.price.skew())
print('Kurtosis: ', data.price.kurt())

# It's visible that the price does not follow normal distribution
# Therefore, I'm going to transform it using a log1p method
data['price'] = np.log1p(data.price)
fig_price_norm = ff.create_distplot([data.price], ['Price'])
fig_price_norm.update_layout(title_text='Normalized Price Data')
# >> fig_price_norm.show()

# Filling missing values
missing_values_count = data.isnull().sum()
print('Missing values:\n', missing_values_count)

# It looks like there's a lot of missing values in last_review and reviews_per_month columns and
# they are connected to each other.. If there's no reviews, there's no reviews per month or a last review
data = data.drop(['last_review'], axis=1)
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

# There's also two missing names, since it's a small number it's fair to just drop them
data = data[data['name'].notna()]

mvc_new = data.isnull().sum()
print('Missing values after cleaning:\n', mvc_new)

# Visualising features with the price as our y axis
def showcaseAll(data):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_df = data.select_dtypes(include=numerics)

    # Establish a canvas for the plots
    col_num = int(num_df.shape[1]/2)
    multiplot_fig = make_subplots(rows=2, cols=col_num,
                                  horizontal_spacing=0.05)
    row = 1
    col = 1
    for column in num_df:
        if column != 'price':
            # Select the trace of the figure
            new_fig = pltx.scatter(data, x=column, y='price', color='price')
            new_fig.update_traces(marker=dict(size=3))
            new_fig_trace = new_fig['data'][0]
            # Add new figure to the canvas
            multiplot_fig.add_trace(new_fig_trace,
                                    row=row, col=col)
            # Add titles
            multiplot_fig.update_xaxes(title_text=column, row=row, col=col)
            multiplot_fig.update_yaxes(title_text='price', row=row, col=col)

            col += 1
            if col > col_num:
                col = 1
                row = 2

    # Show the plots
    multiplot_fig.update_layout(title_text="Feature and price plots",
                                coloraxis=dict(colorscale='Twilight'))
    multiplot_fig.show()


# >> showcaseAll(data)

# Visualising some singular features since I'm interested to look into them
# I picked following, room_type and neighbourhood
test_fig = pltx.scatter(data, x='room_type', y='price', color='price')
test_fig.update_layout(coloraxis=dict(colorscale='Twilight'))
# >> test_fig.show()

# I also wanted to visualize how the price differences look on the map
map_fig = pltx.density_mapbox(data, lat='latitude', lon='longitude', z='price', radius=3, mapbox_style='carto-positron',
                              zoom=11)
map_fig.update_layout(coloraxis=dict(colorscale='Aggrnyl'))
# >> map_fig.show()

# I also wanted to see which neighbourhoods and room types are the most popular ones, change
# the x value accordingly to which one you want to display
neigh_count_fig = pltx.histogram(histfunc="count",  x=data['room_type'], marginal="violin")
# >> neigh_count_fig.show()

# Exploring skewness in the features and correcting it
feat_skew = data.skew()
fs_cat = []
fs_val = []
for index, value in feat_skew.items():
    fs_cat.append(index)
    fs_val.append(value)
feat_skew = [fs_cat, fs_val]

feat_skew_fig = pltx.bar(feat_skew, x=feat_skew[0], y=feat_skew[1], title="Skewness by feature",
                         color=feat_skew[1], color_continuous_scale='GnBu')
# >> feat_skew_fig.show()

# Correcting feature skewness using log1p, the goal was to transform the data
# using an easy method
for i in fs_cat:
    if i != 'price':
        data[i] = np.log1p(data[i])
        print(i)

print('Skewness after normalization:\n', data.skew())
# >> showcaseAll(data)

# Exploring the names and how they connect to the price
wordcloud = WordCloud(background_color='white',
                      width=1000,
                      height=700
                      ).generate(' '.join(data['name']))

wordcloud_fig = pltx.imshow(wordcloud, width=1000, height=700)
# >> wordcloud_fig.show()

# Exploring importance of words related to price
# I'm picking places with the same features but difference in names and comparing their
# average prices
def showDifference(data, keywords):
    # Save the results in a dictionary form
    result = {'values': keywords, 'avg_price_keyword': [], 'avg_price_no_keyword': []}
    for keyword in keywords:
        keyword_cols = data.loc[data['name'].str.contains(keyword)]

        most_frequent_room_type = keyword_cols['room_type'].value_counts()[:2].index.tolist()
        most_frequent_regions = keyword_cols['neighbourhood_group'].value_counts()[:2].index.tolist()

        # Compare the columns to these with similar features but no keyword
        no_keyword_cols = data.loc[~(data['name'].str.contains(keyword)) &
                                   (data['room_type'].isin(most_frequent_room_type)) &
                                   (data['neighbourhood_group'].isin(most_frequent_regions))]

        avg_price_keyword = round(keyword_cols['price'].mean(), 2)
        avg_price_no_keyword = round(no_keyword_cols['price'].mean(), 2)

        # Append to the dictionary
        result['avg_price_keyword'].append(avg_price_keyword)
        result['avg_price_no_keyword'].append(avg_price_no_keyword)

    return pd.DataFrame.from_dict(result)


keywords = ['cozy|cosy', 'mrt', 'central', 'modern', 'balcony', '2br|3br', 'chinatown', 'spacious', 'loft', 'luxury',
            'penthouse', 'studio', 'park', 'dorm']
keyword_price_df = showDifference(data, keywords)

keywords_fig = go.Figure()
keywords_fig.add_trace(go.Bar(
    x=keyword_price_df['values'],
    y=keyword_price_df['avg_price_keyword'],
    name='With keyword',
    marker_color='Indigo'
))
keywords_fig.add_trace(go.Bar(
    x=keyword_price_df['values'],
    y=keyword_price_df['avg_price_no_keyword'],
    name='Without keyword',
    marker_color='LightSteelBlue'
))
keywords_fig.update_layout(title='Difference in price between names with and without the keyword')
# >> keywords_fig.show()

# Show the increase/decrease in price in %
keyword_dict = keyword_price_df.to_dict()
change_in_percent = {}
amount_of_val = len(keyword_dict['values'])

for i in range(amount_of_val):
    with_keyword = keyword_dict['avg_price_keyword'][i]
    without_keyword = keyword_dict['avg_price_no_keyword'][i]
    percent_change = 100*((with_keyword - without_keyword)/np.abs(without_keyword))
    change_in_percent[keyword_dict['values'][i]] = round(percent_change, 2)

for key, value in change_in_percent.items():
    print(key + ': ' + str(value) + '%')

# Encoding caterogical features
# First I remove the name value since it will pollute the one hot encoding and does
# not contribute that much to the end result
data = data.drop(['name', 'host_name'], axis=1)
data = y = pd.get_dummies(data)
print('Data after encoding: ', data.shape)

# Exploring how features are correlated to each other and the price of a rental
corr = data.corr()
corr_fig = pltx.imshow(corr, color_continuous_scale='GnBu')
corr_fig.update_layout(title='Correlation between features')
# >> corr_fig.show()
