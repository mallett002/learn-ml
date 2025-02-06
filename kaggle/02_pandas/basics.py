import pandas as pd

# data_frame = pd.read_csv('./data.csv', index_col=0)
random_df_data = pd.read_csv('./data.csv')

# print(random_df_data.shape) # --> (record_count, column_count)
# print(random_df_data.head())




# Making a data frame from sratch with index labels:
fruit_sales = pd.DataFrame({
    'Apples': [35, 41, 19, 10],
    'Bananas': [21, 34, 31, 3],
    'Kiwis': [92, 48, 19, 9],
}, index=(['2017 Sales', '2018 Sales', '2019 Sales', '2020 Sales']))
# Makes this vv
#                 Apples	Bananas     Kiwis   
# 2017 Sales	  35	    21          92
# 2018 Sales	  41	    34          48
# 2019 Sales	  19        31          19
# 2020 Sales	  10         3           9





# Make a series with index labels and a name:
ingredients = pd.Series(
    ['4 cups', '1 cup', '2 large', '1 can'],
     index=['Flour', 'Milk', 'Eggs', 'Spam'],
     name='Dinner')
# Makes this vv

# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object




# Write a csv to file from a data frame:
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.to_csv('cows_and_goats.csv') # vv
# ,Cows,Goats
# Year 1,12,22
# Year 2,20,19


# Selecting data:
# Select by column name
fruit_sales['Apples']

# iloc[start:stop] start(inclusive), stop(exclusive)
# Select 1st row of data
fruit_sales.iloc[0]

# Select data from 1st column
fruit_sales.iloc[:, 0]

# Select just 2nd and 3rd rows from 1st column
fruit_sales.iloc[1:3, 0]

# Select 1st, 2nd, 3rd rows of first col (passing array)
fruit_sales.iloc[[0, 1, 2], 0]

# Selecting from the end
fruit_sales.iloc[-3, :] # Select 3rd to last row, all columns
fruit_sales.iloc[-3:] # selects multiple rows, end to 3rd to last, all columns


# Test questions
# Select the value of apples sold in 2019
fruit_sales.iloc[2, 0]

# Select the entire row of data for 2018 sales (all fruits in 2018)
fruit_sales.iloc[1, :]

# Select all the values of kiwis sold (all years)
fruit_sales.iloc[:, 2]

# Select the last two rows of data (for 2019 and 2020 sales)
fruit_sales.iloc[-2:]

# Select all the banana sales except for 2020 (2nd col, all but last row)
fruit_sales.iloc[0:3, 1]
# print(fruit_sales.iloc[:-1, 1]) # what AI says (all but last row, 2nd col)




# Label based selection with loc[row(start:stop), col(start:stop)]  start(inclusive):stop(inclusive)
# Get first entry in Apples:
fruit_sales.loc['2017 Sales', 'Apples']

# Get all data for apples & kiwis
fruit_sales.loc[:, ['Apples', 'Kiwis']]

# Can set the index on the df
fruit_sales.set_index('Apples')






# Conditional selection:
reviews = pd.read_csv('./wine_reviews_sample.csv')
# reviews.country == 'Italy' # produces series of True False bools

# get just the reviews that are from Italy
reviews.loc[reviews.country == 'Italy'] 

# get Italian reviews over 90
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)] 
reviews.loc[reviews.country.isin(['Italy', 'France'])]

# get Italian reviews or those rated over 90
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]

# get price that is not null
reviews.loc[reviews.price.notnull()]





# Assigning data
reviews['critic'] = 'Billy'
reviews.critic

#range(start, stop, step)
list(range(5, 0, -1)) # start at 5, stop at 0 exclusive, go back 1 each iteration

# reviews['index_backwards'] = range(len(reviews), 0, -1)
# reviews['index_backwards']





##### Summary Functions ####
reviews.describe() # see high level aggregate info
reviews.taster_name.describe() # see aggregate info on taster_name (string value)
reviews.points.mean() # see average of points allotted
reviews.taster_name.unique() # see all the taster names
reviews.taster_name.value_counts() # See tasters and how often they occur in data


# map higher level function
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)


##########################################
#### .apply() ####
##########################################
# apply - Similar to map, but transforms whole dataframe by calling custom dataframe on each row
# axis:
    # 0 or "index" --> (vertical) Doing something to each column (default)
    # 1 or "columns" (horizontal) Doing something to each row

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns') 



# How axis works in apply: ---------------------------------------------------------------
#  .apply notes
treats = pd.DataFrame({
    'cookies_sold': [1, 2, 3],
    'donuts_sold': [4, 5, 6],
    'muffins_sold': [7, 8, 9],
})

# print(treats)
# print('\n\n')

# 0 --> columns (vertical) Doing something to each column (default)
def sum_cols(df):
    return df.sum()

treats_sold = treats.apply(lambda df: df.sum()) # --> sums the data for each column (0 or "index")

# print("Treats sold by treat:")
# print(treats_sold)
# print('\n\n')

# 1 --> index  (horizontal) Doing something to each row
def sum_rows(row):
    return row.sum()

treats_sold_by_row = treats.apply(sum_rows, axis=1) # --> sums the data for each row (1 or "columns")

# print("Treats sold by row:")
# print(treats_sold_by_row)
# print('\n\n')


# default 0 or index -- apply function to each column
# 1 or columns -- apply function to each row

# Other example of .apply (to remean points)
example_reviews = pd.DataFrame({
    'points': [95, 80, 85],
    'other_column': ['A', 'B', 'C']
})

 # Calculate the mean of 'points'
review_points_mean = example_reviews['points'].mean()
# print(review_points_mean)

 # Define the function to remean points
def remean_points(row):
   row['points'] = row['points'] - review_points_mean
   return row

# Apply the function to each row
example_reviews = example_reviews.apply(remean_points, axis=1) # columns, so do something to each row

# print(example_reviews['points'])

# ----------------------------------------------------------------------------------------


# faster way to remean points column
reviews.points - review_points_mean


# combine country and region in dataset
reviews.country + '-' + reviews.region_1


# Get wine with highest points-to-price ratio
bargain_idx = (reviews.points / reviews.price).idxmax() # gets the row label of the highest value
bargain_wine = reviews.loc[bargain_idx, 'title']


# create "descriptor_counts" - how many times each of the word fruity or tropical appears in description col
n_trop = reviews.description.map(lambda d: 'tropical' in d).sum()
n_fruity = reviews.description.map(lambda d: 'fruity' in d).sum()

descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])


# Create a series "star_ratings" with the number of stars corresponding to each review in the dataset.
def apply_stars(row):
    if row.country == 'Canada' or row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2

    return 1

# The answer:
reviews_with_stars = reviews.apply(apply_stars, axis='columns')

# Can add stars on the reviews DF like this now:
reviews['stars'] = reviews_with_stars






##### Grouping and Sorting ####
# When group by, index essentially becomes what you grouped by

reviews.points.value_counts()
# does ths same:
reviews.groupby('points').points.count()
# ^^ Puts in groups by points, get's the points col and counts how many rows (# times appeared)

reviews.groupby('points').price.min()
# groups points together points --> price
# 87 --> [15.99, 39.99, 29.00] 
# 91 --> [21.99, 93.99]
# 59 --> ...
# Then, selects the min for each one


# Select name of first wine reviewed for each winery in dataset
first_wines = reviews.groupby('winery').apply(lambda df: df.title.iloc[0], include_groups=False)
# ^^ include_groups=False removes 'winery from the df

# Pick best wine by country and province
reviews.loc[0] # --> first row
reviews.points.idxmax() # --> gets the index of the row with most points
best_wines_by_country_prov = reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()], include_groups=False)

# agg: run several summary functions on DF simultaneously
statistical_summary = reviews.groupby('country').price.agg([len, 'min', 'max'])


# Multi-indexes
# has 2 row labels:
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
# convert back into regular index:
# countries_reviewed.reset_index()

# Sorting
# countries_reviewed = countries_reviewed.reset_index()
countries_reviewed = countries_reviewed.sort_values(by='len') # sort results asc
countries_reviewed = countries_reviewed.sort_values(by='len', ascending=False) # sort results desc 
countries_reviewed.sort_index() # sort by index
countries_reviewed.sort_values(by=['country', 'len']) # sort by more than 1 col at a time


# practice for grouping and sorting
# most common wine reviewers - Groups by twitter handle and then counts how many in each group
reviews.groupby('taster_twitter_handle').size()

# Best wine for $
# Create series. index is wine prices. val is max points for that price. sort price asc.
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()

# min and max prices for each variety of wine
# Create a DF whose index is the variety, vals are min and max of it
price_extremes = reviews.groupby('variety').price.agg(['min', 'max'])


# Most expensive wine varieties
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

# Series. index is reviewers. values is avg score by that reviewer
reviewer_mean_ratings = reviews.groupby('taster_name')['points'].mean()

# What combination of countries and varieties are most common?
# Series index is MultiIndex of {country, variety} pairs.
# Sort vales in Series in desc based on wine count
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False) # sort results desc 
# reviews.groupby(['country', 'variety']) --> Group them into pairs of country,variety
# Get the amount of occurances of this combination --> .size()
# sort them in descending order --> .sort_values(ascending=False)










##### Data Types and Missing Values #####
# See types:
reviews.price.dtype # --> on column
reviews.dtypes # --> on all df columns

# Change dtypes
reviews.points.astype('float32')
reviews.price.dtype

# Turn int into string
point_strings = reviews.points.astype('str')

# transform whole df
memory_effecient_reviews = reviews.astype({'points': 'int32', 'stars': 'int32', 'price': 'float32'})


# Missing data
pd.isnull(reviews.country) # --> Series: indexes where reviews.country is null
reviews[pd.isnull(reviews.country)] # --> gets new df with rows that have missing country

# find amount of rows with null country
no_country = len(reviews[reviews.country.isnull()])
# or this way (like it more):
no_country = pd.isnull(reviews.country).sum()


# fill in missing values
reviews.country = reviews.country.fillna('Unknown')

# replace values
reviews.taster_twitter_handle = reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")


# What are most common wine-producing regions?
# Series counting # of times each value occurs in region_1 (field often missing data)
# replace missing vals with "Unknown"
# sort desc
reviews.region_1 = reviews.region_1.fillna('Unknown')
reviews_per_region = reviews.region_1.value_counts().sort_values(ascending=False)

# or just do it all in 1 line:
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)
# or this
reviews_per_region = reviews.fillna({'region_1': 'Unknown'}).groupby('region_1')['region_1'].count().sort_values(ascending=False)









######## Renaming & Combining #######

# change index/column names - rename()

# rename points column to score:
reviews_with_score = reviews.rename(columns={'points': 'score'}) 

# change index name of 1st & 2nd entry
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
# set_index usually more convenient than rename of index 


reviews.rename_axis("wines", axis='rows') # adds title for row indexes
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

# Combining
# concat, join and merge functions

# concat - smushes dataframes/series together when they have the same fields (columns)
canadian_youtube = pd.read_csv("./CAvideos.csv")
british_youtube = pd.read_csv("./GBvideos.csv")

combined_videos = pd.concat([canadian_youtube, british_youtube])

# join
left = canadian_youtube.set_index(['title', 'trending_date'])
right = canadian_youtube.set_index(['title', 'trending_date'])


# Joins them by putting the data in the same row if they have the same title and trending_date
joined = left.join(right, lsuffix='_CAN', rsuffix='_UK')









########################################################################################################
####### PRACTICE ########
########################################################################################################

# 1. Select name of first wine reviewed for each winery in dataset
reviews.groupby('winery').title.first()
reviews.groupby('winery').apply(lambda df: df.title.iloc[0], include_groups=False)



# 2. Pick best wine by country and province
# Most points

reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df['points'].idxmax()], include_groups=False).points


# 3. Pick best wine by country and province for the value
# highest ratio of points/price:

# Step 1: Add the ratio column to the original DataFrame
reviews['ratio'] = reviews['points'] / reviews['price']

# Step 2: Use the ratio column within the groupby operation and calculate the index of the max ratio

best_vals = reviews.loc[reviews.groupby(['country', 'province'])['ratio'].idxmax().sort_values(ascending=False)]
# print(reviews.loc[reviews.title == 'Nicosia 2013 Vulkà Bianco (Etna)'])


# 4. Most common wine reviewers
reviews.groupby('taster_twitter_handle').size()

# 5. Get best rated wines per price
best_rating_per_price = reviews.groupby('price')['points'].max().sort_values()


# 6. min and max prices for each variety of wine
min_max_prices = reviews.groupby('variety')['price'].agg(['min', 'max']).sort_values(by=['min', 'max'])

# 7. Most expensive wine varieties
most_exp_varieties = reviews.groupby('variety')['price'].max().sort_values(ascending=False)

# 8. Average scores by reviewers
reviews.groupby('taster_twitter_handle')['points'].mean()

# 9. What combination of countries and varieties ar emost common?
reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)


### A few more practice prompts ###

# 10. Find the winery with the most reviews in each country
most_reviewed_winery_per_country = reviews.groupby(['country', 'winery']).size().sort_values(ascending=False)

# 11. Get the average price per variety and province
reviews.groupby(['variety', 'province']).price.agg('mean').sort_values(ascending=False)

# 12. Group by 'taster_twitter_handle', calculate the mean points, and get the top 5 tasters
reviews.groupby('taster_twitter_handle')['points'].mean().sort_values(ascending=False).head(5)

# 13. Group by 'winery', find the wine with the highest price for each winery
# print('mine:')
highest_price_per_winery = reviews.groupby('winery')['price'].max().sort_values(ascending=False)
# print(highest_price_per_winery)
ais_highest_price_per_winery = reviews.loc[reviews.groupby('winery')['price'].idxmax()].sort_values(by=['price'], ascending=False)

# 14. Group by 'country' 'variety', count the total number of reviews
total_reviews_per_country_variety = reviews.groupby(['country', 'variety']).size().reset_index(name='the total')




##### A bit harder #####



# 15. Find the Average Score of Wines for Each Taster, Ignoring Any Tasters Who Have Fewer Than 2 Reviews:
# Group by 'taster_twitter_handle' and count the number of reviews for each taster
taster_review_counts = reviews.groupby('taster_twitter_handle').size()

# Filter out tasters with fewer than 2 reviews
frequent_tasters = taster_review_counts[taster_review_counts >= 2].index # get list of taster_twitter_handles with more than 2 reviews

# Calculate the average score for each taster with at least 2 reviews
# build df from ones that have the frequent tasters as their taster_twitter_handle
# groupby taster_twitter_handle
# get avg points for those
avg_scores_frequent_tasters = (
    reviews[
        reviews['taster_twitter_handle'].isin(frequent_tasters)
    ]
    .groupby('taster_twitter_handle')['points']
    .mean()
)

# 16. # Identify the Winery with the Highest Average Points per Country:
# # Group by 'country' and 'winery', calculate the mean points, and find the winery with the highest average points in each country
highest_avg_points_winery_per_country = ( 
    reviews.groupby(['country', 'winery'])
        .points
        .mean()  # Calculate the average points for each country-winery combination
        .groupby(level=0)  # Re-group by just the country (the mean calculation created a multi-index of (country, winery))
        .idxmax()  # Get the indexes (country, winery) with the highest average points within each country
)


# 17. # Find the Percentage of Wines Scoring 90+ Points in Each Country:
# # Calculate the total number of wines and the number of wines scoring 90+ points for each country
# total_wines_per_country = reviews.groupby('country').size()
# high_scoring_wines_per_country = reviews[reviews['points'] >= 90].groupby('country').size()
# # Calculate the percentage of high-scoring wines
# percentage_high_scoring_wines = (high_scoring_wines_per_country / total_wines_per_country) * 100


# 18. # Determine the Wine Variety with the Most Consistent Ratings (Lowest Standard Deviation of Points) in Each Country:
# # Group by 'country' and 'variety', calculate the standard deviation of points, and find the variety with the lowest standard deviation in each country
# most_consistent_variety_per_country = reviews.groupby(['country', 'variety'])['points'].std().groupby(level=0).idxmin()


# 19. # Calculate the Correlation Between Price and Points for Each Country:
# # Group by 'country' and calculate the correlation between price and points
# # correlation_price_points_per_country = reviews.groupby('country').apply(lambda df: df['price'].corr(df['points']), include_groups=False)


print('\n')