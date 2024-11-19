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
reviews.taster_name.describe() # see aggregate info on taster_name
reviews.points.mean() # see average of points allotted
reviews.taster_name.unique() # see all the taster names
print(reviews.taster_name.value_counts()) # See tasters and how often they occur in data














# nums = [1,3,4,6,8,10,13], target = 13
def two_sum(nums, target):
    # pointers on the outside indexes
    left = 0
    right = len(nums) - 1

    # if curr sum > target: all with current "right" will be greater. move "right" back one
    # if curr < target: all with left will be smaller. move "left" up one

    while left < right: # while they haven't met
        sum = nums[left] + nums[right]

        if sum == target:
            return True

        if sum > target:
            right -= right;
            continue;
        
        if sum < target:
            left += left;
            continue;
    
    return False