import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()

museum_data = pd.read_csv('./archive/museums.csv')
flight_data = pd.read_csv('./archive/flight_delays.csv', index_col='Month')

# -- Create line plot --
# plt.figure(figsize=(16,6))

# sns.lineplot(data=museum_data)

# plt.show()


# -- Bar graph --
# # bar chart showing the average arrival delay for Spirit Airlines (airline code: NK) flights, by month.
# plt.figure(figsize=(10, 6))

# # title
# plt.title('Average Arrival Delay for Spirit Airlines Flights, by Month')

# # Bar chart showing average arrival delay for Spirit Airlines flights by month
# month = flight_data.index
# sns.barplot(x=month, y=flight_data['NK'], palette='viridis', hue=month) # --> need to use data.index for month bc we made it the index
# plt.ylabel('Average delay (in minutes)')

# plt.show()





# -- Heat Map --
# plt.figure(figsize=(14,7))
# plt.title("Average Arrival Delay for Each Airline, by Month")
# plt.xlabel("Airline")

# sns.heatmap(data=flight_data, annot=True) # annot puts numbers on heatmap

# plt.show()



# -- Exercises with platform gaming -----------------------------------------------------------------------------------------------------------
ign_data = pd.read_csv('./archive/ign_scores.csv', index_col='Platform')
# print(ign_data)




# What is the highest average score received by PC games for any genre?
high_score = ign_data.loc['PC'].max()
# print(high_score)





# On the Playstation Vita platform, which genre has the 
worst_genre = ign_data.loc['PlayStation Vita'].idxmin() # --> idxmin gets the column name for min val (col bc axis=0 for df which is default)
# print(f"worst_genre: {worst_genre}")




# ------------------- Bar Chart for racing ---------------------------------
# Bar chart that shows the average score for racing games, for each platform. Your chart should have one bar for each platform.
# racing_by_platform = ign_data.groupby('Platform')['Racing']

# print(ign_data.head())

# What I did:
# plt.figure(figsize=(10, 6))
# plt.title('Avarage scores for racing games by platform')
# platform = ign_data.index
# sns.barplot(x=platform, y=ign_data['Racing'], palette='viridis', hue=platform)
# plt.ylabel('avarage scores by platform')
# plt.show()

# What they did:
# plt.figure(figsize=(8, 6))
# sns.barplot(x=ign_data['Racing'], y=ign_data.index, palette='viridis', hue=ign_data.index)
# plt.xlabel("")
# plt.title("Average scores for racing by platform")
# plt.show()




# ----------- Heat map of average scores by genre and platform -------------------------
# plt.figure(figsize=(20,10))
# plt.title("Average scores by platform and genre")
# plt.xlabel("Genre")
# sns.heatmap(ign_data, annot=True)
# plt.show()


# ---------- Average scores in total by platform -------------------
# Calculate the average score for each platform (row-wise)
# average_by_platform = ign_data.mean(axis=1) # average by row (axis column)

# Sort platforms by average score
# sorted_by_scores = average_by_platform.sort_values(ascending=False)
# print(sorted_by_scores)

# plt.figure(figsize=(10, 6))
# plt.title('Avarage Scores by Platform')
# sns.barplot(
#     y=sorted_by_scores.index,
#     x=sorted_by_scores.values,
#     palette='crest',
#     hue=sorted_by_scores.index,
#     legend=False)
# plt.xlabel('')
# plt.ylabel('Platform')
# plt.show()




# ---------- Scatter Plots -------------------
insurance_data = pd.read_csv('./archive/insurance.csv')

# print(insurance_data.head())

# scatterplot:
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# regression line (help see line of best fit for scatter plot)
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# color code points by 'smoker' and plot other columns (bmi & charges)
# ie. same data as above, but color codes if they are smoker or not
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])

# 2 regression lines
sns.lmplot(x='bmi', y='charges', hue='smoker', data=insurance_data)

plt.show()








