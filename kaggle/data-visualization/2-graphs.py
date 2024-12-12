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



# -- Exercises with platform gaming
ign_data = pd.read_csv('./archive/ign_scores.csv', index_col='Platform')
# print(ign_data)

# What is the highest average score received by PC games for any genre?
high_score = ign_data.loc['PC'].max()
# print(high_score)

# On the Playstation Vita platform, which genre has the 
# lowest average score? Please provide the name of the column, and put your answer 
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
# worst_genre = ign_data.loc['PlayStation Vita'].min()
worst_genre = ign_data.loc['PlayStation Vita'].idxmin() # --> idxmin gets the column name for min val (col bc axis=0 for df which is default)
# print(f"worst_genre: {worst_genre}")

