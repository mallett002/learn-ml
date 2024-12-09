import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.plotting.register_matplotlib_converters()

museum_data = pd.read_csv('./archive/museums.csv')

# -- Create line plot --
plt.figure(figsize=(16,6))

sns.lineplot(data=museum_data)

# plt.show()


# -- Bar graph --
flight_data = pd.read_csv('./archive/flight_delays.csv', index_col='Month')
print(flight_data.head())

# bar chart showing the average arrival delay for Spirit Airlines (airline code: NK) flights, by month.
plt.figure(figsize=(10, 6))

# title
plt.title('Average Arrival Delay for Spirit Airlines Flights, by Month')

# Bar chart showing average arrival delay for Spirit Airlines flights by month
month = flight_data.index
sns.barplot(x=month, y=flight_data['NK'], palette='viridis', hue=month) # --> need to use data.index for month bc we made it the index
plt.ylabel('Average delay (in minutes)')

plt.show()
