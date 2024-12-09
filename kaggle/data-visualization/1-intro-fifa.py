import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pd.plotting.register_matplotlib_converters()

cups = pd.read_csv('./archive/WorldCups.csv')

# print(cups.head())

# # Set the width and height of the figure
# plt.figure(figsize=(16,6))

# # Line chart showing how FIFA rankings evolved over time 
# sns.lineplot(data=cups, x='Year', y='GoalsScored')

# # Add title and labels
# plt.title('Goals Scored Over the Years in FIFA World Cups')
# plt.xlabel('Year')
# plt.ylabel('Goals Scored')

# plt.show()

# Show Goals Scored by Country Over the Years:-------------------------------------------------
grouped = cups.groupby(['Country', 'Year']).GoalsScored.sum().reset_index()

# Set the figure size
plt.figure(figsize=(16, 8))

# Plot the line graph
sns.lineplot(data=grouped, x='Year', y='GoalsScored', hue='Country', marker='o')

# Add labels and a title
plt.title('Goals Scored by Country Over the Years', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Goals Scored', fontsize=14)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.show()