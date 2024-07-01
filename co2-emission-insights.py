#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'CO2_emissions_by_Aviation.csv'
data = pd.read_csv(file_path)

# Display basic information about the data
basic_info = data.info()
basic_info

# Display the first few rows of the data
data.head()



# In[2]:


# Checking for missing values and duplicate entries
missing_values = data.isnull().sum()
duplicate_rows = data.duplicated().sum()

# Convert 'date' column to datetime format for better analysis
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Check the unique values and potentially inconsistent entries in 'country' and 'sector' columns
unique_countries = data['country'].unique()
unique_sectors = data['sector'].unique()

missing_values, duplicate_rows, unique_countries, unique_sectors


# Basic Trend Analysis showing how CO2 emissions changed over time in each country? Are there any noticeable trends or patterns?****

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Grouping data by country and date and summing up the CO2 values
grouped_data = data.groupby(['country', 'date']).agg({'value': 'sum'}).reset_index()

# Plotting the trends of CO2 emissions over time for each country
plt.figure(figsize=(15, 8))

# Ensuring each country has a different color
countries = grouped_data['country'].unique()
palette = sns.color_palette("hsv", len(countries))

sns.lineplot(x='date', y='value', hue='country', palette=palette, data=grouped_data)
plt.title('Trend of CO2 Emissions Over Time by Country')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# Sectors that contribute the most to CO2 emissions in each country, and how did that vary across different countries?

# In[4]:


# Grouping data by country and sector to sum up CO2 emissions
sectorwise_emissions = data.groupby(['country', 'sector']).agg({'value': 'sum'}).reset_index()

# Plotting CO2 emissions by sector for each country
plt.figure(figsize=(15, 8))

sns.barplot(x='value', y='country', hue='sector', data=sectorwise_emissions)
plt.title('CO2 Emissions by Aviation in Each Country')
plt.xlabel('Total CO2 Emissions')
plt.ylabel('Country')
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# Heatmap to provide a clearer and more detailed visualization of the CO2 emissions by sector in each country.

# In[5]:


# Creating a pivot table for a better visualization
pivot_data = sectorwise_emissions.pivot(index='country', columns='sector', values='value')

# Plotting the pivot data
plt.figure(figsize=(15, 10))

sns.heatmap(pivot_data, annot=True, fmt=".2f", linewidths=.5, cmap='YlGnBu')
plt.title('Heatmap of CO2 Emissions by Aviation in Each Country')
plt.xlabel('Aviation')
plt.ylabel('Country')

plt.show()

