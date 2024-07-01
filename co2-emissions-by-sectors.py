# In[2]:
import pandas as pd

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('CO2_emissions_by_Aviation.csv')

# Show the first few rows of the DataFrame
data.head()


# In[3]:


# Basic information about the dataset
num_entries = len(data)
num_countries = data['country'].nunique()
num_sectors = data['sector'].nunique()
date_range = (data['date'].min(), data['date'].max())

# Check for missing values
missing_values = data.isnull().sum()

num_entries, num_countries, num_sectors, date_range, missing_values


# In[4]:


# Unique countries and sectors in the dataset
unique_countries = data['country'].unique()
unique_sectors = data['sector'].unique()

unique_countries, unique_sectors


# In[5]:


import matplotlib.pyplot as plt

# Aggregate total CO2 emissions by country
total_emissions_by_country = data.groupby('country')['value'].sum().sort_values(ascending=False)

# Plot total CO2 emissions by country
plt.figure(figsize=(10, 6))
total_emissions_by_country.plot(kind='bar', color='steelblue')
plt.title('Total CO2 Emissions by Country')
plt.xlabel('Country')
plt.ylabel('Total CO2 Emissions')
plt.show()


# In[6]:


# Aggregate total CO2 emissions by sector
total_emissions_by_sector = data.groupby('sector')['value'].sum().sort_values(ascending=False)

# Plot total CO2 emissions by sector
plt.figure(figsize=(10, 6))
total_emissions_by_sector.plot(kind='bar', color='steelblue')
plt.title('Total CO2 Emissions by Aviation')
plt.xlabel('Aviation')
plt.ylabel('Total CO2 Emissions')
plt.show()


# In[7]:


# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Create a 'year' column for easy yearly aggregation
data['year'] = data['date'].dt.year

# Show the first few rows of the DataFrame to confirm changes
data.head()


# In[8]:


# Aggregate total CO2 emissions by year
total_emissions_by_year = data.groupby('year')['value'].sum()

# Plot total CO2 emissions by year
plt.figure(figsize=(10, 6))
total_emissions_by_year.plot(kind='line', marker='o', color='steelblue')
plt.title('Total CO2 Emissions by Year')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.xticks(total_emissions_by_year.index)
plt.grid(True)
plt.show()

# Aggregate total CO2 emissions by year and sector
total_emissions_by_year_sector = data.groupby(['year', 'sector'])['value'].sum().unstack()

# Plot total CO2 emissions by year for each sector
plt.figure(figsize=(12, 8))

for sector in unique_sectors:
    plt.plot(total_emissions_by_year_sector.index, total_emissions_by_year_sector[sector], marker='o', label=sector)

plt.title('Total CO2 Emissions by Year and Aviation')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.xticks(total_emissions_by_year_sector.index)
plt.legend(title='Aviation')
plt.grid(True)
plt.show()


# In[10]:


# Select the top emitting countries
top_countries = ['China', 'US', 'EU27 & UK']

# Aggregate total CO2 emissions by year and country for the top countries
total_emissions_by_year_country = data[data['country'].isin(top_countries)].groupby(['year', 'country'])['value'].sum().unstack()

# Plot total CO2 emissions by year for each top country
plt.figure(figsize=(12, 8))

for country in top_countries:
    plt.plot(total_emissions_by_year_country.index, total_emissions_by_year_country[country], marker='o', label=country)

plt.title('Total CO2 Emissions by Year for Top Emitting Countries')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.xticks(total_emissions_by_year_country.index)
plt.legend(title='Country')
plt.grid(True)
plt.show()


# In[11]:


# Aggregate total CO2 emissions by year, sector, and country for China
total_emissions_by_year_sector_china = data[data['country'] == 'China'].groupby(['year', 'sector'])['value'].sum().unstack()

# Plot total CO2 emissions by year for each sector in China
plt.figure(figsize=(12, 8))

for sector in unique_sectors:
    plt.plot(total_emissions_by_year_sector_china.index, total_emissions_by_year_sector_china[sector], marker='o', label=sector)

plt.title('Total CO2 Emissions by Year and Aviation for China')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.xticks(total_emissions_by_year_sector_china.index)
plt.legend(title='Aviation')
plt.grid(True)
plt.show()


# In[12]:


# Aggregate total CO2 emissions by year, sector, and country for the US
total_emissions_by_year_sector_us = data[data['country'] == 'US'].groupby(['year', 'sector'])['value'].sum().unstack()

# Plot total CO2 emissions by year for each sector in the US
plt.figure(figsize=(12, 8))

for sector in unique_sectors:
    plt.plot(total_emissions_by_year_sector_us.index, total_emissions_by_year_sector_us[sector], marker='o', label=sector)

plt.title('Total CO2 Emissions by Year and Aviation for the US')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.xticks(total_emissions_by_year_sector_us.index)
plt.legend(title='Aviation')
plt.grid(True)
plt.show()


# In[13]:


# Aggregate total CO2 emissions by year, sector, and country for the EU27 & UK
total_emissions_by_year_sector_eu27_uk = data[data['country'] == 'EU27 & UK'].groupby(['year', 'sector'])['value'].sum().unstack()

# Plot total CO2 emissions by year for each sector in the EU27 & UK
plt.figure(figsize=(12, 8))

for sector in unique_sectors:
    plt.plot(total_emissions_by_year_sector_eu27_uk.index, total_emissions_by_year_sector_eu27_uk[sector], marker='o', label=sector)

plt.title('Total CO2 Emissions by Year and Aviation for the EU27 & UK')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.xticks(total_emissions_by_year_sector_eu27_uk.index)
plt.legend(title='Aviation')
plt.grid(True)
plt.show()


# In[14]:


# Create a 'month' column for easy monthly aggregation
data['month'] = data['date'].dt.month

# Aggregate total CO2 emissions by year and month for the 'WORLD' category
total_emissions_by_year_month_world = data[data['country'] == 'WORLD'].groupby(['year', 'month'])['value'].sum().unstack()

# Plot total CO2 emissions by month for each year for the 'WORLD' category
plt.figure(figsize=(12, 8))

for year in total_emissions_by_year_month_world.index:
    plt.plot(total_emissions_by_year_month_world.columns, total_emissions_by_year_month_world.loc[year], marker='o', label=year)

plt.title('Total CO2 Emissions by Month for Each Year (WORLD)')
plt.xlabel('Month')
plt.ylabel('Total CO2 Emissions')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.grid(True)
plt.show()


# In[15]:


# Aggregate total CO2 emissions by year and month for China
total_emissions_by_year_month_china = data[data['country'] == 'China'].groupby(['year', 'month'])['value'].sum().unstack()

# Plot total CO2 emissions by month for each year for China
plt.figure(figsize=(12, 8))

for year in total_emissions_by_year_month_china.index:
    plt.plot(total_emissions_by_year_month_china.columns, total_emissions_by_year_month_china.loc[year], marker='o', label=year)

plt.title('Total CO2 Emissions by Month for Each Year (China)')
plt.xlabel('Month')
plt.ylabel('Total CO2 Emissions')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.grid(True)
plt.show()


# In[16]:


# Aggregate total CO2 emissions by year and month for the US
total_emissions_by_year_month_us = data[data['country'] == 'US'].groupby(['year', 'month'])['value'].sum().unstack()

# Plot total CO2 emissions by month for each year for the US
plt.figure(figsize=(12, 8))

for year in total_emissions_by_year_month_us.index:
    plt.plot(total_emissions_by_year_month_us.columns, total_emissions_by_year_month_us.loc[year], marker='o', label=year)

plt.title('Total CO2 Emissions by Month for Each Year (US)')
plt.xlabel('Month')
plt.ylabel('Total CO2 Emissions')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.grid(True)
plt.show()


# In[17]:


# Aggregate total CO2 emissions by year and month for the EU27 & UK
total_emissions_by_year_month_eu27_uk = data[data['country'] == 'EU27 & UK'].groupby(['year', 'month'])['value'].sum().unstack()

# Plot total CO2 emissions by month for each year for the EU27 & UK
plt.figure(figsize=(12, 8))

for year in total_emissions_by_year_month_eu27_uk.index:
    plt.plot(total_emissions_by_year_month_eu27_uk.columns, total_emissions_by_year_month_eu27_uk.loc[year], marker='o', label=year)

plt.title('Total CO2 Emissions by Month for Each Year (EU27 & UK)')
plt.xlabel('Month')
plt.ylabel('Total CO2 Emissions')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.grid(True)
plt.show()


# In[18]:


# Create a 'day_of_week' column (0 = Monday, 6 = Sunday)
data['day_of_week'] = data['date'].dt.dayofweek

# Aggregate average daily CO2 emissions by day of week and year for the 'WORLD' category
average_daily_emissions_by_day_of_week_world = data[data['country'] == 'WORLD'].groupby(['year', 'day_of_week'])['value'].mean().unstack()

# Plot average daily CO2 emissions by day of week for each year for the 'WORLD' category
plt.figure(figsize=(12, 8))

for year in average_daily_emissions_by_day_of_week_world.index:
    plt.plot(average_daily_emissions_by_day_of_week_world.columns, average_daily_emissions_by_day_of_week_world.loc[year], marker='o', label=year)

plt.title('Average Daily CO2 Emissions by Day of Week for Each Year (WORLD)')
plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Average Daily CO2 Emissions')
plt.xticks(range(7))
plt.legend(title='Year')
plt.grid(True)
plt.show()


# In[19]:


# Aggregate average daily CO2 emissions by day of week and year for China
average_daily_emissions_by_day_of_week_china = data[data['country'] == 'China'].groupby(['year', 'day_of_week'])['value'].mean().unstack()

# Plot average daily CO2 emissions by day of week for each year for China
plt.figure(figsize=(12, 8))

for year in average_daily_emissions_by_day_of_week_china.index:
    plt.plot(average_daily_emissions_by_day_of_week_china.columns, average_daily_emissions_by_day_of_week_china.loc[year], marker='o', label=year)

plt.title('Average Daily CO2 Emissions by Day of Week for Each Year (China)')
plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Average Daily CO2 Emissions')
plt.xticks(range(7))
plt.legend(title='Year')
plt.grid(True)
plt.show()


# In[20]:


# Aggregate average daily CO2 emissions by day of week and year for the US
average_daily_emissions_by_day_of_week_us = data[data['country'] == 'US'].groupby(['year', 'day_of_week'])['value'].mean().unstack()

# Plot average daily CO2 emissions by day of week for each year for the US
plt.figure(figsize=(12, 8))

for year in average_daily_emissions_by_day_of_week_us.index:
    plt.plot(average_daily_emissions_by_day_of_week_us.columns, average_daily_emissions_by_day_of_week_us.loc[year], marker='o', label=year)

plt.title('Average Daily CO2 Emissions by Day of Week for Each Year (US)')
plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Average Daily CO2 Emissions')
plt.xticks(range(7))
plt.legend(title='Year')
plt.grid(True)
plt.show()


# In[21]:


# Aggregate average daily CO2 emissions by day of week and year for the EU27 & UK
average_daily_emissions_by_day_of_week_eu27_uk = data[data['country'] == 'EU27 & UK'].groupby(['year', 'day_of_week'])['value'].mean().unstack()

# Plot average daily CO2 emissions by day of week for each year for the EU27 & UK
plt.figure(figsize=(12, 8))

for year in average_daily_emissions_by_day_of_week_eu27_uk.index:
    plt.plot(average_daily_emissions_by_day_of_week_eu27_uk.columns, average_daily_emissions_by_day_of_week_eu27_uk.loc[year], marker='o', label=year)

plt.title('Average Daily CO2 Emissions by Day of Week for Each Year (EU27 & UK)')
plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Average Daily CO2 Emissions')
plt.xticks(range(7))
plt.legend(title='Year')
plt.grid(True)
plt.show()