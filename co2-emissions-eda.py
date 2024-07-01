import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.io as pio


# # Load Data

# In[5]:

# pio.renderers.default = 'svg'
data = pd.read_csv('CO2_emissions_by_Aviation.csv')


# In[6]:


data.shape


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


#No null values present
data.isnull().sum()


# In[11]:


data['date'] = pd.to_datetime(data['date'],dayfirst=True)
data['Year'] = data['date'].dt.year


# # Exploratory Data Analysis
# 

# ### Comprehensive Analysis of Country-wise CO2 Emissions Since 2019
# 
# **Observations**
# 1. The total global CO2 emissions since 2019 have reached a significant level of 155k MtCO2/day.
# 2. Among the major contributors to CO2 emissions, China takes the lead, followed closely by the United States and the combined emissions of EU27 & UK.

# In[12]:


co2_emissions_by_country = data.groupby('country').sum(numeric_only=True)['value'].reset_index()

# Create the bar graph
fig = go.Figure(data=[go.Bar(x=co2_emissions_by_country['country'], y=co2_emissions_by_country['value'],marker=dict(color='rgba(255, 0, 0, 0.6)'))])

# Customize the layout
fig.update_layout(
    title="Total CO2 Emission by country since 2019",
    xaxis_title="Country",
    yaxis_title="CO2 Emissions (MtCO2/day)",
)

# Display the graph
fig.show()


# In[13]:


co2_emissions_by_country = co2_emissions_by_country[~co2_emissions_by_country['country'].isin(['WORLD', 'ROW', 'EU27 & UK'])]

px.choropleth(co2_emissions_by_country, locations='country', locationmode='country names', color='value',
              hover_name='country', color_continuous_scale='Viridis', projection='natural earth')


# ### Time Series for Global CO2 Emissions for each year
# 
# **Observations**
# 1. The time series plot illustrates that during winters (November to February), global CO2 emissions tend to rise due to an increase in the demand for heating, which leads to higher energy consumption.

# In[14]:


#Extract World CO2 Emissions
data_world = data[data['country']=='WORLD']
df = data_world.groupby('date').sum(numeric_only=True)['value'].reset_index() #Group by date
df = df.sort_values(by='date') #Sort values by date

# Extract year from each data point
df['Year'] = df['date'].dt.year

# Create subplots for each year
fig = make_subplots(rows=len(df['Year'].unique()), cols=1, shared_xaxes=True,subplot_titles=[f"Year {year}" for year in df['Year'].unique()],
    vertical_spacing=0.1)

# Create separate graphs for each year
for idx, year in enumerate(df['Year'].unique(), 1):
    df_year = df[df['Year'] == year]

    fig.add_trace(
        go.Scatter(
            x=df_year['date'].dt.dayofyear,
            y=df_year['value'],
            mode='lines',
            name=str(year),
            hovertext=df_year['date'].dt.strftime('%Y-%m-%d'),
            hoverinfo= 'text+y',
            line_shape='linear'
        ),
        row=idx,
        col=1
    )

    # Update layout for each subplot
    fig.update_xaxes(title_text="Date", row=idx, col=1)
    fig.update_yaxes(title_text="CO2 Emission", row=idx, col=1)

    # Set the x-axis tick positions and labels to be the first day of each month
    month_starts = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='MS')
    tickvals = month_starts.dayofyear
    ticktext = month_starts.strftime('%b')

    fig.update_xaxes(row=idx, col=1, tickvals=tickvals, ticktext=ticktext)


# Update the overall layout
fig.update_layout(
    title="Global CO2 Emission Time Series for Each Year",
    showlegend=False,
    height=1500
)

# Show the plot
fig.show()


# ### Time Series Plot for Global CO2 Emissions
# 
# **Observations**
# 
# 1. Seasonality is evident in the years 2019, 2021, 2022, and 2023, with lower CO2 emissions during summers and higher emissions during winters, making this dataset suitable for time series analysis.
# 2. Notably, the year 2020 stands as an outlier due to the significant dip in CO2 emissions observed during the lockdown period, reflecting the impact of reduced economic activity on global emissions.

# In[15]:


data_world = data[data['country'] == 'WORLD']
df = data_world.groupby('date').sum(numeric_only=True)['value'].reset_index()  # Group by date
df = df.sort_values(by='date')  # Sort values by date
df['Year'] = df['date'].dt.year
today = pd.to_datetime('today')
df = df[df['date'] <= today]


# Create a single graph for all years
fig = go.Figure()

# Add traces for each year to the same graph
for year in df['Year'].unique():
    df_year = df[df['Year'] == year]

    fig.add_trace(
        go.Scatter(
            x=df_year['date'],
            y=df_year['value'],
            mode='lines',
            name=str(year),
            hoverinfo='x+y',
            line_shape='linear'
        )
    )

# Update layout for the graph
fig.update_layout(
    title="Global CO2 Emission Time Series",
    xaxis_title="Date",
    yaxis_title="CO2 Emission (MtCO2/day)",
    showlegend=True,
    height=800
)

# Show the plot
fig.show()


# ### Time Series Plot for CO2 Emissions of each country

# In[16]:


def plot_country_co2_emissions(country):
    data_country = data[data['country'] == country]
    df = data_country.groupby('date').sum(numeric_only=True)['value'].reset_index()
    df = df.sort_values(by='date')
    df['Year'] = df['date'].dt.year

    # Define a list of colors with transparency (alpha)
    colors = {
        2021: 'rgba(0, 0, 255, 0.3)',
        2022: 'rgba(0, 128, 0, 0.5)',
        2023: 'rgba(255, 0, 0, 0.6)'
    }

    fig = go.Figure()

    for year in [2021, 2022, 2023]:
        df_year = df[df['Year'] == year]

        fig.add_trace(go.Scatter(
            x=df_year['date'].dt.dayofyear,
            y=df_year['value'],
            mode='lines',
            hovertext=df_year['date'].dt.strftime('%Y-%m-%d'),  # Use the new hover_text column for hoverinfo
            hoverinfo='text+y',  # Show the full date and value on hover
            line_shape='linear',  # Use a linear line shape for smoother lines
            line=dict(color=colors[year]),  # Assign the color based on the year
            name=str(year)
        ))

        # Set the x-axis tick positions and labels to be the first day of each month
        month_starts = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='MS')
        tickvals = month_starts.dayofyear
        ticktext = month_starts.strftime('%b')
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

    # Update the layout
    fig.update_layout(
        title= country + " CO2 Emission Time Series",
        xaxis=dict(title="Month", tickformat="%b"),  # Display only the abbreviated month name
        yaxis=dict(title="CO2 Emission (MtCO2/day)"),
        legend_title="Year",
        hovermode='x',
    )

    # Show the plot
    fig.show()

for country in data['country'].unique():
    plot_country_co2_emissions(country)


# ### Total CO2 Emissions by sector since 2019
# 
# **Observations**
# 
# 1. The power sector leads with 60k MtCO2/day CO2 emissions, closely followed by industries and group transportation.
# 2. Domestic and international aviation, while still contributing to CO2 emissions, have comparatively lesser responsibility for the total emissions.

# In[17]:


data_world = data[data['country']=='WORLD']

co2_emissions_by_sector = data_world.groupby('sector').sum(numeric_only=True)['value'].reset_index()

# Create the bar graph
fig = go.Figure(data=[go.Bar(x=co2_emissions_by_sector['sector'], y=co2_emissions_by_sector['value'],marker=dict(color='rgba(255, 0, 0, 0.6)'))])

# Customize the layout
fig.update_layout(
    title="Global CO2 Emission by Aviation since 2019",
    xaxis_title="Aviation",
    yaxis_title="CO2 Emissions (MtCO2/day) ",
)

# Display the graph
fig.show()


# ### Box Plot for CO2 Emissions by sector since 2019

# In[18]:


data_world = data[data['country']=='WORLD']

# Create a box plot using Plotly Express
fig = px.box(data_world, x='sector', y='value',
             title="Box Plot of Global CO2 Emissions by Aviation",
             labels={'value': 'CO2 Emissions (MtCO2/day)', 'sector': 'Sector'})

# Display the graph
fig.show()

data_world = data[data['country']=='WORLD']

# Create a time series graph using Plotly Express
fig = px.line(data_world, x='date', y='value', color='sector',
              title="Time Series Plot of Global CO2 Emissions by Aviation",
              labels={'value': 'CO2 Emissions (MtCO2/day)', 'sector': 'Sector', 'date': 'Date'})

# Display the graph
fig.show()


# ### Time Series Plot for each country CO2 Emissions by Sector
# 
# **Observations**
# 
# 1. Different regions of the world have varying sectors contributing more to CO2 emissions.
# 2. Residential CO2 emissions exhibit seasonality in almost all countries, with some experiencing more significant rises and peaks, such as France, Germany, UK, and US due to colder climates. However, Brazil stands out as the only country with relatively stable Residential CO2 emissions throughout the year.
# 3. Russia's CO2 emissions from the power sector surpass those of other sectors, which is a unique characteristic not as prominently observed in other countries. There is a slightly noticeable trend in Japan and India as well, though in most countries, the power sector remains the biggest contributor to CO2 emissions.
# 4. Brazil has shown a noticeable decrease in CO2 emissions from the power sector since the end of 2021.

# In[20]:


# Create a dictionary to map each sector to a color with modified opacity
sector_colors = {
    'Power': 'rgba(31, 119, 180, 0.8)',             # Blue
    'Industry': 'rgba(255, 127, 14, 0.7)',          # Orange
    'Ground Transport': 'rgba(44, 160, 44, 0.6)',   # Green
    'Residential': 'rgba(214, 39, 40, 0.6)',        # Red
    'Domestic Aviation': 'rgba(148, 103, 189, 0.4)',# Purple
    'International Aviation': 'rgba(140, 86, 75, 0.3)'# Brown
}

def plot_country_and_sector_co2_emissions(fig, data_country, row):
    for sector, sector_data in data_country.groupby('sector'):
        fig.add_trace(
            go.Scatter(x=sector_data['date'], y=sector_data['value'],
                       mode='lines', line=dict(width=2, color=sector_colors[sector]),
                       name=sector, showlegend=(row == 1)  # Show legend only for the first country
            ),
            row=row, col=1
        )

# Get a list of unique countries
unique_countries = data['country'].unique()

# Create subplots with one column and a number of rows based on the number of unique countries
fig = make_subplots(rows=len(unique_countries), cols=1,
                    subplot_titles=[country + " Time Series Plot of CO2 Emissions by Sector" for country in unique_countries])

# Plot data for each country in separate subplots
for i, country in enumerate(unique_countries):
    data_country = data[data['country'] == country]
    plot_country_and_sector_co2_emissions(fig, data_country, row=i + 1)

# Update layout and display the graph
fig.update_layout(showlegend=True, height=6000, title_text="CO2 Emissions by Aviation for Different Countries")

fig.show()


# ### Total Country & Sector Distribution since 2019
# 
# **Observations**
# 
# 1. China stands as the largest contributor to CO2 emissions globally.
# 2. Despite being the second-largest country by population, India exhibits significantly lower CO2 emissions per capita compared to other nations.

# In[21]:


co2_emissions_by_country = data.groupby(['country','sector']).sum(numeric_only=True)['value'].reset_index()
co2_emissions_by_country = co2_emissions_by_country[~co2_emissions_by_country['country'].isin(['WORLD'])]

# Create a faceted bar chart using Plotly Express
fig = px.bar(co2_emissions_by_country, x='country', y='value', color='sector', barmode='group',
             title="CO2 Emission by country & Aviation since 2019",
             labels={'value': 'CO2 Emissions (MtCO2/day)', 'country': 'Country','sector':'Sector'})

# Display the graph
fig.show()

