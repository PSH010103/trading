import pandas as pd

data = pd.read_csv('./leadingindicators/nonfarm_expect.csv')
# Create a DataFrame
df = pd.DataFrame(data)

# Convert 'Release Date' to datetime format (we'll ignore the month in parentheses)
df['Release Date'] = pd.to_datetime(df['Release Date'].str.split(' ').str[:3].apply(' '.join), format='%b %d %Y')

# Set 'Release Date' as the index
df.set_index('Release Date', inplace=True)
# Drop the 'Time' and 'Previous' columns
df.drop(columns=['Time', 'Previous'], inplace=True)

print(df)
# Subtract 'Forecast' from 'Actual' and create a new column 'Nonfarm Payrolls'
df['Nonfarm Payrolls'] = - pd.to_numeric(df['Actual'].str.replace('K', '000')) + pd.to_numeric(df['Forecast'].str.replace('K', '000'))

# Drop the 'Actual' and 'Forecast' columns
df.drop(columns=['Actual', 'Forecast'], inplace=True)
df = df[['Nonfarm Payrolls']]
df.index.name = 'date'

df.to_csv('./leadingindicators/nonfarm.csv')