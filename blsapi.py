from api_keys import BLS_API_KEY
import requests
import json
import pandas as pd

def fetch_bls_data(series_ids, start_year, end_year, api_key):
    url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {'Content-type': 'application/json'}
    data = json.dumps({
        "seriesid": series_ids,
        "startyear": start_year,
        "endyear": end_year,
        "registrationkey": api_key
    })

    response = requests.post(url, data=data, headers=headers)
    json_data = json.loads(response.text)
    
    
    return json_data

def json_to_dataframe(json_data, series_names):
    all_series_data = []
    
    for series in json_data['Results']['series']:
        series_id = series['seriesID']
        series_data = series['data']
        
        # Convert to DataFrame
        df = pd.DataFrame(series_data)
        
        # Convert 'value' column to numeric type
        df['value'] = pd.to_numeric(df['value'])
        
        # Convert 'year' and 'period' to datetime
        df['date'] = pd.to_datetime(df['year'] + df['period'], format='%YM%m')
        
        # Shift dates by one month
        df['date'] = df['date'] + pd.offsets.MonthEnd(1)
        
        # Keep only date and value, and rename value column to series name
        df = df[['date', 'value']]
        df = df.rename(columns={'value': series_names.get(series_id, series_id)})
        
        all_series_data.append(df)
    
    # Merge all series data on the date column
    combined_df = pd.concat(all_series_data, axis=1)
    
    # Remove duplicate date columns
    combined_df = combined_df.loc[:,~combined_df.columns.duplicated()]
    
    # Set date as index and sort
    combined_df = combined_df.set_index('date').sort_index()
    
    return combined_df


# Example usage
api_key = "7424f795f85a4eb1a4d79252176bf376"


# Manual dictionary for series names
series_names = {
    "CES0000000001": "Nonfarm employment", # Difference to get nonfarm payrolls
}

series_ids = list(series_names.keys())

# Fetch data
result = fetch_bls_data(series_ids, 2015, 2024, BLS_API_KEY)

# Convert to DataFrame with custom column names
df = json_to_dataframe(result, series_names)
df.to_csv('./leadingindicators/bls.csv')
# Display the first few rows of the DataFrame
print(df.tail())
