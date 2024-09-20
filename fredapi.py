import requests
import pandas as pd
from api_keys import FRED_API_KEY
from datetime import datetime

def fetch_fred_data(series_ids, api_key, start_date, end_date):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    all_data = []

    for series_id in series_ids:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": "m",  # monthly frequency
            "units": "lin"  # linear units
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if 'observations' in data:
            series_data = pd.DataFrame(data['observations'])
            series_data['date'] = pd.to_datetime(series_data['date'])
            series_data['date'] = series_data['date'] + pd.offsets.MonthEnd(1)
            series_data = series_data.rename(columns={'value': series_id})
            series_data = series_data.set_index('date')
            all_data.append(series_data[[series_id]])

    combined_df = pd.concat(all_data, axis=1)
    return combined_df

# Example usage
api_key = FRED_API_KEY
series_ids = [
    "UNRATE",    # Unemployment rate
    "AWHMAN",    # Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing
    "ICSA",      # Initial Claims
    "ACOGNO",    # Manufacturers' New Orders: Consumer Goods and Materials
    "NEWORDER",  # Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft
    "PERMIT",    # New Private Housing Units Authorized by Building Permits
    "SP500",     # S&P 500
    "T10YFF",    # 10-Year Treasury Constant Maturity Minus Federal Funds Rate
    "USALOLITOAASTSAM", # Leading indicator
    "T10Y2Y",    # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    "UMCSENT",  # University of Michigan: Consumer Sentiment
    "A191RL1Q225SBEA", # Real Gross Domestic Product, Percent Change from Preceding Period
    "FEDFUNDS",  # Federal Funds Effective Rate
    'VIXCLS',    # VIX data
]

start_date = "2015-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# Fetch data
df = fetch_fred_data(series_ids, api_key, start_date, end_date)

# Display the first few rows of the DataFrame
print(df.head())
df.to_csv('./leadingindicators/fred.csv')