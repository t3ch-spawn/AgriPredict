"""
Historical Price Analysis Module
Get historical prices with ATH, ATL, average, and current price statistics
"""

import pandas as pd
from typing import Dict


def get_historical_prices(
    commodity,
    state,
    start_date = "2024-11",
    end_date = "2025-11",
    data_path = './my_food_prices_avg.csv'
) -> Dict:
    """
    Get historical prices for a product from start_date to end_date
    Returns: all time high, all time low, average price, current price

    Parameters:
    -----------
    commodity : str
        The commodity name (e.g., 'Rice (local)', 'Beans (red)')
    state : str
        The state name (e.g., 'Adamawa', 'Borno')
    start_date : str
        Start date in format 'YYYY-MM' (default: '2024-11')
    end_date : str
        End date in format 'YYYY-MM' (default: '2025-11')
    data_path : str
        Path to CSV data file

    Returns:
    --------
    dict : Contains historical data and statistics (ATH, ATL, average, current price)

    Example:
    --------
    >>> result = get_historical_prices('Rice (local)', 'Adamawa', '2024-11', '2025-11')
    >>> print(result['statistics']['all_time_high'])
    >>> print(result['data'])
    """

    # Load data
    df = pd.read_csv('./my_food_prices_avg.csv')
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Convert string dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Filter data
    filtered_data = df[
        (df["commodity"] == commodity) &
        (df["state"] == state) &
        (df["date"] >= start) &
        (df["date"] <= end)
    ].copy()

    if len(filtered_data) == 0:
        return {
            'status': 'ERROR',
            'message': f'No data found for {commodity} in {state} between {start_date} and {end_date}'
        }

    # Sort by date
    filtered_data = filtered_data.sort_values("date")

    # Format data for output
    historical_data = [
        {
            "date": row["date"].strftime("%Y-%m"),
            "price": f"{row['price']:.2f}"
        }
        for _, row in filtered_data.iterrows()
    ]

    # Calculate statistics
    all_time_high = float(filtered_data["price"].max())
    all_time_low = float(filtered_data["price"].min())
    average_price = float(filtered_data["price"].mean())
    current_price = float(filtered_data.iloc[-1]["price"])

    # Find when ATH and ATL occurred
    ath_date = filtered_data.loc[filtered_data["price"].idxmax(), "date"].strftime("%Y-%m")
    atl_date = filtered_data.loc[filtered_data["price"].idxmin(), "date"].strftime("%Y-%m")

    return {
        'status': 'SUCCESS',
        'commodity': commodity,
        'state': state,
        'period': {
            'start': start_date,
            'end': end_date
        },
        'data': historical_data,
        'statistics': {
            'all_time_high': {
                'price': round(all_time_high, 2),
                'date': ath_date
            },
            'all_time_low': {
                'price': round(all_time_low, 2),
                'date': atl_date
            },
            'average_price': round(average_price, 2),
            'current_price': round(current_price, 2),
            'price_range': round(all_time_high - all_time_low, 2),
            'volatility': round(filtered_data["price"].std(), 2)
        }
    }

result1 = get_historical_prices(
    commodity='Rice (local)',
    state='Adamawa',
    start_date='2024-11',
    end_date='2025-11'
)

print(result1)