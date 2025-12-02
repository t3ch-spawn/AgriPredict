"""
State Comparison Module
Compare forecasted prices across multiple states with analytics
"""

import pandas as pd
from typing import Dict, List
from .forecast import forecast_year


def compare_states_forecast(
    commodity: str,
    states: List[str],
    target_year: int,
    data_path: str = './my_food_prices_avg.csv',
    model_dir: str = 'saved_models'
) -> Dict:
    """
    Compare forecasted prices across multiple states for a commodity
    Returns data in format suitable for multi-line graph and analytics

    Parameters:
    -----------
    commodity : str
        The commodity name (e.g., 'Rice (local)', 'Beans (red)')
    states : list
        List of state names (e.g., ['Yobe', 'Adamawa', 'Borno'])
    target_year : int
        Year to forecast and compare
    data_path : str
        Path to CSV data file
    model_dir : str
        Directory containing trained models

    Returns:
    --------
    dict : Contains multi-line graph data and analytics including:
        - Cheapest state
        - Most expensive state
        - Best month to buy

    Example:
    --------
    >>> result = compare_states_forecast('Rice (local)', ['Yobe', 'Adamawa', 'Borno'], 2026)
    >>> print(result['graph_data'])
    >>> print(result['analytics']['cheapest_state'])
    >>> print(result['analytics']['best_month_to_buy'])
    """

    # Generate forecasts for each state
    state_forecasts = []
    all_prices = []

    for state in states:
        forecast = forecast_year(commodity, state, target_year, data_path, model_dir)

        if forecast['status'] == 'SUCCESS':
            state_forecasts.append({
                'state': state,
                'data': forecast['data']
            })

            # Collect all prices for analytics
            prices = [float(d['price']) for d in forecast['data']]
            all_prices.extend([(state, month, price) for month, price in enumerate(prices, 1)])

    if len(state_forecasts) == 0:
        return {
            'status': 'ERROR',
            'message': 'Failed to generate forecasts for any state'
        }

    # Analytics: Find cheapest state, most expensive state, best month to buy
    df_analysis = pd.DataFrame(all_prices, columns=['state', 'month', 'price'])

    # State with cheapest average price
    state_avg = df_analysis.groupby('state')['price'].mean()
    cheapest_state = state_avg.idxmin()
    cheapest_avg_price = state_avg.min()

    # State with most expensive average price
    most_expensive_state = state_avg.idxmax()
    most_expensive_avg_price = state_avg.max()

    # Best month to buy (lowest average price across all states)
    month_avg = df_analysis.groupby('month')['price'].mean()
    best_month_to_buy = int(month_avg.idxmin())
    best_month_price = month_avg.min()

    # Month names for better readability
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    return {
        'status': 'SUCCESS',
        'commodity': commodity,
        'year': target_year,
        'graph_data': state_forecasts,
        'analytics': {
            'cheapest_state': {
                'state': cheapest_state,
                'average_price': round(cheapest_avg_price, 2)
            },
            'most_expensive_state': {
                'state': most_expensive_state,
                'average_price': round(most_expensive_avg_price, 2)
            },
            'best_month_to_buy': {
                'month': best_month_to_buy,
                'month_name': month_names[best_month_to_buy - 1],
                'average_price': round(best_month_price, 2)
            },
            'price_difference': round(most_expensive_avg_price - cheapest_avg_price, 2),
            'savings_potential': round((most_expensive_avg_price - cheapest_avg_price) / most_expensive_avg_price * 100, 2)
        }
    }
