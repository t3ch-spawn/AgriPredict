"""
Forecasting Module
Forecast commodity prices for a specific year using XGBoost models
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict


def forecast_year(
    commodity: str,
    state: str,
    target_year: int,
    data_path: str = './my_food_prices_avg.csv',
    model_dir: str = 'saved_models'
) -> Dict:
    """
    Forecast prices for a specific commodity, state, and year

    Parameters:
    -----------
    commodity : str
        The commodity name (e.g., 'Rice (local)', 'Beans (red)')
    state : str
        The state name (e.g., 'Adamawa', 'Borno')
    target_year : int
        Year to forecast (e.g., 2026, 2027)
    data_path : str
        Path to CSV data file
    model_dir : str
        Directory containing trained models

    Returns:
    --------
    dict : Contains forecasted prices for each month

    Example:
    --------
    >>> result = forecast_year('Rice (local)', 'Adamawa', 2026)
    >>> print(result['data'])
    >>> print(result['statistics']['average_forecast'])
    """

    # Load models and metadata
    trained_models, model_metadata = _load_models(model_dir)

    # Check if model exists
    if commodity not in trained_models:
        return {
            'status': 'ERROR',
            'message': f'Model for {commodity} not found. Available commodities: {list(trained_models.keys())}'
        }

    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Get historical data for feature engineering
    commodity_state_data = df[
        (df['commodity'] == commodity) &
        (df['state'] == state)
    ].sort_values('date').tail(12)

    if len(commodity_state_data) == 0:
        return {
            'status': 'ERROR',
            'message': f'No historical data found for {commodity} in {state}'
        }

    # Generate predictions for each month
    monthly_forecasts = []

    for month in range(1, 13):
        prediction = _predict_single_month(
            commodity,
            state,
            target_year,
            month,
            commodity_state_data,
            df,
            trained_models,
            model_metadata
        )

        if prediction['status'] == 'SUCCESS':
            monthly_forecasts.append({
                'date': f"{target_year}-{month:02d}",
                'price': f"{prediction['predicted_price']:.2f}"
            })

    if len(monthly_forecasts) == 0:
        return {
            'status': 'ERROR',
            'message': 'Failed to generate forecasts'
        }

    # Calculate forecast statistics
    prices = [float(f['price']) for f in monthly_forecasts]

    return {
        'status': 'SUCCESS',
        'commodity': commodity,
        'state': state,
        'year': target_year,
        'data': monthly_forecasts,
        'statistics': {
            'average_forecast': round(np.mean(prices), 2),
            'max_forecast': round(np.max(prices), 2),
            'min_forecast': round(np.min(prices), 2),
            'model_type': model_metadata[commodity]['type']
        }
    }


def _load_models(model_dir: str):
    """Load trained models and metadata"""
    trained_models = {}
    model_metadata = {}

    if not os.path.exists(model_dir):
        print(f"Warning: Model directory '{model_dir}' not found.")
        return trained_models, model_metadata

    # Load metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    if os.path.exists(metadata_path):
        model_metadata = joblib.load(metadata_path)

    # Load individual models
    for commodity in model_metadata.keys():
        model_type = model_metadata[commodity]['type']
        model_path = os.path.join(model_dir, f'{commodity}_{model_type}.pkl')

        if os.path.exists(model_path):
            trained_models[commodity] = joblib.load(model_path)

    return trained_models, model_metadata


def _predict_single_month(
    commodity: str,
    state: str,
    year: int,
    month: int,
    historical_data: pd.DataFrame,
    df: pd.DataFrame,
    trained_models: dict,
    model_metadata: dict
) -> Dict:
    """Internal method to predict price for a single month"""

    # Get recent prices for lag features
    recent_price = historical_data['price'].iloc[-1]
    price_lag_1 = historical_data['price'].iloc[-1]
    price_lag_3 = historical_data['price'].iloc[-3] if len(historical_data) >= 3 else recent_price
    price_lag_6 = historical_data['price'].iloc[-6] if len(historical_data) >= 6 else recent_price
    price_lag_12 = historical_data['price'].iloc[-12] if len(historical_data) >= 12 else recent_price

    rolling_3 = historical_data['price'].iloc[-3:].mean()
    rolling_6 = historical_data['price'].iloc[-6:].mean()
    rolling_12 = historical_data['price'].iloc[-12:].mean()

    # Calculate aggregate features
    state_avg_price_lag1 = df[
        (df['state'] == state) &
        (df['commodity'] == commodity)
    ]['price'].mean()

    commodity_avg_price_lag1 = df[df['commodity'] == commodity]['price'].mean()
    commodity_median_price = df[df['commodity'] == commodity]['price'].median()

    price_to_median_ratio = recent_price / commodity_median_price if commodity_median_price > 0 else 1.0
    price_momentum = historical_data['price'].iloc[-1] - historical_data['price'].iloc[-6] if len(historical_data) >= 6 else 0

    # Create feature vector
    features = [
        price_lag_1, price_lag_3, price_lag_6, price_lag_12,
        rolling_3, rolling_6, rolling_12,
        state_avg_price_lag1, commodity_avg_price_lag1,
        price_to_median_ratio, price_momentum
    ]

    X_future = np.array(features).reshape(1, -1)

    # Get prediction
    model = trained_models[commodity]
    prediction = model.predict(X_future)[0]

    # Apply calibration if volatile
    metadata = model_metadata[commodity]
    if metadata['type'] == 'volatile':
        variance_scale = metadata.get('variance_scale', 1.0)
        mean_diff = metadata.get('mean_diff', 0.0)
        prediction = (prediction - np.mean(features)) * variance_scale + np.mean(features) + mean_diff

    return {
        'status': 'SUCCESS',
        'predicted_price': round(prediction, 2),
        'model_type': metadata['type']
    }
