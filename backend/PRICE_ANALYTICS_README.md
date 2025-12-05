# AgriPredict Price Analytics Functions

This module provides three main functions for commodity price analysis and forecasting.

## Features

### 1. Historical Price Analysis (`get_historical_prices`)
Get historical prices for a product from Nov 2024 to Nov 2025 (or any custom date range).

**Returns:**
- Historical price data
- All-time high (ATH)
- All-time low (ATL)
- Average price
- Current price
- Volatility metrics

### 2. Year Forecasting (`forecast_year`)
Forecast prices for a specific commodity and state for an entire year.

**Returns:**
- Monthly forecasted prices
- Average, max, and min forecast
- Model type (normal/volatile)

### 3. State Comparison (`compare_states_forecast`)
Compare forecasted prices across multiple states with multi-line graph data.

**Returns:**
- Graph data for multiple states (ready for plotting)
- Cheapest state
- Most expensive state
- Best month to buy
- Savings potential percentage

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas numpy xgboost scikit-learn joblib
```

## Usage

### Function 1: Get Historical Prices

```python
from price_analytics import get_historical_prices

result = get_historical_prices(
    commodity='Rice (local)',
    state='Adamawa',
    start_date='2024-11',
    end_date='2025-11'
)

print(result)
```

**Output Structure:**
```python
{
    'status': 'SUCCESS',
    'commodity': 'Rice (local)',
    'state': 'Adamawa',
    'period': {
        'start': '2024-11',
        'end': '2025-11'
    },
    'data': [
        {'date': '2024-11', 'price': '3000.00'},
        {'date': '2024-12', 'price': '3100.00'},
        # ... more data points
    ],
    'statistics': {
        'all_time_high': {
            'price': 3883.34,
            'date': '2025-01'
        },
        'all_time_low': {
            'price': 2316.67,
            'date': '2025-10'
        },
        'average_price': 3400.56,
        'current_price': 2316.67,
        'price_range': 1566.67,
        'volatility': 736.33
    }
}
```

### Function 2: Forecast Year

```python
from price_analytics import forecast_year

result = forecast_year(
    commodity='Rice (local)',
    state='Adamawa',
    target_year=2026
)

print(result)
```

**Output Structure:**
```python
{
    'status': 'SUCCESS',
    'commodity': 'Rice (local)',
    'state': 'Adamawa',
    'year': 2026,
    'data': [
        {'date': '2026-01', 'price': '857.75'},
        {'date': '2026-02', 'price': '857.75'},
        {'date': '2026-03', 'price': '857.75'},
        # ... all 12 months
    ],
    'statistics': {
        'average_forecast': 857.75,
        'max_forecast': 857.75,
        'min_forecast': 857.75,
        'model_type': 'volatile'
    }
}
```

### Function 3: Compare States Forecast

```python
from price_analytics import compare_states_forecast

result = compare_states_forecast(
    commodity='Rice (local)',
    states=['Yobe', 'Adamawa', 'Borno'],
    target_year=2026
)

print(result)
```

**Output Structure:**
```python
{
    'status': 'SUCCESS',
    'commodity': 'Rice (local)',
    'year': 2026,
    'graph_data': [
        {
            'state': 'Yobe',
            'data': [
                {'date': '2026-01', 'price': '1096.44'},
                {'date': '2026-02', 'price': '1096.44'},
                # ... 12 months
            ]
        },
        {
            'state': 'Adamawa',
            'data': [
                {'date': '2026-01', 'price': '857.75'},
                # ... 12 months
            ]
        },
        {
            'state': 'Borno',
            'data': [
                {'date': '2026-01', 'price': '1423.37'},
                # ... 12 months
            ]
        }
    ],
    'analytics': {
        'cheapest_state': {
            'state': 'Adamawa',
            'average_price': 857.75
        },
        'most_expensive_state': {
            'state': 'Borno',
            'average_price': 1423.37
        },
        'best_month_to_buy': {
            'month': 1,
            'month_name': 'Jan',
            'average_price': 1125.85
        },
        'price_difference': 565.62,
        'savings_potential': 39.74
    }
}
```

## Using the Class Interface

For better performance when making multiple calls, use the `PriceAnalytics` class:

```python
from price_analytics import PriceAnalytics

# Initialize once
analytics = PriceAnalytics(
    data_path='./my_food_prices_avg.csv',
    model_dir='saved_models'
)

# Make multiple calls without reloading data/models
result1 = analytics.get_historical_prices('Rice (local)', 'Adamawa', '2024-11', '2025-11')
result2 = analytics.forecast_year('Rice (local)', 'Adamawa', 2026)
result3 = analytics.compare_states_forecast('Rice (local)', ['Yobe', 'Adamawa', 'Borno'], 2026)
```

## Data Structure Requirements

### Input CSV Format
The CSV file should have these columns:
- `date`: Date in YYYY-MM-DD format
- `state`: State name
- `commodity`: Commodity name
- `price`: Price value

### Model Directory Structure
The models should be in the following structure:
```
saved_models/
├── model_metadata.pkl
├── Rice (local)_normal.pkl
├── Rice (local)_volatile.pkl
├── Beans (red)_normal.pkl
└── ... (other commodity models)
```

## Plotting the Multi-Line Graph (Example)

```python
import matplotlib.pyplot as plt
from price_analytics import compare_states_forecast

# Get the comparison data
result = compare_states_forecast(
    commodity='Rice (local)',
    states=['Yobe', 'Adamawa', 'Borno'],
    target_year=2026
)

if result['status'] == 'SUCCESS':
    plt.figure(figsize=(12, 6))

    # Plot each state
    for state_data in result['graph_data']:
        dates = [d['date'] for d in state_data['data']]
        prices = [float(d['price']) for d in state_data['data']]
        plt.plot(dates, prices, marker='o', label=state_data['state'])

    plt.title(f"{result['commodity']} Price Forecast - {result['year']}")
    plt.xlabel('Date (Month)')
    plt.ylabel('Price (NGN)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print analytics
    print(f"\nCheapest State: {result['analytics']['cheapest_state']['state']}")
    print(f"Most Expensive State: {result['analytics']['most_expensive_state']['state']}")
    print(f"Best Month to Buy: {result['analytics']['best_month_to_buy']['month_name']}")
    print(f"Savings Potential: {result['analytics']['savings_potential']}%")
```

## Error Handling

All functions return a status field. Always check it before processing results:

```python
result = get_historical_prices('Rice (local)', 'Adamawa')

if result['status'] == 'SUCCESS':
    # Process the data
    print(result['statistics']['all_time_high'])
else:
    # Handle error
    print(f"Error: {result['message']}")
```

## Notes

1. **Model Requirements**: The forecasting functions require trained models. Ensure you've run `xgb_training_final.py` first to train and save the models.

2. **Data Availability**: Historical price analysis will only return data available in the CSV file for the specified date range.

3. **Forecast Accuracy**: Forecasts are based on XGBoost models trained on historical data. The `model_type` field indicates whether the commodity is classified as 'normal' or 'volatile'.

4. **State Names**: Use exact state names as they appear in your dataset (e.g., 'Adamawa', not 'adamawa').

5. **Commodity Names**: Use exact commodity names as they appear in your dataset (e.g., 'Rice (local)', not 'rice').

## Running the Example

To see the functions in action:

```bash
cd backend
python price_analytics.py
```

This will run demonstrations of all three functions.

## Integration with Frontend

These functions return JSON-compatible dictionaries, making them easy to integrate with REST APIs or web frontends:

```python
# Example Flask integration
from flask import Flask, jsonify
from price_analytics import get_historical_prices, forecast_year, compare_states_forecast

app = Flask(__name__)

@app.route('/api/historical/<commodity>/<state>')
def get_historical(commodity, state):
    result = get_historical_prices(commodity, state)
    return jsonify(result)

@app.route('/api/forecast/<commodity>/<state>/<int:year>')
def get_forecast(commodity, state, year):
    result = forecast_year(commodity, state, year)
    return jsonify(result)

@app.route('/api/compare/<commodity>/<int:year>')
def compare_states(commodity, year):
    states = request.args.getlist('states')
    result = compare_states_forecast(commodity, states, year)
    return jsonify(result)
```

## License

Part of the AgriPredict project.
