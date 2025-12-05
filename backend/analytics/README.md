# Analytics Module

Clean, modular price analytics functions for AgriPredict.

## Structure

```
analytics/
├── __init__.py              # Main exports
├── historical_prices.py     # Historical price analysis with ATH/ATL
├── forecast.py              # Year forecasting using XGBoost
└── state_comparison.py      # Multi-state comparison with analytics
```

## Quick Start

```python
from analytics import get_historical_prices, forecast_year, compare_states_forecast

# 1. Get historical prices
result = get_historical_prices('Rice (local)', 'Adamawa', '2024-11', '2025-11')

# 2. Forecast a year
result = forecast_year('Rice (local)', 'Adamawa', 2026)

# 3. Compare states
result = compare_states_forecast('Rice (local)', ['Yobe', 'Adamawa', 'Borno'], 2026)
```

## Run Examples

```bash
cd backend
python examples.py
```

## Module Details

### historical_prices.py
- **Function:** `get_historical_prices(commodity, state, start_date, end_date)`
- **Returns:** Historical data + statistics (ATH, ATL, average, current, volatility)

### forecast.py
- **Function:** `forecast_year(commodity, state, target_year)`
- **Returns:** 12 months of forecasted prices + statistics
- **Uses:** XGBoost models from `saved_models/`

### state_comparison.py
- **Function:** `compare_states_forecast(commodity, states, target_year)`
- **Returns:** Multi-line graph data + analytics
- **Analytics:** Cheapest state, most expensive state, best month to buy

## Data Format

All functions return consistent JSON-like dictionaries with:
- `status`: 'SUCCESS' or 'ERROR'
- `message`: Error message (if error)
- `data`: Main data array
- `statistics` or `analytics`: Computed metrics
