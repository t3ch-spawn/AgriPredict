import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime


df = pd.read_csv('./my_food_prices_avg.csv')

df["date"] = pd.to_datetime(df["date"])
# df.index = pd.to_timedelta(df.index)

fig, ax = plt.subplots(figsize=(14,6))


# plot_state_product_trends(
#     df=df,
#     state="Oyo",
#     products=["Rice (local)", "Yam", "Bread"]
# )

# plot_across_states(df, "Rice (local)", ["Borno", "Kano", "Yobe"])



df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lag features
df["price_lag_1"] = df.groupby(["state", "commodity"])["price"].shift(1)
df["price_lag_3"] = df.groupby(["state", "commodity"])["price"].shift(3)
df["price_lag_6"] = df.groupby(["state", "commodity"])["price"].shift(6)
df["price_lag_12"] = df.groupby(["state", "commodity"])["price"].shift(12)
df["rolling_3"] = df.groupby(["state", "commodity"])["price"].shift(1).rolling(3).mean()
df["rolling_6"] = df.groupby(["state", "commodity"])["price"].shift(1).rolling(6).mean()
df["rolling_12"] = df.groupby(["state", "commodity"])["price"].shift(1).rolling(12).mean()

# df = pd.get_dummies(df, columns=["state", "commodity"], drop_first=True)

enc_state = LabelEncoder()
enc_product = LabelEncoder()

df["state_enc"] = enc_state.fit_transform(df["state"])
df["commodity_enc"] = enc_product.fit_transform(df["commodity"])


# Drop missing lag rows
df = df.dropna()



# State-specific lag (average price for that state across all commodities)
df["state_avg_price_lag1"] = df.groupby("state")["price"].shift(1)

# Commodity-specific lag (average price for that commodity across all states)
df["commodity_avg_price_lag1"] = df.groupby("commodity")["price"].shift(1)

# Ratio: this commodity's price vs its typical price
df["commodity_median_price"] = df.groupby("commodity")["price"].transform("median")
df["price_to_median_ratio"] = df["price"] / df["commodity_median_price"]

# Price momentum (how fast prices are changing)
df["price_momentum"] = df.groupby(["state", "commodity"])["price"].shift(1) - df.groupby(["state", "commodity"])["price"].shift(3)

features = ["price_lag_1", "price_lag_3", "price_lag_6", "price_lag_12",
            "rolling_3", "rolling_6", "rolling_12",
            "state_avg_price_lag1", "commodity_avg_price_lag1",
            "price_to_median_ratio", "price_momentum", "month_sin","month_cos"]

target = 'price'

train_df = df[df["date"].dt.year.between(2020, 2023)]
val_df   = df[df["date"].dt.year == 2024]
test_df  = df[df["date"].dt.year == 2025]


x_train = train_df[features]
y_train = train_df[target]

x_val = val_df[features]
y_val = val_df[target]

x_test = test_df[features]
y_test = test_df[target]



# ==================== STEP 1: AUTO-CLASSIFY COMMODITIES ====================
"""
COMMODITY VOLATILITY CLASSIFICATION LOGIC:
A commodity is classified as VOLATILE if it exhibits any of these characteristics:
1. High volatility variance: test_std differs from train_std by >30%
   - Indicates price swings are unpredictable and change between periods
2. Large price shift: year-over-year price change >15%
   - Suggests structural market changes affecting prices
3. High coefficient of variation: test_cv > 0.25 (std/mean)
   - Prices fluctuate wildly relative to their average value
4. Poor model fit: R² < 0.5 on validation data
   - Model struggles to learn the price patterns

NORMAL commodities have stable, predictable patterns where:
- Volatility remains consistent between training and test periods
- Prices don't shift dramatically year-over-year
- Price variations are modest relative to average price
- Model achieves good fit (R² >= 0.5)
"""

# print("="*80)
# print("STEP 1: COMMODITY VOLATILITY CLASSIFICATION")
# print("="*80)

all_commodities = df[df["date"].dt.year == 2025]['commodity'].unique()
normal_commodities = []
volatile_commodities = []
classification_results = []

for commodity in all_commodities:
    train_commodity = df[(df["date"].dt.year.between(2020, 2024)) & (df["commodity"] == commodity)]
    val_commodity = df[(df["date"].dt.year == 2024) & (df["commodity"] == commodity)]
    test_commodity = df[(df["date"].dt.year == 2025) & (df["commodity"] == commodity)]
    
    # Skip if insufficient data
    if len(train_commodity) < 20:
        print(f"\n❌ {commodity}: SKIPPED (only {len(train_commodity)} training points)")
        continue
    
    if len(test_commodity) == 0:
        print(f"\n❌ {commodity}: SKIPPED (no 2025 test data)")
        continue
    
    # Calculate volatility metrics
    train_std = train_commodity["price"].std()
    test_std = test_commodity["price"].std()
    train_mean = train_commodity["price"].mean()
    test_mean = test_commodity["price"].mean()
    
    # Volatility variance: how much std dev changes
    volatility_change = abs(test_std - train_std) / train_std * 100 if train_std > 0 else 0
    
    # Price level shift: year-over-year percentage change
    price_shift = ((test_mean - train_mean) / train_mean * 100) if train_mean > 0 else 0
    
    # Coefficient of variation: measures relative variability (std/mean)
    # Higher CV = more volatile prices relative to average
    test_cv = test_std / test_mean if test_mean > 0 else 0
    
    # Test model performance on validation data
    x_train_c = train_commodity[features]
    y_train_c = train_commodity[target]
    x_val_c = val_commodity[features]
    y_val_c = val_commodity[target]
    
    try:
        model_test = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        model_test.fit(x_train_c, y_train_c, verbose=0)
        
        if len(x_val_c) > 0:
            y_pred_val = model_test.predict(x_val_c)
            r2_val = r2_score(y_val_c, y_pred_val)
        else:
            r2_val = 0.5
    except:
        r2_val = 0.5
    
    # Classification decision
    is_volatile = False
    reasons = []
    
    # Rule 1: Volatility variance >30%
    if volatility_change > 30:
        is_volatile = True
        reasons.append(f"Volatility variance: {volatility_change:.1f}% (threshold: 30%)")
    
    # Rule 2: Price shift >15%
    if abs(price_shift) > 15:
        is_volatile = True
        reasons.append(f"Price shift: {price_shift:.1f}% (threshold: 15%)")
    
    # Rule 3: Coefficient of variation >0.25
    if test_cv > 0.25:
        is_volatile = True
        reasons.append(f"Coeff. of variation: {test_cv:.3f} (threshold: 0.25)")
    
    # Rule 4: Poor model fit (R² < 0.5)
    if r2_val < 0.5:
        is_volatile = True
        reasons.append(f"Model R²: {r2_val:.3f} (threshold: 0.5)")
    
    # Store results
    if is_volatile:
        volatile_commodities.append(commodity)
        status = "🔴 VOLATILE"
    else:
        normal_commodities.append(commodity)
        status = "🟢 NORMAL"
    
    classification_results.append({
        'commodity': commodity,
        'classification': 'VOLATILE' if is_volatile else 'NORMAL',
        'volatility_change_%': volatility_change,
        'price_shift_%': price_shift,
        'coefficient_variation': test_cv,
        'model_r2': r2_val,
        'reasons': '; '.join(reasons) if reasons else 'Stable commodity'
    })
    
#     print(f"\n{status} {commodity}")
#     print(f"  Volatility Change: {volatility_change:.1f}% (train_std={train_std:.0f} → test_std={test_std:.0f})")
#     print(f"  Price Shift: {price_shift:.1f}% (₦{train_mean:.0f} → ₦{test_mean:.0f})")
#     print(f"  Coefficient of Variation: {test_cv:.3f} (measures relative price fluctuation)")
#     print(f"  Model R² on validation: {r2_val:.4f}")
#     if reasons:
#         print(f"  Classification Reasons: {'; '.join(reasons)}")

# print(f"\n{'='*80}")
# print(f"Classification Summary:")
# print(f"  🟢 NORMAL: {normal_commodities}")
# print(f"  🔴 VOLATILE: {volatile_commodities}")
# print(f"{'='*80}\n")

# ==================== STEP 2: TRAIN AND SAVE MODELS ====================
# print("="*80)
# print("STEP 2: TRAINING AND SAVING MODELS")
# print("="*80)

model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

trained_models = {}
model_metadata = {}

# Process normal commodities
for commodity in normal_commodities:
    train_commodity = df[(df["date"].dt.year.between(2020, 2024)) & (df["commodity"] == commodity)]
    test_commodity = df[(df["date"].dt.year == 2025) & (df["commodity"] == commodity)].copy()
    
    if len(train_commodity) > 20 and len(test_commodity) > 0:
        x_train_c = train_commodity[features]
        y_train_c = train_commodity[target]
        
        model_c = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
        model_c.fit(x_train_c, y_train_c, verbose=0)
        
        # Save model
        model_path = os.path.join(model_dir, f'{commodity}_normal.pkl')
        joblib.dump(model_c, model_path)
        trained_models[commodity] = model_c
        
        model_metadata[commodity] = {
            'type': 'normal',
            'saved_date': datetime.now().isoformat(),
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.05
        }
        
        # print(f"✓ Trained & saved: {commodity} (NORMAL model)")

# Process volatile commodities with calibration
for commodity in volatile_commodities:
    train_commodity = df[(df["date"].dt.year.between(2020, 2024)) & (df["commodity"] == commodity)]
    val_commodity = df[(df["date"].dt.year == 2024) & (df["commodity"] == commodity)]
    test_commodity = df[(df["date"].dt.year == 2025) & (df["commodity"] == commodity)].copy()
    
    if len(train_commodity) > 20 and len(test_commodity) > 0:
        x_train_c = train_commodity[features]
        y_train_c = train_commodity[target]
        x_val_c = val_commodity[features]
        y_val_c = val_commodity[target]
        
        model_c = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01, 
                              subsample=0.7, colsample_bytree=0.7, random_state=42)
        model_c.fit(x_train_c, y_train_c, verbose=0)
        
        # Calculate calibration parameters
        y_pred_val = model_c.predict(x_val_c)
        pred_std_val = np.std(y_pred_val)
        actual_std_val = np.std(y_val_c)
        variance_scale = actual_std_val / pred_std_val if pred_std_val > 0 else 1.0
        mean_diff = np.mean(y_val_c) - np.mean(y_pred_val)
        
        # Save model
        model_path = os.path.join(model_dir, f'{commodity}_volatile.pkl')
        joblib.dump(model_c, model_path)
        trained_models[commodity] = model_c
        
        model_metadata[commodity] = {
            'type': 'volatile',
            'saved_date': datetime.now().isoformat(),
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'variance_scale': float(variance_scale),
            'mean_diff': float(mean_diff)
        }
        
        # print(f"✓ Trained & saved: {commodity} (VOLATILE model with calibration)")

# Save metadata
metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
joblib.dump(model_metadata, metadata_path)
print(f"\n✓ All models saved to '{model_dir}/' directory")
print(f"{'='*80}\n")

# ==================== STEP 3: GENERATE 2025 PREDICTIONS ====================
# print("="*80)
# print("STEP 3: GENERATING 2025 PREDICTIONS")
# print("="*80)

predictions_list = []

# Process normal commodities
for commodity in normal_commodities:
    train_commodity = df[(df["date"].dt.year.between(2020, 2024)) & (df["commodity"] == commodity)]
    test_commodity = df[(df["date"].dt.year == 2025) & (df["commodity"] == commodity)].copy()
    
    if len(train_commodity) > 20 and len(test_commodity) > 0:
        x_test_c = test_commodity[features]
        
        model_c = trained_models[commodity]
        y_pred_c = model_c.predict(x_test_c)
        
        test_commodity['prediction'] = y_pred_c
        predictions_list.append(test_commodity)

# Process volatile commodities with calibration
for commodity in volatile_commodities:
    val_commodity = df[(df["date"].dt.year == 2024) & (df["commodity"] == commodity)]
    test_commodity = df[(df["date"].dt.year == 2025) & (df["commodity"] == commodity)].copy()
    
    if len(test_commodity) > 0:
        x_test_c = test_commodity[features]
        
        model_c = trained_models[commodity]
        y_pred_test = model_c.predict(x_test_c)
        
        # Apply calibration
        variance_scale = model_metadata[commodity]['variance_scale']
        mean_diff = model_metadata[commodity]['mean_diff']
        test_mean = np.mean(y_pred_test)
        y_pred_calibrated = (y_pred_test - test_mean) * variance_scale + test_mean + mean_diff
        
        test_commodity['prediction'] = y_pred_calibrated
        predictions_list.append(test_commodity)

# Combine all predictions
results_df = pd.concat(predictions_list, ignore_index=True)

# Select and order columns
output_df = results_df[['date', 'state', 'commodity', 'price', 'prediction']].copy()

# Sort by date, state, and commodity
output_df = output_df.sort_values(['date', 'state', 'commodity']).reset_index(drop=True)

# Round to 4 decimal places
output_df['price'] = output_df['price'].round(4)
output_df['prediction'] = output_df['prediction'].round(4)

# Calculate error metrics
output_df['error'] = output_df['prediction'] - output_df['price']
output_df['error_pct'] = (output_df['error'] / output_df['price'] * 100).round(2)

# Save to CSV
output_df.to_csv('2025_price_predictions.csv', index=False)

# print(f"✓ CSV saved as '2025_price_predictions.csv'")
# print(f"Total predictions: {len(output_df)}\n")
# print(f"First 10 rows:")
# print(output_df.head(10))


# ==================== STEP 4: FUTURE PREDICTION FUNCTION ====================

def _month_sin_cos(month):
    # month: 1-12
    sin = np.sin(2 * np.pi * month / 12)
    cos = np.cos(2 * np.pi * month / 12)
    return sin, cos

def _get_recent_series_for_state_commodity(df, state, commodity, n=12):
    subset = df[(df['state'] == state) & (df['commodity'] == commodity)].sort_values('date')
    # Return last n prices (may be shorter than n)
    prices = subset['price'].values
    return prices[-n:] if len(prices) > 0 else np.array([])

def _build_feature_vector_from_history(prices_history, df, state, commodity, target_month):
    """
    prices_history: 1D array of most recent prices for (state,commodity) in chronological order.
                    e.g. [price_t-11, ..., price_t-1, price_t]  (last element is the most recent observed)
                    When forecasting recursively, prices_history will include previous predictions.
    df: original dataframe for computing commodity/state aggregates if needed.
    target_month: int 1..12
    """
    # ensure we have enough length for indexing; when missing, use most recent available
    if len(prices_history) == 0:
        recent_price = df[(df['state']==state)&(df['commodity']==commodity)]['price'].iloc[-1]
        prices_history = np.array([recent_price])
    else:
        recent_price = prices_history[-1]

    # lags: use prices_history; if not enough elements, reuse recent_price
    def get_lag(k):
        if len(prices_history) >= k+1:
            return prices_history[-k]   # -1 is t, -2 is t-1, so mapping: price_lag_1 -> prices_history[-1]
        else:
            return recent_price

    price_lag_1 = get_lag(1)
    price_lag_3 = get_lag(3)
    price_lag_6 = get_lag(6)
    price_lag_12 = get_lag(12)

    # rolling windows (use tail of history; require at least something)
    rolling_3 = prices_history[-3:].mean() if len(prices_history) >= 3 else prices_history.mean()
    rolling_6 = prices_history[-6:].mean() if len(prices_history) >= 6 else prices_history.mean()
    rolling_12 = prices_history[-12:].mean() if len(prices_history) >= 12 else prices_history.mean()

    # state and commodity aggregates — use the most recent available value (lag-1 style)
    state_prices = df[df['state'] == state].sort_values('date')['price'].values
    commodity_prices = df[df['commodity'] == commodity].sort_values('date')['price'].values

    state_avg_price_lag1 = state_prices[-1] if len(state_prices) > 0 else recent_price
    commodity_avg_price_lag1 = commodity_prices[-1] if len(commodity_prices) > 0 else recent_price

    # median-based ratio (as you used before)
    commodity_median_price = np.median(commodity_prices) if len(commodity_prices) > 0 else recent_price
    price_to_median_ratio = recent_price / commodity_median_price if commodity_median_price > 0 else 1.0

    # momentum: t - t-6 (if available)
    if len(prices_history) >= 6:
        price_momentum = prices_history[-1] - prices_history[-6]
    else:
        price_momentum = prices_history[-1] - prices_history[0] if len(prices_history) > 1 else 0.0

    # month seasonality
    month_sin, month_cos = _month_sin_cos(target_month)

    feature_values = [
        price_lag_1, price_lag_3, price_lag_6, price_lag_12,
        rolling_3, rolling_6, rolling_12,
        state_avg_price_lag1, commodity_avg_price_lag1,
        price_to_median_ratio, price_momentum,
        month_sin, month_cos
    ]

    return np.array(feature_values, dtype=float)

def predict_future_price(commodity, state, year, month, use_calibration=True):
    """
    Predict a single (year,month) for commodity,state.
    For future years beyond current data, uses recursive prediction.
    """
    if commodity not in trained_models:
        return {'status': 'ERROR', 'message': f'Model for {commodity} not found. Available: {list(trained_models.keys())}'}
    
    # Get the most recent date in historical data
    recent_data = df[(df['state'] == state) & (df['commodity'] == commodity)].sort_values('date')
    if len(recent_data) == 0:
        return {'status': 'ERROR', 'message': f'No historical data found for {commodity} in {state}'}
    
    last_historical_date = recent_data['date'].max()
    last_year = last_historical_date.year
    last_month = last_historical_date.month
    
    # Calculate months from last historical date to target date
    target_date = pd.Timestamp(year=year, month=month, day=1)
    months_ahead = (target_date.year - last_year) * 12 + (target_date.month - last_month)
    
    # If predicting within historical range or just 1-2 months ahead, use direct prediction
    if months_ahead <= 2:
        recent_prices = _get_recent_series_for_state_commodity(df, state, commodity, n=12)
        X_future = _build_feature_vector_from_history(recent_prices, df, state, commodity, month).reshape(1, -1)
        
        model = trained_models[commodity]
        pred = float(model.predict(X_future)[0])
        
        metadata = model_metadata.get(commodity, {'type':'normal'})
        if use_calibration and metadata.get('type') == 'volatile':
            mean_diff = metadata.get('mean_diff', 0.0)
            pred = pred + mean_diff
        
        return {
            'status': 'SUCCESS',
            'commodity': commodity,
            'state': state,
            'year': year,
            'month': month,
            'predicted_price': round(pred, 2),
            'recent_price': round(float(recent_prices[-1]), 2),
            # 'model_type': metadata.get('type', 'unknown'),
            # 'confidence': 'High' if metadata.get('type') == 'normal' else 'Medium (Volatile commodity)'
        }
    
    # For predictions far in the future, recursively predict month by month
    else:
        print('far ahead')
        # Start with historical prices
        history = _get_recent_series_for_state_commodity(df, state, commodity, n=12).tolist()
        
        # Predict month by month until we reach target date
        current_year = last_year
        current_month = last_month + 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        
        predicted_price = None
        
        while (current_year < year) or (current_year == year and current_month <= month):
            # Build features using current history
            X_future = _build_feature_vector_from_history(np.array(history), df, state, commodity, current_month).reshape(1, -1)
            model = trained_models[commodity]
            pred = float(model.predict(X_future)[0])
            
            # Apply calibration
            metadata = model_metadata.get(commodity, {'type': 'normal'})
            if use_calibration and metadata.get('type') == 'volatile':
                mean_diff = metadata.get('mean_diff', 0.0)
                pred = pred + mean_diff
            
            # If this is our target month/year, save it
            if current_year == year and current_month == month:
                predicted_price = pred
            
            # Add prediction to history for next iteration
            history.append(pred)
            if len(history) > 12:
                history = history[-12:]
            
            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        return {
            'status': 'SUCCESS',
            'commodity': commodity,
            'state': state,
            'year': year,
            'month': month,
            'predicted_price': round(predicted_price, 2),
            'recent_price': round(float(_get_recent_series_for_state_commodity(df, state, commodity, n=1)[-1]), 2),
            # 'model_type': metadata.get('type', 'unknown'),
            # 'confidence': 'Low (Far future prediction)' if months_ahead > 12 else 'Medium'
        }

def predict_future_year(commodity, state, year, start_month=1):
    if commodity not in trained_models:
        return [{'status': 'ERROR', 'message': f'Model for {commodity} not found.'}]
    
    # Check if historical data exists
    recent_data = df[(df['state'] == state) & (df['commodity'] == commodity)]
    if len(recent_data) == 0:
        return [{'status': 'ERROR', 'message': f'No historical data found for {commodity} in {state}'}]
    
    results = []
    current_month = start_month
    current_year = year
    
    # Predict 12 consecutive months
    for i in range(12):
        # Use predict_future_price for this month
        prediction = predict_future_price(
            commodity=commodity,
            state=state,
            year=current_year,
            month=current_month,
            use_calibration=True
        )
        
        # Check if prediction was successful
        if prediction['status'] == 'ERROR':
            return [prediction]  # Return error immediately
        
        # Add to results
        results.append({
            'year': current_year,
            'month': current_month,
            'date': f"{current_year}-{current_month:02d}",
            'predicted_price': prediction['predicted_price'],
            # 'confidence': prediction['confidence'],
            # 'model_type': prediction['model_type']
        })
        
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    return results


def pair_statistics(state, commodity):
    """
    Compute statistical and error metrics for a single (state, commodity) pair
    using the already-loaded output_df, which MUST contain:
        - 'state'
        - 'commodity'
        - 'price' (actual)
        - 'prediction' (model prediction)
    """

    # Filter for that pair
    sub = output_df[(output_df["state"] == state) & 
                    (output_df["commodity"] == commodity)].copy()

    if sub.shape[0] == 0:
        print(f"\n❌ No rows found for {commodity} in {state}.\n")
        return {"error": "No data for this pair"}

    # Extract numeric arrays
    actual = sub["price"].astype(float).values
    pred   = sub["prediction"].astype(float).values

    # Basic stats (actual prices)
    n = len(sub)
    mean_price   = float(np.mean(actual))
    median_price = float(np.median(actual))
    std_price    = float(np.std(actual, ddof=0))
    skew_price   = float(pd.Series(actual).skew()) if n > 2 else float("nan")

    # Error metrics
    mae  = float(mean_absolute_error(actual, pred))
    rmse = float(root_mean_squared_error(actual, pred))

    # Mean Absolute Percentage Error (guard division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_vals = np.abs((actual - pred) / actual)
        mape_pct = float(np.nanmean(mape_vals) * 100)

    # Residuals
    resid = pred - actual
    mean_error   = float(np.mean(resid))
    median_error = float(np.median(resid))
    std_error    = float(np.std(resid, ddof=0))
    mad_error    = float(np.median(np.abs(resid - median_error)))

    # MAE relative to the actual mean
    pct_error_mean = (mae / mean_price * 100.0) if mean_price != 0 else float("nan")

    r2 = float(r2_score(actual, pred))

    df_filtered = output_df[
        output_df["state"].str.strip().str.lower() == str(state).strip().lower()
    ]
    df_filtered = df_filtered[
        df_filtered["commodity"].str.strip().str.lower() == str(commodity).strip().lower()
    ].copy()

    if df_filtered.empty:
        return []

    # ensure date column is datetime
    df_filtered["date"] = pd.to_datetime(df_filtered["date"])

    # create a month label for x-axis (e.g., "Jan 2025")
    df_filtered["month_label"] = df_filtered["date"].dt.strftime("%b %Y")
    # also create a sortable month key (YYYY-MM) to sort the rows
    df_filtered["month_key"] = df_filtered["date"].dt.strftime("%Y-%m")

    # if you expect multiple rows per month, aggregate by month_key (take mean)
    agg = (
        df_filtered
        .groupby(["month_key", "month_label"], as_index=False)
        .agg(
            actual_price = ("price", "mean"),
            predicted_price = ("prediction", "mean")
        )
        .sort_values("month_key")
    )

    # convert to list of dicts (suitable to JSONify and send to frontend)
    graphResult = agg[["month_label", "actual_price", "predicted_price"]].rename(
        columns={"month_label": "month"}
    ).to_dict(orient="records")

    # Print nicely
    # print(f"\n📌 Statistics for {commodity} in {state} (N = {n})")
    # print("--------------------------------------------------")
    # print(f"Price statistics:")
    # print(f"  Mean price     : {mean_price:.2f}")
    # print(f"  Median price   : {median_price:.2f}")
    # print(f"  Std deviation  : {std_price:.2f}")
    # print(f"  Skewness       : {skew_price:.3f}\n")

    # print("Error metrics (model performance):")
    # print(f"  MAE            : {mae:.3f}")
    # print(f"  RMSE           : {rmse:.3f}")
    # print(f"  MAPE           : {mape_pct:.2f}%")
    # print(f"  MAE as % mean  : {pct_error_mean:.2f}%\n")

    # print("Residual summary:")
    # print(f"  Mean error     : {mean_error:.3f}")
    # print(f"  Median error   : {median_error:.3f}")
    # print(f"  Std error      : {std_error:.3f}")
    # print(f"  MAD error      : {mad_error:.3f}")
    # print("--------------------------------------------------\n")

    # Return values as dictionary
    return {
        "state": state,
        "commodity": commodity,
        "n": n,
        "mean_price": mean_price,
        "median_price": median_price,
        "std_price": std_price,
        "skew_price": skew_price,
        "mae": mae,
        "rmse": rmse,
        "r_2": r2,
        "mape_pct": mape_pct,
        "pct_error_mean": pct_error_mean,
        "mean_error": mean_error,
        "median_error": median_error,
        "std_error": std_error,
        "mad_error": mad_error,
        "graph": graphResult
    }


# for st in output_df["state"].unique():
#     pair_statistics(output_df, st, "Yam")


# ==================== EXAMPLE USAGE ====================
print("\n" + "="*80)
print("STEP 4: FUTURE PRICE PREDICTION EXAMPLES")
print("="*80 + "\n")

# Example predictions
examples = [
    ('Rice (local)', 'Borno', 2026, 1),
    ('Rice (local)', 'Borno', 2026, 4),
    # ('Rice (local)', 'Borno', 2027, 4),
    # ('Rice (local)', 'Yobe', 2026, 6),
    # ('Yam', 'Borno', 2026, 6),
    # ('Beans (red)', 'Yobe', 2026, 9),
    # ('Oranges', 'Adamawa', 2027, 1),
    # ('Tomatoes', 'Borno', 2027, 11)
]

# for commodity, state, year, month in examples:
#     result = predict_future_price(commodity, state, year, month)
#     print(f"{result['commodity']} in {result['state']} ({year}-{result['month']:02d}):")
#     print(f"  Predicted Price: ₦{result['predicted_price']}")
#     print(f"  Recent Price: ₦{result['recent_price']}")
#     # print(f"  Model Type: {result['model_type']}")
#     # print(f"  Confidence: {result['confidence']}\n")

# print("="*80)
# print("✓ Pipeline complete! Models saved, predictions generated.")
# print("="*80)


sns.set_theme(style="whitegrid")


def plot_error_distribution(output_df,
                            commodity,
                            state,
                            error_type="percent",   # 'percent' or 'absolute'
                            bins=20,
                            figsize=(10, 6),
                            show_plot=True,
                            save_path=None):
    """
    Plot boxplot + histogram of model errors for a given commodity & state (based on output_df).
    Also returns histogram data as an array of JS-friendly objects for React charts.

    Parameters
    ----------
    output_df : pd.DataFrame
      Must contain columns: 'state', 'commodity', 'price' (actual), 'prediction' (predicted),
      optionally 'date' (if present, will filter to year==2025).
    commodity : str
      Commodity name to filter.
    state : str
      State name to filter.
    error_type : str
      'percent' (percent error = (pred-actual)/actual *100) OR 'absolute' (pred - actual).
    bins : int
      Number of histogram bins.
    figsize : tuple
      Figure size for matplotlib.
    show_plot : bool
      If True, calls plt.show(). If False, returns figure & axes without showing.
    save_path : str or None
      If provided, path to save the figure (PNG). No effect on returned hist data.

    Returns
    -------
    result : dict
      {
        "n": int,                    # number of observations used
        "mae": float,
        "rmse": float,
        "errors": np.ndarray,        # raw errors (pred - actual)
        "abs_errors": np.ndarray,    # absolute errors
        "pct_errors": np.ndarray,    # percent errors (in %)
        "histogram_bins": [          # list of JS-friendly bin objects
            {
              "binStart": float,
              "binEnd": float,
              "binCenter": float,
              "count": int,
              "density": float
            }, ...
        ],
        "fig": matplotlib.figure.Figure,  # matplotlib figure (useful for further saving/manipulation)
        "ax_hist": matplotlib.axes.Axes,
        "ax_box": matplotlib.axes.Axes
      }

    Example usage
    -------------
    res = plot_error_distribution(output_df, "Rice (local)", "Borno", error_type='percent', bins=30)
    # use res['histogram_bins'] in React chart
    """

    # Basic column sanity checks
    required_cols = {"state", "commodity", "price", "prediction"}
    missing = required_cols - set(output_df.columns)
    if missing:
        raise ValueError(f"output_df is missing required columns: {missing}")

    df = output_df.copy()

    # If there's a date column, restrict to 2025 rows (the user's request)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"].dt.year == 2025]
        except Exception:
            # if parsing fails, ignore and proceed with entire df
            pass

    # Filter for the requested pair
    sub = df[(df["commodity"] == commodity) & (df["state"] == state)].copy()

    if sub.shape[0] == 0:
        print(f"No rows found for commodity='{commodity}' in state='{state}' (after 2025 filtering).")
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "errors": np.array([]),
            "abs_errors": np.array([]),
            "pct_errors": np.array([]),
            "histogram_bins": [],
            "fig": None,
            "ax_hist": None,
            "ax_box": None
        }

    # Ensure numeric
    sub["price"] = pd.to_numeric(sub["price"], errors="coerce")
    sub["prediction"] = pd.to_numeric(sub["prediction"], errors="coerce")
    sub = sub.dropna(subset=["price"]).reset_index(drop=True)

    # Compute errors
    actual = sub["price"].values.astype(float)
    pred = sub["prediction"].values.astype(float)
    errors = pred - actual
    abs_errors = np.abs(errors)
    # percent errors in percent points (e.g. 10 => 10%)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_errors = np.where(actual != 0, (errors / actual) * 100.0, np.nan)

    # Basic metrics
    n = len(actual)
    mae = float(np.nanmean(abs_errors)) if n > 0 else float("nan")
    rmse = float(np.sqrt(np.nanmean((errors) ** 2))) if n > 0 else float("nan")

    # Select which error to plot/histogram
    if error_type == "absolute":
        plot_values = abs_errors
        xlabel = "Absolute error (₦)"
        hist_label = "Absolute error"
    elif error_type == "percent":
        plot_values = pct_errors
        xlabel = "Percent error (%)"
        hist_label = "Percent error"
    else:
        raise ValueError("error_type must be 'percent' or 'absolute'")

    # Drop NaNs from plot values (percent errors can have NaN if actual==0)
    valid_mask = ~np.isnan(plot_values)
    plot_values_clean = plot_values[valid_mask]

    # Build histogram bins (numpy)
    counts, bin_edges = np.histogram(plot_values_clean, bins=bins, density=False)
    # densities for percent of total
    densities = counts.astype(float) / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Build JS-friendly bin objects
    histogram_bins = []
    for i in range(len(counts)):
        histogram_bins.append({
            "binStart": float(bin_edges[i]),
            "binEnd": float(bin_edges[i+1]),
            "binCenter": float(bin_centers[i]),
            "count": int(counts[i]),
            "density": float(densities[i])
        })

    # PLOTTING
    fig, (ax_box, ax_hist) = plt.subplots(nrows=2, ncols=1, figsize=figsize,
                                          gridspec_kw={"height_ratios": [0.2, 0.8]})
    # Boxplot (small top panel)
    sns.boxplot(x=plot_values_clean, ax=ax_box, orient="h", showfliers=True)
    ax_box.set( xlabel="" )
    ax_box.set_title(f"{commodity} — {state}  |  Boxplot of {hist_label}")

    # Histogram + KDE (bottom panel)
    # Use seaborn histplot for nicer defaults
    sns.histplot(plot_values_clean, bins=bin_edges, kde=True, ax=ax_hist, stat="count")
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"{commodity} — {state}  |  Histogram of {hist_label} (n={n})")

    # Annotate MAE & RMSE on histogram
    text_x = 0.95
    text_y = 0.95
    stats_text = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nN = {n}"
    # place stats at upper-right in axes fraction coordinates
    ax_hist.text(text_x, text_y, stats_text, transform=ax_hist.transAxes,
                 fontsize=10, ha="right", va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    output_dir = "plot_images"
    os.makedirs(output_dir, exist_ok=True)

# If no save_path was provided, auto-generate one
    if not save_path:
        # safe filename using commodity + state + error type
        filename = f"{commodity}_{state}_{error_type}_errors.png".replace(" ", "_")
        save_path = os.path.join(output_dir, filename)

# Save figure
    fig.savefig(save_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # don't keep it open if not showing

    result = {
        "n": int(n),
        "mae": mae,
        "rmse": rmse,
        "errors": errors,
        "abs_errors": abs_errors,
        "pct_errors": pct_errors,
        "histogram_bins": histogram_bins,
        "fig": fig,
        "ax_hist": ax_hist,
        "ax_box": ax_box
    }

    return result



# res = plot_error_distribution(output_df, "Yam", "Borno", error_type="percent", bins=25)