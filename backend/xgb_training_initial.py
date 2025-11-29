import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib


df = pd.read_csv('./my_food_prices_avg.csv')

df["date"] = pd.to_datetime(df["date"])
# df.index = pd.to_timedelta(df.index)

fig, ax = plt.subplots(figsize=(14,6))

def plot_across_states(df, product, states):
    plt.figure(figsize=(14,6))
    # df.plot(ax=ax, label="")
    for st in states:
        subset = df[(df["state"] == st) & (df["commodity"] == product)].sort_values("date")
        plt.plot(subset["date"], subset["price"], label=st)

    plt.title(f"Price of {product} Across States")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_state_product_trends(df, state, products, date_col="date"):

    # Filter
    data = df[
        (df["state"] == state) &
        (df["commodity"].isin(products))
    ]

    # Sort by date
    data = data.sort_values(date_col)

    # Create the plot
    plt.figure(figsize=(12, 6))

    for p in products:
        subset = data[data["commodity"] == p]
        plt.plot(
            subset[date_col],
            subset["price"],
            label=p,
            linewidth=2
        )

    plt.title(f"Price Trend for Selected Commodities in {state}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

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


features = ["price_lag_1", "price_lag_3", "price_lag_6", "price_lag_12",
            "rolling_3", "rolling_6", "rolling_12"]
target = 'price'

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
            "price_to_median_ratio", "price_momentum"]


train_df = df[df["date"].dt.year.between(2020, 2023)]
val_df   = df[df["date"].dt.year == 2024]
test_df  = df[df["date"].dt.year == 2025]


x_train = train_df[features]
y_train = train_df[target]

x_val = val_df[features]
y_val = val_df[target]

x_test = test_df[features]
y_test = test_df[target]

features = ["price_lag_1", "price_lag_3", "price_lag_6", "price_lag_12",
            "rolling_3", "rolling_6", "rolling_12",
            "state_avg_price_lag1", "commodity_avg_price_lag1",
            "price_to_median_ratio", "price_momentum"]

target = 'price'

normal_commodities = ['Beans (red)', 'Oranges', 'Tomatoes']
volatile_commodities = ['Rice (local)', 'Yam']

predictions_list = []

# Process normal commodities
for commodity in normal_commodities:
    train_commodity = df[(df["date"].dt.year.between(2020, 2024)) & (df["commodity"] == commodity)]
    test_commodity = df[(df["date"].dt.year == 2025) & (df["commodity"] == commodity)].copy()
    
    if len(train_commodity) > 20 and len(test_commodity) > 0:
        x_train_c = train_commodity[features]
        y_train_c = train_commodity[target]
        x_test_c = test_commodity[features]
        
        model_c = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
        model_c.fit(x_train_c, y_train_c, verbose=0)
        
        y_pred_c = model_c.predict(x_test_c)
        
        # Add predictions to test data
        test_commodity['prediction'] = y_pred_c
        predictions_list.append(test_commodity)

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
        x_test_c = test_commodity[features]
        
        model_c = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01, 
                              subsample=0.7, colsample_bytree=0.7, random_state=42)
        model_c.fit(x_train_c, y_train_c, verbose=0)
        
        # Get predictions
        y_pred_test = model_c.predict(x_test_c)
        y_pred_val = model_c.predict(x_val_c)
        
        # Calibration
        pred_std_val = np.std(y_pred_val)
        actual_std_val = np.std(y_val_c)
        variance_scale = actual_std_val / pred_std_val if pred_std_val > 0 else 1.0
        mean_diff = np.mean(y_val_c) - np.mean(y_pred_val)
        
        # Apply calibration
        test_mean = np.mean(y_pred_test)
        y_pred_calibrated = (y_pred_test - test_mean) * variance_scale + test_mean + mean_diff
        
        # Add predictions to test data
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

print("CSV saved as '2025_price_predictions.csv'")
print(f"\nTotal predictions: {len(output_df)}")
print(f"\nFirst 10 rows:")
print(output_df.head(10))



# # Overall metrics
# overall_mae = mean_absolute_error(all_actuals, all_predictions)
# overall_r2 = r2_score(all_actuals, all_predictions)

# print(f"\n{'='*50}")
# print(f"Overall MAE: {overall_mae:.2f}")
# print(f"Overall R²: {overall_r2:.4f}")
# print(f"{'='*50}")

# # Summary dataframe
# results_df = pd.DataFrame(results_summary)
# print("\nResults Summary:")
# print(results_df.to_string(index=False))


# for commodity in df["commodity"].unique():
#     # Filter by commodity
#     train_commodity = df[(df["date"].dt.year.between(2020, 2024)) & (df["commodity"] == commodity)]
#     val_commodity = df[(df["date"].dt.year == 2024) & (df["commodity"] == commodity)]
#     test_commodity = df[(df["date"].dt.year == 2025) & (df["commodity"] == commodity)]
    
#     if len(train_commodity) > 20 and len(test_commodity) > 0:
#         x_train_c = train_commodity[features]
#         y_train_c = train_commodity[target]
#         x_test_c = test_commodity[features]
#         y_test_c = test_commodity[target]
        
#         model_c = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
#         model_c.fit(x_train_c, y_train_c, verbose=0)
        
#         y_pred_c = model_c.predict(x_test_c)
        
#         # Post-process with commodity-specific adjustment
#         shift = price_by_year_commodity[(price_by_year_commodity["commodity"] == commodity) & 
#                                         (price_by_year_commodity["year"] == 2025)]["price"].values
#         if len(shift) > 0:
#             y_pred_c = y_pred_c + (shift[0] - train_commodity["price"].mean()) * 0.5
        
#         all_predictions.extend(y_pred_c)
#         all_actual.extend(y_test_c.values)
        
#         mae = np.abs(y_pred_c - y_test_c.values).mean()
#         print(f"{commodity}: MAE = {mae:.2f}")

# print(f"\nOverall MAE: {np.abs(np.array(all_predictions) - np.array(all_actual)).mean():.2f}")







# # # joblib.dump(model, "xgb_food_price_model.pkl")


# # model = joblib.load("./xgb_food_price_model.pkl")

# test_df['prediction'] = model.predict(x_test)

# test_df_needed = test_df[["date", "state", "commodity", 'price', 'prediction']].copy()

# print(test_df)
# test_df_needed.to_csv('prices_predict1.csv', index=False)


# print(df)



# ==================== MATPLOTLIB VISUALIZATIONS ====================

# sns.set_style("whitegrid")
# plt.rcParams['figure.figsize'] = (15, 10)

# # 1. Line plot: Actual vs Predicted by Commodity (faceted by state)
# commodities = output_df['commodity'].unique()
# states = sorted(output_df['state'].unique())
# num_commodities = len(commodities)

# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# axes = axes.flatten()

# for idx, commodity in enumerate(commodities):
#     ax = axes[idx]
#     comm_data = output_df[output_df['commodity'] == commodity].sort_values('date')
    
#     for state in states:
#         state_data = comm_data[comm_data['state'] == state]
#         ax.plot(state_data['date'], state_data['price'], marker='o', label=f'{state} (Actual)', linewidth=2)
#         ax.plot(state_data['date'], state_data['prediction'], marker='s', linestyle='--', 
#                 label=f'{state} (Predicted)', linewidth=2, alpha=0.7)
    
#     ax.set_title(f'{commodity}', fontsize=12, fontweight='bold')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price (₦)')
#     ax.tick_params(axis='x', rotation=45)
#     ax.legend(fontsize=8, loc='best')
#     ax.grid(True, alpha=0.3)

# # Hide unused subplots
# for idx in range(num_commodities, len(axes)):
#     axes[idx].set_visible(False)

# plt.tight_layout()
# plt.savefig('01_actual_vs_predicted_by_commodity.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 01_actual_vs_predicted_by_commodity.png")
# plt.show()

# # 2. Scatter plot: Actual vs Predicted (colored by commodity)
# plt.figure(figsize=(10, 8))
# commodities_list = output_df['commodity'].unique()
# colors = plt.cm.Set3(np.linspace(0, 1, len(commodities_list)))

# for idx, commodity in enumerate(commodities_list):
#     comm_data = output_df[output_df['commodity'] == commodity]
#     plt.scatter(comm_data['price'], comm_data['prediction'], label=commodity, 
#                s=100, alpha=0.6, color=colors[idx])

# # Add diagonal line for perfect prediction
# max_val = max(output_df['price'].max(), output_df['prediction'].max())
# min_val = min(output_df['price'].min(), output_df['prediction'].min())
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

# plt.xlabel('Actual Price (₦)', fontsize=12)
# plt.ylabel('Predicted Price (₦)', fontsize=12)
# plt.title('Actual vs Predicted Prices - All Data', fontsize=14, fontweight='bold')
# plt.legend(fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('02_actual_vs_predicted_scatter.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 02_actual_vs_predicted_scatter.png")
# plt.show()

# # 3. Bar plot: Average Actual vs Predicted by State-Commodity
# fig_data = output_df.groupby(['state', 'commodity'])[['price', 'prediction']].mean().reset_index()
# fig_data['label'] = fig_data['state'] + ' - ' + fig_data['commodity']

# x = np.arange(len(fig_data))
# width = 0.35

# fig, ax = plt.subplots(figsize=(16, 8))
# ax.bar(x - width/2, fig_data['price'], width, label='Actual', color='skyblue', edgecolor='black')
# ax.bar(x + width/2, fig_data['prediction'], width, label='Predicted', color='lightcoral', edgecolor='black')

# ax.set_xlabel('State - Commodity', fontsize=12, fontweight='bold')
# ax.set_ylabel('Average Price (₦)', fontsize=12, fontweight='bold')
# ax.set_title('Average Actual vs Predicted Prices by State-Commodity', fontsize=14, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(fig_data['label'], rotation=45, ha='right')
# ax.legend(fontsize=11)
# ax.grid(True, alpha=0.3, axis='y')

# plt.tight_layout()
# plt.savefig('03_bar_comparison_by_state_commodity.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 03_bar_comparison_by_state_commodity.png")
# plt.show()

# # 4. Heatmap: Error % by State and Commodity
# pivot_error = output_df.pivot_table(values='error_pct', index='state', columns='commodity', aggfunc='mean')

# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_error, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0, 
#             cbar_kws={'label': 'Error %'}, linewidths=1, linecolor='black')
# plt.title('Prediction Error % Heatmap (State vs Commodity)', fontsize=14, fontweight='bold')
# plt.xlabel('Commodity', fontsize=12)
# plt.ylabel('State', fontsize=12)
# plt.tight_layout()
# plt.savefig('04_error_heatmap.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 04_error_heatmap.png")
# plt.show()

# # 5. Box plot: Error distribution by commodity
# plt.figure(figsize=(12, 6))
# error_data = [output_df[output_df['commodity'] == comm]['error_pct'].values 
#               for comm in output_df['commodity'].unique()]
# plt.boxplot(error_data, labels=output_df['commodity'].unique())
# plt.ylabel('Error (%)', fontsize=12, fontweight='bold')
# plt.xlabel('Commodity', fontsize=12, fontweight='bold')
# plt.title('Prediction Error Distribution by Commodity', fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3, axis='y')
# plt.tight_layout()
# plt.savefig('05_error_distribution.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 05_error_distribution.png")
# plt.show()

# # 6. Time series plot for each state
# for state in states:
#     state_data = output_df[output_df['state'] == state].sort_values('date')
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
#     # Actual prices
#     for commodity in state_data['commodity'].unique():
#         comm_data = state_data[state_data['commodity'] == commodity]
#         ax1.plot(comm_data['date'], comm_data['price'], marker='o', label=commodity, linewidth=2)
#     ax1.set_title(f'{state} - Actual Prices (2025)', fontsize=13, fontweight='bold')
#     ax1.set_ylabel('Price (₦)', fontsize=11)
#     ax1.legend(fontsize=10)
#     ax1.grid(True, alpha=0.3)
#     ax1.tick_params(axis='x', rotation=45)
    
#     # Predicted prices
#     for commodity in state_data['commodity'].unique():
#         comm_data = state_data[state_data['commodity'] == commodity]
#         ax2.plot(comm_data['date'], comm_data['prediction'], marker='s', linestyle='--', 
#                 label=commodity, linewidth=2)
#     ax2.set_title(f'{state} - Predicted Prices (2025)', fontsize=13, fontweight='bold')
#     ax2.set_ylabel('Price (₦)', fontsize=11)
#     ax2.set_xlabel('Date', fontsize=11)
#     ax2.legend(fontsize=10)
#     ax2.grid(True, alpha=0.3)
#     ax2.tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
#     plt.savefig(f'06_timeseries_{state}.png', dpi=300, bbox_inches='tight')
#     print(f"✓ Saved: 06_timeseries_{state}.png")
#     plt.show()