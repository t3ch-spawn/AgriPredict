import streamlit as st
import pandas as pd
from pmdarima import auto_arima


df = pd.read_csv('./wfp_food_prices_nga.csv')

# Delete the first row, since it's just repeating the heading of each column
df = df.drop(0)

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract the year and filter, from 2018 ending to 2025
df_2 = df[
    ((df['date'].dt.year == 2018) & (df['date'].dt.month >= 10)) | 
    (df['date'].dt.year.between(2020, 2025))
]


df_2025 = df[
    (df['date'].dt.year == 2025) 
]

# df_2025.to_csv('my_food_prices_2025.csv', index=False)
# df_state = df_2025[df_2025["admin1"].str.strip().str.lower() == "adamawa"]
# df_state.to_csv('food_prices_adamawa.csv', index=False)
# top = df_state["commodity"].str.strip().str.lower().value_counts().idxmax()
# count = df_state["commodity"].str.strip().str.lower().value_counts().max()
# print(top, count)

# Checking the unique commodities in 2025 in the dataset
# print(df_2025['commodity'].unique())

# df_filtered is the data set we will work with from now on

# Drop unnecessary columns that do not relate with our project goals
df_2 = df_2.drop(['market_id', 'latitude', 'category', 'commodity_id'] , axis=1)

# rename admin column to state
df_2 = df_2.rename(columns={"admin1": "state"})


# Find the unique values in the commodity column
# unique_commodities = df_2['commodity'].unique().tolist()

# Find the list of states available
# unique_states = df_2['state'].unique().tolist()
# print(sorted(unique_states))

# Get the rows for only Borno State(or other states) and check the commodities they produce
# borno_rows = df_2[df_2['state'] == 'Borno']

# borno_commodities = borno_rows['commodity'].unique().tolist()

# print(sorted(borno_commodities))


product_list = ["Rice (local)", "Yam", "Oranges", "Beans (red)"]

product_state_list = [
    {"state": "Borno", "products": product_list},
    {"state": "Yobe", "products": product_list},
    {"state": "Adamawa", "products": product_list},
    # {"state": "Kano", "products": product_list},
    # {"state": "Katsina", "products": product_list},
    # {"state": "Oyo", "products": product_list},
    # {"state": "Lagos", "products": product_list},
]

# check for the unique dates to see if the dates are constant for the 15th
# print(df_2['date'].unique().tolist())

state_to_products = {item["state"]: item["products"] for item in product_state_list}

# Function to filter df to states and respective products that are in the products_state_list variable
def row_matches(row):
    state = row["state"]
    product = row["commodity"]

    # If the state is not in our allowed list — reject row
    if state not in state_to_products:
        return False
    
    # If the product is allowed for this state — keep row
    return product in state_to_products[state]

# Apply filtering
filtered_df = df_2[df_2.apply(row_matches, axis=1)]
filtered_df = filtered_df.sort_values('date', ascending=True)

# print(filtered_df)

# Check the pricetype values
# print(filtered_df['pricetype'].unique().tolist())

# Filter products to only be retail prodcuts
filtered_df = filtered_df[filtered_df['pricetype'] == 'Retail']

# Checking unique values of markets in a state
# borno_prices = filtered_df[filtered_df['state'] == 'Borno'] 
# print(borno_prices['market'].unique().tolist())

# Changing the price column to numeric in case any values are categorical
filtered_df["price"] = pd.to_numeric(filtered_df["price"], errors="coerce")


# Group by (date, state, commodity, unit), find the average of all market prices to have a single price for a commodity in a state for one month
df_avg = (
    filtered_df.groupby(["date", "state", "commodity", 'unit'], as_index=False)['price']
      .mean()
)

df_avg.to_csv('my_food_prices_avg.csv', index=False)



