
"""
AgriPredict Analytics - Usage Examples
Run this file to see demonstrations of all analytics functions
"""

from analytics import get_historical_prices, forecast_year, compare_states_forecast


def main():
    print("=" * 80)
    print("AGRIPREDICT PRICE ANALYTICS - DEMONSTRATION")
    print("=" * 80)

    # Example 1: Get historical prices
    print("\n1. HISTORICAL PRICE ANALYSIS")
    print("-" * 80)

    result1 = get_historical_prices(
        commodity='Rice (local)',
        state='Adamawa',
        start_date='2024-11',
        end_date='2025-11'
    )

    if result1['status'] == 'SUCCESS':
        print(f"Commodity: {result1['commodity']}")
        print(f"State: {result1['state']}")
        print(f"\nStatistics:")
        print(f"  All-Time High: NGN{result1['statistics']['all_time_high']['price']} ({result1['statistics']['all_time_high']['date']})")
        print(f"  All-Time Low: NGN{result1['statistics']['all_time_low']['price']} ({result1['statistics']['all_time_low']['date']})")
        print(f"  Average Price: NGN{result1['statistics']['average_price']}")
        print(f"  Current Price: NGN{result1['statistics']['current_price']}")
        print(f"  Volatility: NGN{result1['statistics']['volatility']}")
        print(f"\nData points: {len(result1['data'])}")
    else:
        print(f"Error: {result1['message']}")

    # Example 2: Forecast for a year
    print("\n2. YEARLY FORECAST")
    print("-" * 80)

    result2 = forecast_year(
        commodity='Rice (local)',
        state='Adamawa',
        target_year=2026
    )

    if result2['status'] == 'SUCCESS':
        print(f"Forecast for {result2['year']}")
        print(f"Average Forecast: NGN{result2['statistics']['average_forecast']}")
        print(f"Max Forecast: NGN{result2['statistics']['max_forecast']}")
        print(f"Min Forecast: NGN{result2['statistics']['min_forecast']}")
        print(f"Model Type: {result2['statistics']['model_type']}")
        print(f"\nMonthly forecasts:")
        for item in result2['data'][:6]:  # Show first 6 months
            print(f"  {item['date']}: NGN{item['price']}")
    else:
        print(f"Error: {result2['message']}")

    # Example 3: State comparison
    print("\n3. STATE COMPARISON")
    print("-" * 80)

    result3 = compare_states_forecast(
        commodity='Rice (local)',
        states=['Yobe', 'Adamawa', 'Borno'],
        target_year=2026
    )

    if result3['status'] == 'SUCCESS':
        print(f"Commodity: {result3['commodity']}")
        print(f"Year: {result3['year']}")
        print(f"\nAnalytics:")
        print(f"  Cheapest State: {result3['analytics']['cheapest_state']['state']} (NGN{result3['analytics']['cheapest_state']['average_price']})")
        print(f"  Most Expensive State: {result3['analytics']['most_expensive_state']['state']} (NGN{result3['analytics']['most_expensive_state']['average_price']})")
        print(f"  Best Month to Buy: {result3['analytics']['best_month_to_buy']['month_name']} (NGN{result3['analytics']['best_month_to_buy']['average_price']})")
        print(f"  Price Difference: NGN{result3['analytics']['price_difference']}")
        print(f"  Savings Potential: {result3['analytics']['savings_potential']}%")

        print(f"\nGraph Data Structure (first state):")
        if len(result3['graph_data']) > 0:
            first_state = result3['graph_data'][0]
            print(f"  State: {first_state['state']}")
            print(f"  Data points: {len(first_state['data'])}")
            print(f"  Sample: {first_state['data'][:3]}")
    else:
        print(f"Error: {result3['message']}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
