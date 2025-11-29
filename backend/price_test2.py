import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
from datetime import datetime, timedelta
import io

warnings.filterwarnings('ignore')

st.set_page_config(page_title="SARIMA Food Price Forecaster", layout="wide")
st.title("🌾 SARIMA Food Price Forecasting - Nigeria")
# st.markdown("*Real-time price prediction with your WFP data*")

# ============================================
# DATA LOADING & PROCESSING
# ============================================

@st.cache_data
def load_and_process_data(file_path=None):
    """Load and process the CSV data"""
    try:
        if file_path:
            df = pd.read_csv(file_path)
        else:
            # Try to load from a local path if user provides it
            st.warning("Please upload your CSV file")
            return None
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # For this analysis, focus on Retail prices only (more accessible data)
        df = df[df['pricetype'].str.lower() == 'retail'].copy()
        
        # Handle unit standardization - focus on KG for consistency
        df = df[df['unit'].str.lower().str.contains('kg', na=False)].copy()
        
        # Remove rows with missing prices
        df = df.dropna(subset=['price'])
        
        # Remove outliers (prices that are 0 or extremely high)
        df = df[df['price'] > 0]
        
        return df.sort_values('date')
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ============================================
# CALCULATE STATE FACTORS FROM REAL DATA
# ============================================

def calculate_state_factors(df, commodity):
    """Calculate state factors for a specific commodity"""
    commodity_data = df[df['commodity'].str.lower() == commodity.lower()]
    
    if len(commodity_data) == 0:
        return None
    
    state_avg = commodity_data.groupby('admin1')['price'].mean()
    overall_mean = commodity_data['price'].mean()
    
    state_factors = state_avg / overall_mean
    
    return state_factors, state_avg, overall_mean

# ============================================
# TIME SERIES PREPARATION
# ============================================

def create_time_series_for_pair(df, state, commodity):
    """Extract time series for state-commodity pair"""
    subset = df[
        (df['admin1'].str.lower() == state.lower()) & 
        (df['commodity'].str.lower() == commodity.lower())
    ].copy().sort_values('date')
    
    if len(subset) < 10:  # Need at least some data
        return None
    
    # Group by date and take average (multiple markets per state)
    ts = subset.set_index('date')['price'].resample('D').mean()
    
    # Forward fill and backward fill missing dates
    ts = ts.fillna(method='ffill').fillna(method='bfill')
    
    return ts[ts > 0]  # Remove any zero values

# ============================================
# PREDICT FOR SPECIFIC MONTH/WEEK
# ============================================

def get_forecast_for_period(fitted_model, target_date, period_type='month'):
    """
    Get forecast for a specific month or week
    
    target_date: datetime object
    period_type: 'month' or 'week'
    """
    
    if period_type == 'month':
        # Get all days in the target month
        if target_date.month == 12:
            end_date = datetime(target_date.year + 1, 1, 1)
        else:
            end_date = datetime(target_date.year, target_date.month + 1, 1)
        
        days_in_month = (end_date - target_date).days
        
        # Forecast from today to target date
        # (This is simplified - ideally you'd retrain with more recent data)
        forecast_steps = (target_date - datetime.now()).days
        
        if forecast_steps < 1:
            st.warning("Target date is in the past")
            return None
        
        forecast_result = fitted_model.get_forecast(steps=forecast_steps + 30)
        forecast_df = forecast_result.conf_int(alpha=0.05)
        forecast_df['forecast'] = forecast_result.predicted_mean
        
        # Filter for the target month
        forecast_df = forecast_df[
            (forecast_df.index.month == target_date.month) | 
            (forecast_df.index.month == target_date.month % 12 + 1)
        ]
        
        return forecast_df
    
    elif period_type == 'week':
        # Similar logic for weeks
        forecast_steps = (target_date - datetime.now()).days + 7
        
        if forecast_steps < 1:
            st.warning("Target week is in the past")
            return None
        
        forecast_result = fitted_model.get_forecast(steps=forecast_steps)
        forecast_df = forecast_result.conf_int(alpha=0.05)
        forecast_df['forecast'] = forecast_result.predicted_mean
        
        return forecast_df

# ============================================
# SIDEBAR SETUP
# ============================================

st.sidebar.header("📤 Upload Data")
# uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type=['csv'])
uploaded_file = 'kdkjs'

if uploaded_file:
    df = load_and_process_data('./my_food_prices_3.csv')
    
    if df is not None:
        st.sidebar.success(f"✅ Loaded {len(df)} records")
        
        # Get available commodities and states
        commodities = sorted(df['commodity'].unique())
        states = sorted(df['admin1'].unique())
        
        st.sidebar.header("⚙️ Configuration")
        
        # Select commodity and state
        col1, col2 = st.sidebar.columns(2)
        with col1:
            selected_commodity = st.selectbox("Commodity:", commodities)
        with col2:
            selected_state = st.selectbox("State:", states)
        
        # Get time series
        ts = create_time_series_for_pair(df, selected_state, selected_commodity)
        
        if ts is None or len(ts) < 10:
            st.error(f"Insufficient data for {selected_commodity} in {selected_state}")
            st.stop()
        
        # ============================================
        # MAIN TABS
        # ============================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔮 Month/Week Forecast", 
            "📈 Full Forecast", 
            "📉 Analysis",
            "💡 State Comparison"
        ])
        
        # TAB 1: OVERVIEW
        with tab1:
            st.subheader(f"{selected_commodity} in {selected_state}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₦{ts.iloc[-1]:,.0f}")
            with col2:
                st.metric("Average Price", f"₦{ts.mean():,.0f}")
            with col3:
                st.metric("Max Price", f"₦{ts.max():,.0f}")
            with col4:
                st.metric("Min Price", f"₦{ts.min():,.0f}")
            
            # Historical price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts.index,
                y=ts.values,
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"Historical Prices: {selected_commodity} in {selected_state}",
                xaxis_title="Date",
                yaxis_title="Price (NGN)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stationarity test
            is_stat, p_val = adfuller(ts.dropna(), autolag='AIC')[1], adfuller(ts.dropna(), autolag='AIC')[1]
            
            if is_stat <= 0.05:
                st.success("✅ Series is stationary")
            else:
                st.info("⚠️ Series is non-stationary (will be differenced)")
        
        # TAB 2: MONTH/WEEK FORECAST
        with tab2:
            st.subheader("🔮 Forecast for Specific Period")
            
            forecast_type = st.radio("Select period type:", ["Month", "Week"])
            
            col1, col2 = st.columns(2)
            with col1:
                target_year = st.number_input("Year:", min_value=2024, max_value=2026, value=2024)
            with col2:
                if forecast_type == "Month":
                    target_month = st.selectbox("Month:", range(1, 13), 
                                               format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])
                    target_date = datetime(target_year, target_month, 1)
                else:
                    target_week = st.number_input("Week of year:", min_value=1, max_value=52)
                    target_date = datetime(target_year, 1, 1) + timedelta(weeks=target_week-1)
            
            if st.button("Generate Forecast"):
                with st.spinner("Training SARIMA model..."):
                    try:
                        # Train SARIMA
                        model = SARIMAX(
                            ts.dropna(),
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 365),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        fitted_model = model.fit(disp=False)
                        
                        # Get forecast for period
                        forecast_steps = 180  # Forecast 6 months ahead
                        forecast_result = fitted_model.get_forecast(steps=forecast_steps)
                        forecast_df = forecast_result.conf_int(alpha=0.05)
                        forecast_df['forecast'] = forecast_result.predicted_mean
                        forecast_df.columns = ['lower', 'upper', 'forecast']
                        
                        # Filter for target period
                        if forecast_type == "Month":
                            period_data = forecast_df[forecast_df.index.month == target_date.month]
                            period_label = f"{['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][target_date.month-1]} {target_year}"
                        else:
                            period_data = forecast_df.iloc[target_week*7:(target_week+1)*7]
                            period_label = f"Week {target_week} of {target_year}"
                        
                        if len(period_data) == 0:
                            st.warning("No forecast data for selected period")
                        else:
                            # Display forecast metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Forecast", f"₦{period_data['forecast'].mean():,.0f}")
                            with col2:
                                st.metric("Min Forecast", f"₦{period_data['forecast'].min():,.0f}")
                            with col3:
                                st.metric("Max Forecast", f"₦{period_data['forecast'].max():,.0f}")
                            
                            # Plot
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=ts[-90:].index,
                                y=ts[-90:].values,
                                mode='lines',
                                name='Historical (Last 90 days)',
                                line=dict(color='blue')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=period_data.index,
                                y=period_data['forecast'],
                                mode='lines+markers',
                                name=f'{period_label} Forecast',
                                line=dict(color='red', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=period_data.index,
                                y=period_data['upper'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=period_data.index,
                                y=period_data['lower'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name='95% Confidence Interval',
                                fillcolor='rgba(255,0,0,0.2)'
                            ))
                            
                            fig.update_layout(
                                title=f"Forecast: {selected_commodity} in {selected_state} - {period_label}",
                                xaxis_title="Date",
                                yaxis_title="Price (NGN)",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Forecast table
                            st.subheader("Forecast Details")
                            display_df = period_data.copy()
                            display_df['forecast'] = display_df['forecast'].apply(lambda x: f"₦{x:,.0f}")
                            display_df['lower'] = display_df['lower'].apply(lambda x: f"₦{max(x, 0):,.0f}")
                            display_df['upper'] = display_df['upper'].apply(lambda x: f"₦{x:,.0f}")
                            display_df.columns = ['Lower Bound', 'Upper Bound', 'Predicted Price']
                            
                            st.dataframe(display_df, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # TAB 3: FULL FORECAST
        with tab3:
            st.subheader("📈 30-Day Full Forecast")
            
            with st.spinner("Training SARIMA model..."):
                try:
                    model = SARIMAX(
                        ts.dropna(),
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 365),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    
                    fitted_model = model.fit(disp=False)
                    forecast_result = fitted_model.get_forecast(steps=30)
                    forecast_df = forecast_result.conf_int(alpha=0.05)
                    forecast_df['forecast'] = forecast_result.predicted_mean
                    forecast_df.columns = ['lower', 'upper', 'forecast']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("AIC", f"{fitted_model.aic:.2f}")
                    with col2:
                        st.metric("BIC", f"{fitted_model.bic:.2f}")
                    with col3:
                        rmse = np.sqrt(np.mean(fitted_model.resid**2))
                        st.metric("RMSE", f"₦{rmse:,.0f}")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=ts[-60:].index,
                        y=ts[-60:].values,
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df['forecast'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df['upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df['lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='95% CI',
                        fillcolor='rgba(255,0,0,0.2)'
                    ))
                    
                    fig.update_layout(
                        title=f"30-Day Forecast: {selected_commodity} in {selected_state}",
                        xaxis_title="Date",
                        yaxis_title="Price (NGN)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # TAB 4: ANALYSIS
        with tab4:
            st.subheader("📉 Time Series Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Std Dev", f"₦{ts.std():,.0f}")
                st.metric("CV (%)", f"{(ts.std()/ts.mean()*100):.2f}%")
            
            with col2:
                st.metric("Skewness", f"{ts.skew():.3f}")
                st.metric("Kurtosis", f"{ts.kurtosis():.3f}")
            
            # Price distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=ts.values,
                nbinsx=40,
                name='Price Distribution'
            ))
            
            fig.update_layout(
                title="Price Distribution",
                xaxis_title="Price (NGN)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 5: STATE COMPARISON
        with tab5:
            st.subheader("💡 State Price Comparison")
            
            factors, state_avg, overall = calculate_state_factors(df, selected_commodity)
            
            if factors is not None:
                st.write(f"**Commodity:** {selected_commodity}")
                st.write(f"**Overall Average Price:** ₦{overall:,.0f}")
                
                comparison_df = pd.DataFrame({
                    'State': state_avg.index,
                    'Average Price': state_avg.values,
                    'Factor': factors.values,
                    'Difference from Average': (state_avg.values - overall)
                })
                
                comparison_df = comparison_df.sort_values('Average Price', ascending=False)
                
                fig = px.bar(
                    comparison_df,
                    x='State',
                    y='Average Price',
                    color='Factor',
                    title=f"{selected_commodity} Price by State",
                    color_continuous_scale='RdYlGn_r',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Please upload your CSV file to get started")
    st.markdown("""
    ### Expected CSV Format:
    - `date`: YYYY-MM-DD
    - `admin1`: State name
    - `commodity`: Product name (Rice, Yam, Millet, etc.)
    - `price`: Price in NGN
    - `pricetype`: Retail or Wholesale
    - `unit`: KG, L, etc.
    """)

st.markdown("---")
st.markdown("*Built with Streamlit, SARIMA, and real WFP data*")