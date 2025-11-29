import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SARIMA Food Price Forecaster", layout="wide")
st.title("🌾 SARIMA Food Price Forecasting Dashboard")

# ============================================
# DATA PREPARATION FUNCTIONS
# ============================================

@st.cache_data
def load_sample_data():
    """Create sample dataset for demonstration"""
    dates = pd.date_range('2020-01-01', '2025-01-15', freq='D')
    states = ['Sokoto', 'Kano', 'Lagos', 'Kaduna', 'Katsina']
    commodities = ['Rice', 'Yam', 'Tomato', 'Millet', 'Potato']
    
    data = []
    for state in states:
        for commodity in commodities:
            for date in dates:
                base_price = {
                    'Rice': 500, 'Yam': 800, 'Tomato': 200,
                    'Millet': 450, 'Potato': 350
                }[commodity]
                
                # Add seasonality and trend
                seasonal = 100 * np.sin(2 * np.pi * date.dayofyear / 365)
                trend = (date.year - 2020) * 50
                noise = np.random.normal(0, 30)
                price = base_price + seasonal + trend + noise
                
                # State-specific variation
                state_factors = {'Lagos': 1.1, 'Sokoto': 0.9, 'Kano': 1.0,
                               'Kaduna': 0.95, 'Katsina': 0.92}
                price *= state_factors.get(state, 1.0)
                
                data.append({
                    'Date': date,
                    'State': state,
                    'Commodity': commodity,
                    'Price': max(price, 50)
                })
    
    return pd.DataFrame(data)

def create_time_series(df, state, commodity):
    """Extract and prepare time series for a specific state-commodity pair"""
    subset = df[
        (df['State'] == state) & 
        (df['Commodity'] == commodity)
    ].copy().sort_values('Date')
    
    if len(subset) == 0:
        return None
    
    subset = subset.set_index('Date')['Price'].resample('D').mean()
    subset = subset.fillna(method='ffill').fillna(method='bfill')
    
    return subset

def test_stationarity(timeseries):
    """Test if series is stationary"""
    result = adfuller(timeseries.dropna(), autolag='AIC')
    return result[1] <= 0.05, result[1]

# ============================================
# SIDEBAR SETUP
# ============================================

st.sidebar.header("⚙️ Configuration")

df = load_sample_data()

col1, col2 = st.sidebar.columns(2)
with col1:
    selected_state = st.selectbox("Select State:", df['State'].unique())

with col2:
    selected_commodity = st.selectbox("Select Commodity:", df['Commodity'].unique())

forecast_horizon = st.sidebar.slider("Days to Forecast:", min_value=7, max_value=90, value=30)

confidence_level = st.sidebar.slider("Confidence Level (%):", min_value=80, max_value=99, value=95) / 100

# Get time series
ts = create_time_series(df, selected_state, selected_commodity)

if ts is None:
    st.error("No data available for this selection")
    st.stop()

# ============================================
# MAIN TABS
# ============================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Forecast", "📈 Diagnostics", "📉 Analysis"])

# TAB 1: OVERVIEW
with tab1:
    st.subheader(f"Time Series: {selected_commodity} in {selected_state}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₦{ts.iloc[-1]:,.0f}")
    
    with col2:
        st.metric("Average Price", f"₦{ts.mean():,.0f}")
    
    with col3:
        st.metric("Max Price", f"₦{ts.max():,.0f}")
    
    with col4:
        st.metric("Min Price", f"₦{ts.min():,.0f}")
    
    # Plot time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"{selected_commodity} Price History - {selected_state}",
        xaxis_title="Date",
        yaxis_title="Price (NGN)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stationarity test
    is_stationary, p_value = test_stationarity(ts)
    
    st.subheader("Stationarity Test (Augmented Dickey-Fuller)")
    if is_stationary:
        st.success(f"✅ Series is STATIONARY (p-value: {p_value:.4f})")
    else:
        st.warning(f"⚠️ Series is NON-STATIONARY (p-value: {p_value:.4f})")
        st.info("The data will be differenced during SARIMA modeling")

# TAB 2: FORECAST
with tab2:
    st.subheader("🔮 SARIMA Price Forecast")
    
    with st.spinner("Training SARIMA model..."):
        try:
            # Fit SARIMA model
            # Parameters: (p,d,q)(P,D,Q,s)
            # These are reasonable defaults; you can tune them based on your data
            model = SARIMAX(
                ts.dropna(),
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 365),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
            forecast_df = forecast_result.conf_int(alpha=1-confidence_level)
            forecast_df['forecast'] = forecast_result.predicted_mean
            forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
            
            st.success("✅ Model trained successfully!")
            
            # Display model metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AIC", f"{fitted_model.aic:.2f}")
            with col2:
                st.metric("BIC", f"{fitted_model.bic:.2f}")
            with col3:
                st.metric("RMSE", f"₦{np.sqrt(np.mean(fitted_model.resid**2)):,.0f}")
            
            # Plot forecast with confidence interval
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=ts.index,
                y=ts.values,
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['upper_ci'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['lower_ci'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name=f'{int(confidence_level*100)}% Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title=f"SARIMA Forecast: {selected_commodity} in {selected_state}",
                xaxis_title="Date",
                yaxis_title="Price (NGN)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast table
            st.subheader("Forecast Details")
            
            forecast_table = forecast_df.copy()
            forecast_table['forecast'] = forecast_table['forecast'].apply(lambda x: f"₦{x:,.0f}")
            forecast_table['lower_ci'] = forecast_table['lower_ci'].apply(lambda x: f"₦{max(x, 0):,.0f}")
            forecast_table['upper_ci'] = forecast_table['upper_ci'].apply(lambda x: f"₦{x:,.0f}")
            forecast_table.columns = ['Lower Bound', 'Upper Bound', 'Predicted Price']
            
            st.dataframe(forecast_table, use_container_width=True)
            
            # Download forecast
            csv = forecast_df.to_csv()
            st.download_button(
                label="📥 Download Forecast",
                data=csv,
                file_name=f"{selected_commodity}_{selected_state}_forecast.csv"
            )
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.info("This may occur with limited or irregular data. Try a different commodity-state pair.")

# TAB 3: DIAGNOSTICS
with tab3:
    st.subheader("📈 Time Series Diagnostics")
    
    # Seasonality decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if len(ts) >= 730:  # Need at least 2 years for yearly decomposition
        with st.spinner("Analyzing seasonality..."):
            try:
                decomposition = seasonal_decompose(ts, model='additive', period=365)
                
                fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                
                ts.plot(ax=axes[0], title='Original Series')
                axes[0].set_ylabel('Price')
                
                decomposition.trend.plot(ax=axes[1], title='Trend')
                axes[1].set_ylabel('Trend')
                
                decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
                axes[2].set_ylabel('Seasonal')
                
                decomposition.resid.plot(ax=axes[3], title='Residuals')
                axes[3].set_ylabel('Residual')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Could not decompose series: {str(e)}")
    else:
        st.info("Need at least 2 years of data for seasonality decomposition")
    
    # ACF and PACF plots
    st.subheader("Autocorrelation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acf, ax = plt.subplots(figsize=(10, 4))
        plot_acf(ts.dropna(), lags=40, ax=ax)
        ax.set_title('Autocorrelation Function (ACF)')
        st.pyplot(fig_acf)
    
    with col2:
        fig_pacf, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(ts.dropna(), lags=40, ax=ax)
        ax.set_title('Partial Autocorrelation Function (PACF)')
        st.pyplot(fig_pacf)

# TAB 4: ANALYSIS
with tab4:
    st.subheader("📉 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Standard Deviation", f"₦{ts.std():,.0f}")
        st.metric("Coefficient of Variation", f"{(ts.std()/ts.mean()*100):.2f}%")
    
    with col2:
        st.metric("Skewness", f"{ts.skew():.3f}")
        st.metric("Kurtosis", f"{ts.kurtosis():.3f}")
    
    # Monthly statistics
    st.subheader("Monthly Statistics")
    
    ts_monthly = ts.resample('M').agg(['mean', 'min', 'max', 'std'])
    ts_monthly.columns = ['Average', 'Min', 'Max', 'Std Dev']
    ts_monthly = ts_monthly.applymap(lambda x: f"₦{x:,.0f}")
    
    st.dataframe(ts_monthly, use_container_width=True)
    
    # Price distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ts.values,
        nbinsx=50,
        name='Price Distribution',
        marker_color='blue'
    ))
    
    fig.update_layout(
        title="Price Distribution",
        xaxis_title="Price (NGN)",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Dashboard built with Streamlit, SARIMA, and Plotly for accurate food price forecasting*")