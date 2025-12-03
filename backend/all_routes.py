from fastapi import APIRouter,Request   
from historical_prices import get_historical_prices
from xgb_training_final import predict_future_price, pair_statistics, predict_future_year
from state_comparison import compare_states_forecast

router = APIRouter()

@router.get("/historical_prices/{commodity}/{state}")
def fetch_historical_prices(state, commodity):
    return get_historical_prices(commodity=commodity,
    state= state,)

@router.get("/forecast_month/{commodity}/{state}/{year}/{month}")
def fetch_forecast_month(commodity, state, year, month):
    return  predict_future_price(commodity, state, int(year), int(month))

@router.get("/forecast_year/{commodity}/{state}/{year}")
def fetch_forecast_year(commodity, state, year):
    return  predict_future_year(commodity, state, int(year) )

@router.get("/analytics/{state}/{commodity}")
def fetch_analytics(state, commodity):
    return  pair_statistics(state, commodity)

@router.get("/compare/{year}/{commodity}")
def fetch_comparison(year, commodity):
    return  compare_states_forecast(commodity, ["Adamawa", "Borno", "Yobe"], int(year))


