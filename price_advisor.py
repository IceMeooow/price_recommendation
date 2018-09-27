import time
from demand_forecaster import DemandForecaster
from price_forecaster import PriceForecaster

class Advisor:

    def __init__(self, path):
        self.path = path

    def recommend_price(self):
        demand_forecaster = DemandForecaster(self.path)
        data_frame = demand_forecaster.create_and_prepare_full_data()

        t = time.time()
        product_history_in_store, forecasted_demand = demand_forecaster.forecast_demand(data_frame)

        price_forecaster = PriceForecaster(product_history_in_store, forecasted_demand)
        summary_dict = price_forecaster.forecast_price()
        print('Forecasting lasted {} seconds '.format(time.time() - t))
        return summary_dict
