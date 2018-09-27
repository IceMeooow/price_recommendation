import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


class PriceForecaster:
    def __init__(self, product_data, forecasted_demand):
        self.product_data = product_data
        self.forecasted_demand = forecasted_demand

    def forecast_price(self):
        data, demand_description = self._prepare_data()
        data['LEVEL'] = self._find_demand_level(demand_description, data.UNITS)
        forecasted_demand_level = self._find_demand_level(demand_description, self.forecasted_demand)

        samples, labels = self._split_data_on_samples_and_labels(data)
        samples = self._encode_levels(samples)

        X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)
        regrf = self._create_model(X_train, y_train)
        print('Score of RandomForestRegressor {}'.format(regrf.score(X_test, y_test)))

        dataset = self._create_data_for_forecasting_price(X_train.columns, forecasted_demand_level)
        price = regrf.predict(dataset)

        summary_dict = {'Week #{}'.format(datetime.date.today().isocalendar()[1] + 1): {
                                'Price': round(price[0], 2),
                                'Ads': {'Features': ('yes' if dataset.FEATURE[0] == 1 else 'no'),
                                        'Display': ('yes' if dataset.DISPLAY[0] == 1 else 'no'),
                                        'TPR only': ('yes' if dataset.TPR_ONLY[0] == 1 else 'no')
                                        }
                                },
                    'Week #{}'.format(datetime.date.today().isocalendar()[1] + 2): {
                                'Price': round(price[1], 2),
                                'Ads': {'Features': ('yes' if dataset.FEATURE[1] == 1 else 'no'),
                                        'Display': ('yes' if dataset.DISPLAY[1] == 1 else 'no'),
                                        'TPR only': ('yes' if dataset.TPR_ONLY[1] == 1 else 'no')
                                        }
                                },
                    'Mean price': round(price.mean(), 2)
                    }

        return summary_dict

    def _prepare_data(self):
        data = self.product_data[['UNITS', 'PRICE', 'FEATURE', 'DISPLAY', 'TPR_ONLY']]
        demand_description = data.UNITS.describe()
        return data, demand_description

    def _find_demand_level(self, demand_description, series):
        if type(series) == np.ndarray:
            series = series.flatten()

        levels = []
        for item in series:
            if item <= demand_description[demand_description.index == '25%'][0]:
                levels.append('low')
            elif (item > demand_description[demand_description.index == '25%'][0]) and \
                    (item <= demand_description[demand_description.index == '75%'][0]):
                levels.append('medium')
            elif item > demand_description[demand_description.index == '75%'][0]:
                levels.append('high')
        return levels

    def _split_data_on_samples_and_labels(self, data):
        labels = data.PRICE
        samples = data.drop(['UNITS', 'PRICE'], axis=1)
        return samples, labels

    def _encode_levels(self, data):
        encoded_df = pd.get_dummies(data['LEVEL'], prefix='LEVEL', prefix_sep='_')
        data = data.drop(['LEVEL'], axis=1)
        data = data.join(encoded_df, how='right')
        return data

    def _create_model(self, X_train, y_train):
        param_grid = {'n_estimators': [10, 20, 30, 40],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [None, 20, 30, 40]}
        regrf =GridSearchCV(RandomForestRegressor(),param_grid, cv=3)
        regrf.fit(X_train, y_train)
        print(regrf.best_params_ )
        return regrf

    def _create_data_for_forecasting_price(self, columns, forecasted_demand_level):
        dataset = pd.DataFrame(index=[0, 1], columns=columns)

        for idx in dataset.index:
            feature = int(input('will the product be in in-store circular for the week #{}?'
                                '\n (if yes, enter  1, no - 0)'.format(datetime.date.today().isocalendar()[1]+(idx+1))))
            dataset.loc[idx, 'FEATURE'] = feature

            display = int(input('will the product be a part of in-store promotional display for the week #{}?'
                                '\n (if yes, enter  1, no - 0)'.format(datetime.date.today().isocalendar()[1]+(idx+1))))
            dataset.loc[idx, 'DISPLAY'] = display

            tpr_only = int(input('Will the price of the product be temporarily reduced for the week #{} ?'
                                '\n Pproduct will not be on display or in an advertisement.'
                                '\n (if yes, enter  1, no - 0)'.format(datetime.date.today().isocalendar()[1]+(idx+1))))
            dataset.loc[idx, 'TPR_ONLY'] = tpr_only

            if forecasted_demand_level[idx] == 'low':
                dataset.loc[idx, 'LEVEL_high'] = 0
                dataset.loc[idx, 'LEVEL_low'] = 1
                dataset.loc[idx, 'LEVEL_medium'] = 0
            elif forecasted_demand_level[idx] == 'medium':
                dataset.loc[idx, 'LEVEL_low'] = 0
                dataset.loc[idx, 'LEVEL_medium'] = 1
                dataset.loc[idx, 'LEVEL_high'] = 0
            elif forecasted_demand_level[idx] == 'high':
                dataset.loc[idx, 'LEVEL_low'] = 0
                dataset.loc[idx, 'LEVEL_medium'] = 0
                dataset.loc[idx, 'LEVEL_high'] = 1
        return dataset
