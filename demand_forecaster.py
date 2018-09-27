import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping


class Forecaster:

    def __init__(self, path):
        self.path = path

    def create_and_prepare_full_data(self):
        xlsx_file = pd.ExcelFile(self.path)

        store_lookup = pd.read_excel(xlsx_file, sheet_name=xlsx_file.sheet_names[1], header=1, usecols=8)
        products_lookup = pd.read_excel(xlsx_file, sheet_name=xlsx_file.sheet_names[2], header=1, usecols=5)
        transaction_data = pd.read_excel(xlsx_file, sheet_name=xlsx_file.sheet_names[3], header=1, usecols=11)

        df = pd.merge(products_lookup, transaction_data, how='inner', on='UPC')
        df = df.rename(columns={'STORE_NUM': 'STORE_ID'})
        df = pd.merge(df, store_lookup, how='inner', on='STORE_ID')
        df = df.drop(['PARKING_SPACE_QTY'], axis=1)
        df = df.dropna(axis=0)
        return df

    def recommend_price(self, data):
        product_upc = int(input('enter a product UPC code... '))
        store = input('enter a store name... ').upper()

        product_history_in_store = self._prepare_product_data(data, product_upc, store)

        samples, labels = self._split_data_on_samples_and_labels(product_history_in_store)

        train_set, test_set, train_labels, test_labels = self._split_data_on_train_test_set(samples, labels, 80)

        scaler_dict, encoded_train_set = self._encode_train_set(train_set)

        encoded_test_set = self._encode_test_set(test_set, scaler_dict)

        X_train = self._reshape_data(encoded_train_set)
        X_test = self._reshape_data(encoded_test_set)

        lstm_model = self._lstm_neural_network_model(100, X_train.shape[1], X_train.shape[2])

        early_stopping = EarlyStopping(patience=3)
        history = lstm_model.fit(X_train,
                                 train_labels,
                                 epochs=2000,
                                 batch_size=50,
                                 validation_data=(X_test, test_labels),
                                 callbacks=[early_stopping],
                                 verbose=2,
                                 shuffle=False)

        # last_data_in_product_history = train_data.loc[train_data.index[-2:].values]
        # encoded_history = self._encode_test_set(last_data_in_product_history, scaler_dict)
        # encoded_history = self._reshape_data(encoded_history)
        # prediction_for_data = self._predict_labels(lstm_model, encoded_history)
        return history

    def _prepare_product_data(self, data, product_upc, store_name):
        history = data[(data.UPC == product_upc) & (data.STORE_NAME == store_name)]
        history.index = history['WEEK_END_DATE']
        history = history.drop('WEEK_END_DATE', axis=1)
        history.sort_index(inplace=True)
        return history

    def _split_data_on_samples_and_labels(self, data):
        labels = pd.DataFrame()
        labels['u-1'] = data.UNITS.shift(-1)
        labels['u-2'] = data.UNITS.shift(-2)

        data = data.drop(['UNITS'], axis=1)
        samples = data.drop(labels.index[-2:], axis=0)
        labels = labels.dropna(axis=0)
        return samples, labels

    def _split_data_on_train_test_set(self, data, labels, train_size_in_percent):
        train_size = int(len(data) * train_size_in_percent / 100)
        train_set, test_set = data[:train_size], data[train_size:]
        train_labels, test_labels = labels[:train_size], labels[train_size:]
        return train_set, test_set, train_labels, test_labels

    def _encode_train_set(self, train_set):
        encoded_cat_columns = self._encode_cat_column(train_set, ['ADDRESS_CITY_NAME',
                                                                  'ADDRESS_STATE_PROV_CODE',
                                                                  'MSA_CODE',
                                                                  'SEG_VALUE_NAME'])
        scaler_dict, scaled_num_columns = self._scale_num_column(train_set,
                                                                       ['SALES_AREA_SIZE_NUM',
                                                                        'AVG_WEEKLY_BASKETS',
                                                                        'BASE_PRICE',
                                                                        'PRICE',
                                                                        'VISITS',
                                                                        'HHS',
                                                                        'SPEND'])
        encoded_train_set = encoded_cat_columns.join(scaled_num_columns, how='right')
        encoded_train_set = encoded_train_set.join(train_set[['FEATURE', 'DISPLAY', 'TPR_ONLY']], how='right')
        return scaler_dict, encoded_train_set

    def _encode_test_set(self, test_set, scaler_dict):
        encoded_cat_columns = self._encode_cat_column(test_set, ['ADDRESS_CITY_NAME',
                                                                        'ADDRESS_STATE_PROV_CODE',
                                                                        'MSA_CODE',
                                                                        'SEG_VALUE_NAME'])
        scaled_num_columns = pd.DataFrame()
        for column in ['SALES_AREA_SIZE_NUM',
                       'AVG_WEEKLY_BASKETS',
                       'BASE_PRICE',
                       'PRICE',
                       'VISITS',
                       'HHS',
                       'SPEND']:
            scaler = scaler_dict[column]
            encoded_col = pd.DataFrame(scaler.transform(test_set[[column]]), columns=[column])
            encoded_col.index = test_set.index
            scaled_num_columns = scaled_num_columns.join(encoded_col, how='right')

        encoded_test_set = encoded_cat_columns.join(scaled_num_columns, how='right')
        encoded_test_set = encoded_test_set.join(test_set[['FEATURE', 'DISPLAY', 'TPR_ONLY']], how='right')
        return encoded_test_set

    def _reshape_data(self, data):
        new_data = np.array(data).reshape((np.array(data).shape[0], 1, np.array(data).shape[1]))
        return new_data

    def _lstm_neural_network_model(self, number_of_neurons,
                                  first_parameter_of_shape,
                                  second_parameter_of_shape,
                                  random_state=0):
        random.seed(random_state)
        model = Sequential()
        model.add(LSTM(number_of_neurons, input_shape=(first_parameter_of_shape, second_parameter_of_shape)))
        model.add(Dense(2))
        model.compile(loss='mae', optimizer='adam')
        return model

    def _predict_labels(self, model, data):
        predicted_labels = model.predict(data)
        return predicted_labels

    def _calculate_metrics(self, true_labels, predicted_labels):
        rmse = math.sqrt(mean_squared_error(true_labels, predicted_labels))
        mae = mean_absolute_error(true_labels, predicted_labels)
        r2_metric = r2_score(true_labels, predicted_labels)
        metrics = {'RMS Error': rmse, 'R2 score': r2_metric, 'MA Error': mae}
        return metrics

    def _plot_graph_of_loss(self, model_history):
        plt.plot(model_history.history['loss'], label='train')
        plt.plot(model_history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def _encode_cat_column(self, data, list_of_cat_column_name):
        encoded_data = pd.DataFrame()
        for column in list_of_cat_column_name:
            encoded_df = pd.get_dummies(data[column], prefix=column, prefix_sep='_')
            encoded_data = encoded_data.join(encoded_df, how='right')
        return encoded_data

    def _scale_num_column(self, data, list_of_num_column_name):
        scaler_dict = {}
        scaled_data = pd.DataFrame()
        for column in list_of_num_column_name:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_col = pd.DataFrame(scaler.fit_transform(data[[column]]), columns=[column])
            scaled_col.index = data.index
            scaled_data = scaled_data.join(scaled_col, how='right')
            scaler_dict[column] = scaler
        return scaler_dict, scaled_data
