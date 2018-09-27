from demand_forecaster import DemandForecaster
from keras.callbacks import EarlyStopping


forecaster = DemandForecaster('dunnhumby _Breakfast-at-the-Frat/dunnhumby - Breakfast at the Frat.xlsx')

# download data and drop Nan
df = forecaster.create_and_prepare_full_data()

# select a product name, store name, and period (the price will be found for this period)
product_upc = df.UPC[23]
store = df.STORE_NAME[23]

# product_upc = 4116709428
# store = 'MIDDLETOWN'

# select the data for the specified product name, store name
product_history_in_store = forecaster._prepare_product_data(df, product_upc, store)

# create samples/labels sets
samples, labels = forecaster._split_data_on_samples_and_labels(product_history_in_store)

# split data on train/test set
train_set, test_set, train_labels, test_labels = forecaster._split_data_on_train_test_set(samples, labels, 80)

# encode categorical columns and scale numerical columns
scaler_dict, encoded_train_cat_columns, encoded_train_set = forecaster._encode_train_set(train_set)
encoded_test_set = forecaster._encode_test_set(test_set, scaler_dict, encoded_train_cat_columns)

# reshape samples sets
X_train = forecaster._reshape_data(encoded_train_set)
X_test = forecaster._reshape_data(encoded_test_set)

# create LSTM NN model
lstm_model = forecaster._lstm_neural_network_model(100, X_train.shape[1], X_train.shape[2])

# fit LSTM NN model
early_stopping = EarlyStopping(patience=3)
history = lstm_model.fit(X_train,
                         train_labels,
                         epochs=2000,
                         batch_size=50,
                         validation_data=(X_test, test_labels),
                         callbacks=[early_stopping],
                         verbose=2,
                         shuffle=False)

# predict labels for test samples set
prediction = forecaster._predict_labels(lstm_model, X_test)

# calculate metrics
metrics_dict = forecaster._calculate_metrics(test_labels, prediction)
print(metrics_dict)

# visualize the loss
forecaster._plot_graph_of_loss(history)

# predict demand
last_data_in_product_history = product_history_in_store.loc[[product_history_in_store.index[-1]]]
encoded_history = forecaster._encode_test_set(test_set, scaler_dict, encoded_train_cat_columns)
encoded_history = forecaster._reshape_data(encoded_history)
forecaster._predict_labels(lstm_model, encoded_history)
