from forecaster import Forecaster


forecaster = Forecaster('dunnhumby _Breakfast-at-the-Frat/dunnhumby - Breakfast at the Frat.xlsx')

# download data
df = forecaster.create_data()

# fill NAN in price and base price columns
df = forecaster.fill_nan_in_price(df)

# select a product name, store name, and period (the price will be found for this period)
product = df.DESCRIPTION[23]
store = df.STORE_NAME[23]
period = 2

# select the data for the specified product name, store name
product_history_in_store = forecaster._prepare_product_data(df, product, store)

# split data on train/test set
train_data, test_data = forecaster._split_data_on_train_test_set(product_history_in_store, 75)

# create samples/labels sets
train_set, test_set, train_labels, test_labels = forecaster._split_data_on_samples_and_labels(train_data, test_data, -period)

# encode categorical columns and scale numerical columns
scaler_dict, encoded_train_set = forecaster._encode_train_set(train_set)
encoded_test_set = forecaster._encode_test_set(test_set, scaler_dict)

# reshape samples sets
X_train = forecaster._reshape_data(encoded_train_set)
X_test = forecaster._reshape_data(encoded_test_set)

# create LSTM NN model
lstm_model = forecaster._lstm_neural_network_model(100, X_train.shape[1], X_train.shape[2])

# fit LSTM NN model
history = lstm_model.fit(X_train,
                         train_labels,
                         epochs=500,
                         batch_size=50,
                         validation_data=(X_test, test_labels),
                         verbose=2,
                         shuffle=False)

# predict labels for test samples set
prediction = forecaster._predict_labels(lstm_model, X_test)

# calculate metrics
metrics_dict = forecaster._calculate_metrics(test_labels, prediction)
print(metrics_dict)

# visualize the loss
forecaster._plot_graph_of_loss(history)

# predict new price
last_data_in_product_history = train_data.loc[[train_data.index[-period]]]
encoded_history = forecaster._encode_test_set(last_data_in_product_history, scaler_dict)
encoded_history = forecaster._reshape_data(encoded_history)
forecaster._predict_labels(lstm_model, encoded_history)
