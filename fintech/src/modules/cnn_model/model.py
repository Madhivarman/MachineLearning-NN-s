import numpy as np
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D
from sklearn.metrics import mean_squared_error

def conv1d(predictors, labels, epochs = 50, batch_size = 16):
    """
    defines a convolution (1Dimensional) network.
    
    input = two numpy arrays (predictors and labels)
    returns = trained model object;
                model_history (for plotting train vs valid loss)
    """
    filter_len, window_len = predictors.shape[0], predictors.shape[1]
    regressor = Sequential()
    regressor.add(Conv1D(filters = filter_len, kernel_size = window_len,
                        activation = "relu", 
                        input_shape = (predictors.shape[1], 1),
                        padding = "same"))
    #regressor.add(MaxPooling1D(pool_size=2))
    regressor.add(Flatten())
    regressor.add(Dense(10, activation='relu'))
    regressor.add(Dense(1))
    
    regressor.compile(optimizer = "adam", loss = "mse")
    model = regressor.fit(predictors, labels, validation_split = 0.25,
                epochs = epochs, 
                batch_size = batch_size, 
                verbose=2)
    
    return regressor, model


def forecast_fn(trained_model, prime, forecast_period, actual_test_data,last_date=None):
    """
        input
            trained_model = Model Dump
            prime(the last element of the train data which forecast starts)
            last_date(the last date of the trainset. Forecast dates will be marked from this)
            forecast_period(length of the test dataset)
    """
    result = []
    datum = prime
    result_list = np.array([])

    actual_test_data = actual_test_data.values.tolist()

    for i in range(forecast_period):
        to_be_predicted = np.expand_dims(datum, axis=0)
        result = trained_model.predict(to_be_predicted)
        
        #append the actual test data to the result
        #column and predict it for next day
        result_list = np.append(result_list, result)
        datum = np.delete(datum, 0)
        datum = np.append(datum, actual_test_data[i])
        datum = datum.reshape(-1, 1)
    
    result_list = result_list.reshape(-1, 1)
    return result_list

def evaluate_model_performance(y, y0, type='mse'):
    if type == 'mse':
        score = mean_squared_error(y, y0)
    
    return score