import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def lstm(predictors, labels, epochs=50, batch_size=16):

    #network
    model = Sequential()
    model.add(LSTM(units=60, return_sequences=True, input_shape=(predictors.shape[1],1)))
    model.add(LSTM(units=30))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    print(model.summary())
    
    dump = model.fit(predictors, labels, epochs=epochs, batch_size=1, verbose=2)

    return model, dump

def forecast_fn(trained_model, prime, forecast_period, ws, last_date=None):
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

    for i in range(forecast_period):
        to_be_predicted = np.expand_dims(datum, axis=0)
        print(to_be_predicted.shape)
        result = trained_model.predict(to_be_predicted)
        
        #append the result to the data
        result_list = np.append(result_list, result)
        datum = np.delete(datum, 0)
        datum = np.append(datum, result)
        datum = datum.reshape(ws, 1)
    
    result_list = result_list.reshape(-1, 1)
    return result_list
