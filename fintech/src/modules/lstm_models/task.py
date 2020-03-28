import argparse
import logging
import tensorflow as tf 
import pandas as pd
import numpy as np
from data import Dataset
import model
import matplotlib.pyplot as plt 

logging.getLogger().setLevel(logging.INFO)

#model data preparation
def model_data_prep(train_data, seq_length):
    """
    intakes the dataframe  
    returns 
        two numpy arrays of predictors and corresponding labels.
    """
    x_train, y_train = [], [] 

    train = train_data.values
    length = len(train) - seq_length
    
    #reshape into records
    train = train.reshape(-1, 1)

    for i in range(seq_length, length):
        x_train.append(train[i-seq_length:i])
        y_train.append(train[i])
    
    x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train

#main function
def main(params):
    filepath = params['filepath']
    date = params['date']
    forecastcol = params['forecastcolumn']
    epochs = params['epochs']
    bs = params['batch_size']
    ws = params['sequence_length']

    #read the filepath
    df = pd.read_csv(filepath, sep=",")
    dataobj = Dataset(df, date, forecastcol)

    #normal trian and eval split
    #this split doesn't belong to model data preparation
    train_split, eval_split = dataobj.normal_split()

    model_train, model_labels = model_data_prep(train_split, ws)

    logging.info("Train Data Shape:{}".format(model_train.shape))
    logging.info("Train Label Shape:{}".format(model_labels.shape))
    
    #call a model file
    logging.info("============= MODEL TRAINING STARTED =============")
    network, modeldump = model.lstm(model_train, 
                model_labels, epochs=epochs, 
                batch_size = bs)
    

    #model.plot_loss(modeldump)

    logging.info("============== MODEL PREDICTIOn STARTED =============")
    predictions = model.forecast_fn(network, 
                    model_train[-1],
                    len(eval_split),
                    ws)
    
    print(predictions)
    

    assert len(eval_split) == len(predictions), "Length Mismatch between Actuals and Predictions"

    plt.plot(eval_split)
    plt.plot(predictions)
    plt.show()
    
    logging.info("Model Score on Test Data:{}".format(model.evaluate_model_performance(eval_split, predictions)))

if __name__ == '__main__':
    
    #parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--filepath",
                type=str,
                help='Local Path to read the dataset')
    
    parser.add_argument("--date",
                type=str,
                help='Date column in the dataset')
    
    parser.add_argument("--forecastcolumn",
                type=str,
                help='Target column to forecast')
    
    parser.add_argument('--epochs', type=int,
                help='Number of Steps the model has to train')
    
    parser.add_argument('--batch_size', type=int,
                help='Bath Size')
    
    parser.add_argument('--sequence_length', type=int,
                help='Window Sequence Length')
    

    params = parser.parse_args()    

    #passing as a dictionary
    main(vars(params))