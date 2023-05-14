import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
import mysql.connector



def getNextPredictions(disease):
    model = tf.saved_model.load(f'./{disease} Models')
    model.predict()
    return

def genTrainingSet(disease,rFactor,locality):
    diseaseY = disease
    if disease == 'covidDeaths':
        diseaseY = 'Deaths'
        disease = 'covid'
    df = sqlToPandas(disease,locality)
    trainingDf = df[getFeatures(disease)+getY(diseaseY)]
    trainingDf = resample(trainingDf,rFactor)
    return trainingDf

def getFeatures(disease):
    features = ['disease_deaths','deaths/pop','cases/pop','population','urban_population']
    return features

def getY(disease):
    if disease == 'Deaths':
        y = ['disease_deaths']
    else:
        y = ['disease_cases']
    return y

def USData(data,disease):
    newDf = data.groupby(['Year','Week'])
    print(newDf)
    return

def sqlToPandas(name,locality):
    try :
        con = mysql.connector.connect(
        host = "localhost",
        database = "SDDB",
        user = "root",
        password = "password",
        port = 3306
        )
    except:
        print("Could not connect to database.")
    if name == 'diseases':
        query = f"SELECT DISTINCT disease_name FROM weekly_data"
    elif name == 'states':
        query = f"SELECT DISTINCT state FROM population_data"
    elif locality == 'US':
        query = f"SELECT * from disease_weekly_totals WHERE disease = '{name}'"
    elif name == 'covid':
        query = f"SELECT * from weekly_data WHERE disease_name = '{name}' AND state = '{locality}'"
    elif name == 'population':
        query = f"SELECT * from population"
    else:
        query = f"SELECT * from weekly_data WHERE disease_name = '{name}' AND state = '{locality}'"
        
    data = pd.read_sql_query(query, con)
    return data

def resample(data,factor):
    if factor <= 1:
        return data
    expanded = data.rolling(window = 2,axis = 0).sum()
    expanded = expanded.dropna().reset_index(drop=True)
    resampledData = pd.concat([data, expanded]).sort_index(kind='merge').reset_index(drop = True)
    if factor > 1:
        resampledData = resample(resampledData,factor - 1)
    resampledData = data
    return resampledData

#x must be horizontal 
#x is feature set, y is to be predicted

def genModel(dataset,dName,locality,n_steps_in,n_steps_out,split_point,printTest = False):
    outUrl = f'./test/{dName} Models'
    #usDf = sqlToPandas('states')
    #stateSet = usDf['state'].unique()
    scaler = MinMaxScaler(feature_range=(0, 1))
    #url = f'./trainingSets/{dName}/{state}.csv'
    featureSet = getFeatures(dName)
    ySet = getY(dName)
    x = dataset[featureSet].to_numpy()
    y = dataset[ySet].to_numpy()
    features = np.empty((0,0))
    for row in x.T:
        if features.shape == (0,0):
            row = row.reshape(len(row),1)
            scaler.fit_transform(row)
            features = row
        else:
            row = row.reshape(len(row),1)
            features = np.hstack([features,scaler.fit_transform(row)])
    y_old = y.reshape(len(y),1)
    y = scaler.fit_transform(y_old)
    npArr = np.hstack([features,y])
    X, Y = split_sequences(npArr,n_steps_in,n_steps_out)
    #split_point
    train_x , train_y = X[:split_point, :] , Y[:split_point, :]
    test_x , test_y = X[split_point:, :] , Y[split_point:, :]
    #learning rate
    opt = keras.optimizers.Adam(learning_rate=0.001)
    tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=30)
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, len(x[0]))))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.add(Activation('linear'))
    model.compile(loss='mse' , optimizer=opt , metrics=['mse'])
    count = 0
    while True:
        history = model.fit(train_x , train_y , epochs=100 , steps_per_epoch=5, verbose=0 ,validation_data=(test_x, test_y) ,shuffle=False, callbacks = [callback])
        count += 1
        start = split_point
        end = start + n_steps_in 
        last = end + n_steps_out
        y_pred_inv , dataset_test_y , past_data, new_pred = prep_data(features, y_old,n_steps_in ,n_steps_out, start , end , last,model)
        rmse = evaluate_prediction(y_pred_inv , dataset_test_y, 'LSTM' , start , end)
        real_rmse = np.sqrt(history.history["val_mse"][-1])
        if real_rmse < .5 or count > 15:
            break
    #print(f'{locality} loss: {history.history["loss"][-1]}')
    #print(f'{locality} MSE: {history.history["mse"][-1]}')
    #print(f'{locality} Val_Loss: {history.history["val_loss"][-1]}')
    #print(f'{locality} ValMSE: {history.history["val_mse"][-1]}')
    
    model.save(f'{outUrl}/{locality}.h5')
    loss = history.history["val_loss"][-1]
    if printTest == True:
        plot_multistep(past_data, y_pred_inv , dataset_test_y , start , end)
    return model, y_pred_inv, new_pred, real_rmse, loss

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
 X, y = list(), list()
 for i in range(len(sequences)):
  # find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out-1
  # check if we are beyond the dataset
  if out_end_ix > len(sequences):
   break
  # gather input and output parts of the pattern
  seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)

def prep_data(features, y,n_steps_in ,n_steps_out, start , end , last,model):
    #prepare past and groundtruth
    dataset_test_X = features[start:end, :]
    past_data = y[:end , :]
    dataset_test_y = y[end:last , :]
    new_X = features[-1-n_steps_in:,:]
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler1.fit(dataset_test_y)
    
#predictions
    features = dataset_test_X.reshape(1,dataset_test_X.shape[0],dataset_test_X.shape[1])
    new_features = new_X.reshape(1,new_X.shape[0],new_X.shape[1])
    y_pred = model.predict(features)
    new_pred = model.predict(new_features)
    new_pred = scaler1.inverse_transform(new_pred)
    y_pred_inv = scaler1.inverse_transform(y_pred)
    y_pred_inv = y_pred_inv.reshape(n_steps_out,1)
    y_pred_inv = y_pred_inv[:,0]
    
    return y_pred_inv , dataset_test_y , past_data , new_pred
# Calculate MAE and RMSE
                      
def evaluate_prediction(predictions, actual, model_name , start , end):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    #print('Mean Absolute Error: {:.2f}'.format(mae))
    #print('Root Mean Square Error: {:.2f}'.format(rmse))
    #print('')
    return rmse
    
def plot_multistep(history, prediction1 , groundtruth , start , end):
    plt.figure(figsize=(10, 5))
    history = history[int(.75*len(history)):]
    y_mean = mean(prediction1)
    range_history = len(history)
    range_future = list(range(range_history, range_history + len(prediction1)))
    plt.plot(np.arange(range_history), np.array(history), label='History')
    plt.plot(range_future, np.array(prediction1),label='Forecasted with LSTM')
    plt.plot(range_future, np.array(groundtruth),label='GroundTruth')
    plt.legend(loc='upper left')
    plt.title("US Covid Cases".format(start, end, y_mean) ,  fontsize=18)
    plt.xlabel('Week' ,  fontsize=18)
    plt.ylabel('Cases' , fontsize=18)

#not done
def getPredictions(model,trainingY,testData):
    #training data necessary to get the scaler
    y_pred = model.predict(trainingY)
    scaler = MinMaxScaler(feature_range(0,1))
    scaler.fit(trainingY)
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred
