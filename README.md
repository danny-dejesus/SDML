# Senior Design LSTM models for Infectious Disease Risk Predictor

funcs.py includes all functions necessary in the generation of LSTM models
- genTrainingSet generates training set using features, targets, and SQLtoPandas
- getFeatures gets features given a disease name
- getY gets targets for given disease name
- SQLtoPandas connects to SQL and queries for disease and locality
- resample takes care of resampling
- genModel trains and test models and can return the newest predictions
- all other functions are either deprecated or helper functions for data management

runMl.py 
- first generates a training and testing set from SQL DB for each of the states and diseases
- then trains each model
- grabs next predictions
- stores this information into .csv files to then be parsed
