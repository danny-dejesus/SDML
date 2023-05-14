# SDML
Senior Design LSTM models for Infectious Disease Risk Predictor

funcs.py includes all functions necessary in the generation of LSTM models

runMl.py 
- first generates a training and testing set from SQL DB for each of the states and diseases
- then trains each model
- grabs next predictions
- stores this information into .csv files to then be parsed
