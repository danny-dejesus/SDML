import numpy as np
import funcs as f
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
log = pd.DataFrame({'locality':[],'disease_name':[],'RMSE':[],'Loss':[]})
nextPreds = pd.DataFrame({'locality':[],'disease_name':[],'predictions':[]})
logFile = open(f'./test/log.txt','w')
diseases = f.sqlToPandas('diseases','US')['disease_name']
covDeath = pd.Series({'disease_name':'covidDeaths'})
states = f.sqlToPandas('states','US')['state']
usDf = pd.Series({'state':'US'})
states = pd.concat([states,usDf],ignore_index=True)
resampleFactor = 7
stepsIn = 8
stepsOut = 4



for locality in states:
    for disease in diseases:
        try:
            dataset = f.genTrainingSet(disease,resampleFactor,locality)
            splitPoint = int(len(dataset)*.75)
            model, predictions, nextPredictions, rmse, loss = f.genModel(dataset,disease,locality,stepsIn,stepsOut,splitPoint,printTest = False)
            logFile.write(f'Success {locality},{disease}')
            logFile.write(f'\t METRICS: RMSE: {rmse} , Loss: {loss}\n')
            d = {'locality':[locality],'disease_name':[disease],'RMSE':rmse,'Loss':[loss]}
            df = pd.DataFrame(data = d)
            pData = {'locality':[locality],'disease_name':[disease], 'predictions':[nextPredictions]}
            predDf = pd.DataFrame(data = pData)
            log = pd.concat([log,df],ignore_index = True,axis=0)
            nextPreds = pd.concat([predDf,nextPreds],ignore_index = True,axis=0)
        except:
            print(f"Error with {locality} and {disease}")
log.to_csv('./log.csv')
nextPreds.to_csv('./preds.csv')
