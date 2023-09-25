##############################################################################################
# Script to build the dataset of the source domain.
# Used for the study presented in [*].
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification 
# F.Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek 
##############################################################################################

from utilitary import *


##############################################################################################
# Hyper-parameters
##############################################################################################

# Path to the raw dataset: in order: EEG-eye-state, gas-mixture, OPPORTUNITY, energy-appliance
rawDataPaths= ['../data/source/raw/EEG-eye-state/data.arff',
    '../data/source/raw/gas-mixture/',
    '../data/source/raw/OPPORTUNITY/',
    '../data/source/raw/energy-prediction/energydata_complete.csv'] 

savedDataPath = '../data/source/processed/' # Path to save the mixed source dataset

targetDataset = 'DEAP' # 'CogAge' or 'DEAP'



##############################################################################################
# Main
##############################################################################################
if __name__ == '__main__':

    if targetDataset == 'CogAge':
        
        downsample = 15000

        # Source dataset to train the single-channel DNN for smartphone
        timeWindow = 800
        stride = 200

        buildDataFrames(
            dataPathList=rawDataPaths,
            timeWindow=timeWindow,
            stride=stride,
            savePath=savedDataPath+'/'+targetDataset+'/smartphone/',
            downsample=downsample)

        # Source dataset to train the single-channel DNN for the MS Band
        timeWindow = 268
        stride = 67

        buildDataFrames(
            dataPathList=rawDataPaths,
            timeWindow=timeWindow,
            stride=stride,
            savePath=savedDataPath+'/'+targetDataset+'/smartwatch/',
            downsample=downsample)

        # Source dataset to train the single-channel DNN for the JINS glasses
        timeWindow = 80
        stride = 20

        buildDataFrames(
            dataPathList=rawDataPaths,
            timeWindow=timeWindow,
            stride=stride,
            savePath=savedDataPath+'/'+targetDataset+'/smartglasses/',
            downsample=downsample)


    elif targetDataset == 'DEAP':
        timeWindow = 128
        stride = 128
        downsample = 150000

        buildDataFrames(
            dataPathList=rawDataPaths,
            timeWindow=timeWindow,
            stride=stride,
            savePath=savedDataPath+'/'+targetDataset+'/',
            downsample=downsample)





