###################################################################################
# Utilitary file for the study in [*]
# Source datasets: OPPORTUNITY, EEG-eye-state, Gas-mixture, energy-appliance
# Target datasets: DEAP, CogAge
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Moddality Classification 
# F.Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek 
###################################################################################

import numpy as np
from scipy.io.arff import loadarff
from math import floor
from keras import backend as K
import tensorflow as tf
import os
import time
import sys
import random

from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, f1_score

from math import floor


import keras
from keras.models import Sequential, Model, load_model, model_from_json
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Flatten, Activation, Add, merge, Input, concatenate, Lambda 
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM

from pdb import set_trace as st 


#--------------------------------------------------------------------------------
# TDNN definition
#--------------------------------------------------------------------------------
def mlp(
    inputShape,
    denseLayersSize,
    activation,
    nbClasses,
    withSoftmax=True):

    model = Sequential()
   
    # Input layer
    model.add(Dense(denseLayersSize[0],activation=activation,name='dense1',input_shape=inputShape))

    # Dense layers
    if len(denseLayersSize)>1:
        for idx in range(1,len(denseLayersSize)):
            model.add(Dense(denseLayersSize[idx], activation=activation,name='dense'+str(idx+1)))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax',name='softmax'))

    # Return the model
    return model


def bnMlp(
    inputShape,
    denseLayersSize,
    activation,
    nbClasses,
    withSoftmax=True):

    model = Sequential()
   
    # Input layer
    model.add(BatchNormalization(name='bn',input_shape=inputShape))

    # Dense layers
    if len(denseLayersSize)>1:
        for idx in range(len(denseLayersSize)):
            model.add(Dense(denseLayersSize[idx], activation=activation,name='dense'+str(idx+1)))

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax',name='softmax'))

    # Return the model
    return model


def cnn(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activation,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(Conv1D(nkerns[0], kernel_size=filterSizes[0], activation=activation,input_shape=inputShape,name='conv1'))
    model.add(MaxPooling1D(pool_size=poolSizes[0]))

    if len(nkerns)>1:
        for idx in range(1,len(nkerns)):
            model.add(Conv1D(nkerns[idx], kernel_size=filterSizes[idx], activation=activation,name='conv'+str(idx+1)))
            model.add(MaxPooling1D(pool_size=poolSizes[idx]))
  
    # Fully-connected layer
    model.add(Flatten())

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    return model


def bnCnn(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activation,
    nbClasses,
    withSoftmax=True):

    model = Sequential()

    model.add(BatchNormalization(input_shape=inputShape,name='bn'))
    
    if len(nkerns)>1:
        for idx in range(len(nkerns)):
            model.add(Conv1D(nkerns[idx], kernel_size=filterSizes[idx], activation=activation,name='conv'+str(idx+1)))
            model.add(MaxPooling1D(pool_size=poolSizes[idx]))
  
    # Fully-connected layer
    model.add(Flatten())

    # Softmax layer
    if withSoftmax:
        model.add(Dense(nbClasses, activation='softmax'))

    return model



def autoencoder(
    inputShape,
    denseLayersSize,
    activation):

    model = Sequential()

    model.add(Dense(denseLayersSize[0],activation=activation,input_shape=inputShape,name='encoder1'))

    if len(denseLayersSize)>1:
        # Encoder
        for idx in range(1,len(denseLayersSize)):
            model.add(Dense(denseLayersSize[idx],activation=activation,name='encoder'+str(idx+1)))
    
        # Decoder
        for idx in range(len(denseLayersSize)):
            model.add(Dense(denseLayersSize[len(denseLayersSize)-idx-1],activation=activation,name='decoder'+str(idx+1)))

    else:
        model.add(Dense(denseLayersSize[0],activation=activation,name='decoder1'))

    # Output layer
    model.add(Dense(inputShape[0],activation='linear')) 

    return model


def convAutoencoder(
    inputShape,
    nkerns,
    filterSizes,
    poolSizes,
    activation):

    model = Sequential()

    nbLayersEncoder = len(nkerns)

    if nbLayersEncoder > 1:

        # Encoder
        model.add(Conv1D(nkerns[0],kernel_size=filterSizes[0],activation=activation,padding='same',input_shape=inputShape,name='encoder1'))
        model.add(MaxPooling1D(pool_size=poolSizes[0]))
        for idx in range(1,nbLayersEncoder):
            model.add(Conv1D(nkerns[idx],kernel_size=filterSizes[idx],activation=activation,padding='same',name='encoder'+str(idx+1)))
            model.add(MaxPooling1D(pool_size=poolSizes[idx]))
        # Decoder
        for idx in range(0,nbLayersEncoder):
            model.add(UpSampling1D(size=poolSizes[nbLayersEncoder-idx-1][0]))
            model.add(Conv1D(nkerns[nbLayersEncoder-idx-1],kernel_size=filterSizes[nbLayersEncoder-idx-1],activation=activation,padding='same',name='decoder'+str(idx+1)))
        model.add(Flatten())
        model.add(Dense(inputShape[0],activation='linear'))
        model.add(Reshape(inputShape))

    else:
        model.add(Conv1D(nkerns[0],kernel_size=filterSizes[0],activation=activation,padding='same',input_shape=inputShape,name='encoder1'))
        model.add(MaxPooling1D(pool_size=poolSizes[0]))
        model.add(UpSampling1D(size=poolSizes[0]))
        model.add(Conv1D(nkerns[0],kernel_size=filterSizes[0],activation=activation,padding='same',name='decoder1'))
        model.add(Flatten())
        model.add(Dense(inputShape[0],activation='linear'))     
        model.add(Reshape(inputShape))

    return model


#--------------------------------------------------------------------------------
# Source datasets parameters
#--------------------------------------------------------------------------------

### Labels with additional data
labelsTableAllDatasets = dict([
    (0,'accelerometer'), # Acceleration in milli g
    (1,'IMU accelerometer'), # Normalized acceleration in milli g
    (2,'IMU gyroscope'), # Unit unknown
    (3,'IMU magnetometer'), # Unit unknown
    (4,'IMU Eu'), # Measurement in degrees
    (5,'IMU angular velocity'), # Measurement in mm/s
    (6,'IMU compass'), # Measurement in degrees
    (7,'EEG'),
    (8,'concentration'),
    (9,'conductance'),
    (10,'energy use'), # In W/h
    (11,'temperature'), # In degree celsius
    (12,'humidity'), # in %
    (13,'pressure'), # in mmHg
    (14,'wind speed'), # in m/s
    (15,'visibility') # in km
    ])

### Labels without the gas-mixture dataset
labelsTableWithoutGasMixture = dict([
    (0,'accelerometer'), # Acceleration in milli g
    (1,'IMU accelerometer'), # Normalized acceleration in milli g
    (2,'IMU gyroscope'), # Unit unknown
    (3,'IMU magnetometer'), # Unit unknown
    (4,'IMU Eu'), # Measurement in degrees
    (5,'IMU angular velocity'), # Measurement in mm/s
    (6,'IMU compass'), # Measurement in degrees
    (7,'EEG'),
    (8,'energy use'),
    (9,'temperature'),
    (10,'humidity'),
    (11,'pressure'),
    (12,'wind speed'),
    (13,'visbility')])

### Labels with OPPORTUNITY only
labelsTableOpportunityOnly = dict([
    (0,'accelerometer'), # Acceleration in milli g
    (1,'IMU accelerometer'), # Normalized acceleration in milli g
    (2,'IMU gyroscope'), # Unit unknown
    (3,'IMU magnetometer'), # Unit unknown
    (4,'IMU Eu'), # Measurement in degrees
    (5,'IMU angular velocity'), # Measurement in mm/s
    (6,'IMU compass') # Measurement in degrees
    ])

### Labels with EEG only
labelsTableEegOnly = dict([
    (0,'EEG'),
    (1,'other')
    ])

### Labels with gas mixture only
labelsTableGasMixtureOnly = dict([
    (0,'concentration'),
    (1,'conductance')
    ])

### Labels with energy appliance only
labelsTableEnergyApplianceOnly = dict([
    (0,'energy use'), # In W/h
    (1,'temperature'), # In degree celsius
    (2,'humidity'), # in %
    (3,'pressure'), # in mmHg
    (4,'wind speed'), # in m/s
    (5,'visibility') # in km
    ])

### Labels without the OPPORTUNITY dataset
labelsTableWithoutOpportunity = dict([
    (0,'EEG'),
    (1,'concentration'),
    (2,'conductance'),
    (3,'energy use'),
    (4,'temperature'),
    (5,'humidity'),
    (6,'pressure'),
    (7,'wind speed'),
    (8,'visbility')])

nbModalitiesAllDatasets = len(labelsTableAllDatasets.keys())
nbModalitiesOpportunityOnly = len(labelsTableOpportunityOnly.keys())
nbModalitiesEegOnly = len(labelsTableEegOnly.keys())
nbModalitiesGasMixtureOnly = len(labelsTableGasMixtureOnly.keys())
nbModalitiesEnergyApplianceOnly = len(labelsTableEnergyApplianceOnly.keys())
nbModalitiesWithoutGasMixture = len(labelsTableWithoutGasMixture.keys())
nbModalitiesWithoutOpportunity = len(labelsTableWithoutOpportunity.keys())


#--------------------------------------------------------------------------------
# Evaluation metrics
#--------------------------------------------------------------------------------

### shuffleInUnisson : apply the same random permutation to 2 different arrays/vectors
def shuffleInUnisson(a,b): # a and b must be vectors or arrays with the same number of lines
    assert len(a) == len(b)
    randomPermutation = np.random.permutation(len(a))
    return a[randomPermutation], b[randomPermutation], randomPermutation


### Keras evaluation metrics
# Currently implemented: precision, recall, f_beta_score, fmeasure
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


# Function to compute the average accuracy from a confusion matrix
# Returns the average and class accuracies
def computeAverageAccuracy(confusionMatrix):

    nbClasses = len(confusionMatrix)
    classAccuracies = np.zeros((nbClasses),dtype=np.float32)

    for idx in range(nbClasses):
        totalNbExamples = np.sum(confusionMatrix[idx])
        classAccuracies[idx] = confusionMatrix[idx,idx]/float(totalNbExamples)

    return np.mean(classAccuracies), classAccuracies


# Function to compute the average precision
#   - groundTruthLabels: list of labels of size size nb_examples.
#     The labels are assumed to be between 0 and nb_classes-1
#   - softmaxEstimations: array of softmax estimations of size nb_examples x nb_classes
# Returns the MAP and average precisions per class
def computeMeanAveragePrecision(groundTruthLabels,softmaxEstimations):

    nbExamples, nbClasses = softmaxEstimations.shape

    averagePrecisions = np.zeros((nbClasses),dtype=np.float32)

    # For all classes
    for classIdx in range(nbClasses):

        # Sort the softmaxEstimations by decreasing order, and keep the order consistent with the labels
        permutation = list(reversed(np.argsort(softmaxEstimations[:,classIdx])))
        labelArray = np.asarray(groundTruthLabels)
        labelsTmp = list(labelArray[permutation])

        # Convert the labels to binary (1-vs-all)
        for idx in range(len(labelsTmp)):
            if labelsTmp[idx] == classIdx:
                labelsTmp[idx] = 1
            else:
                labelsTmp[idx] = 0

        # Compute the averaged sum of precisions by descending order
        nbPrecisionComputations = 0
        averagePrecisionSum = 0

        for idx in range(len(labelsTmp)):
            if labelsTmp[idx] == 1:
                averagePrecisionSum += np.sum(labelsTmp[:idx+1])/float(idx+1)
                nbPrecisionComputations += 1

        if nbPrecisionComputations == 0:
             averagePrecisions[classIdx] = 0
        else:
            averagePrecisions[classIdx] = averagePrecisionSum/float(nbPrecisionComputations)

    return np.mean(averagePrecisions), averagePrecisions


# Early stopping: interrupts the training process if a monitored metric remained the same for n consecutive epochs
class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', memory=[], verbose=1, consecutiveEpochs=5):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.memory = memory
        self.consecutiveEpochs = consecutiveEpochs
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if len(self.memory) >= self.consecutiveEpochs:
            self.memory = self.memory[1:]
        self.memory += [current]

        if self.memory[1:] == self.memory[:-1] and len(self.memory) == self.consecutiveEpochs:
            if self.verbose > 0:
                print('\n')
                print("Epoch %d: early stopping after %d epochs with the same %s" % (epoch+1,self.consecutiveEpochs,self.monitor))
                print(self.memory)
                print('\n')
            self.model.stop_training = True


#--------------------------------------------------------------------------------
# Methods for processing fo source datasets
#--------------------------------------------------------------------------------

# Function loadOpportunityData: load the data from the clean OPPORTUNITY .txt files
# NOTE: activity labels are not used and therefore discarded
def loadOpportunityData(dataFileFolder):

    # List in order the files in the folder and sort between data and labels
    dataFileList = os.listdir(dataFileFolder)
    dataFileList = [e for e in dataFileList if not 'column_names' in e and not 'label_legend' in e]
    sortedDataFileList = sorted(dataFileList)
    nbFiles = len(sortedDataFileList)

    ### Allocate the data arrays
    # Compute the total number of examples
    nbExamplesInFiles = np.zeros((nbFiles),dtype=int)
    for idx in range(nbFiles):
        data = np.loadtxt(dataFileFolder+'/'+sortedDataFileList[idx],delimiter=' ')
        nbExamplesInFiles[idx] = len(data)
    totalNbExamples = np.sum(nbExamplesInFiles)
    # Compute the dimension of examples
    firstDataExample = np.loadtxt(dataFileFolder+'/'+sortedDataFileList[0],delimiter=' ')
    nbSensors = len(firstDataExample[0])-1
    # Allocate the data arrays
    data = np.zeros((totalNbExamples,nbSensors),dtype=np.float32)

    ### Load the data and labels
    storageIdx = 0
    for idx in range(nbFiles):
        dataTmp = np.loadtxt(dataFileFolder+'/'+sortedDataFileList[idx],delimiter=' ')
        data[storageIdx:storageIdx+nbExamplesInFiles[idx]] = dataTmp[:,:-1]
        storageIdx += nbExamplesInFiles[idx]

    ### Return the data and labels
    # Also returns information on the shape of the data
    return np.float32(data)


### Load the data from the EEG eye-state dataset
# The data and labels are contained in a single .arff file
# NOTE: EEG sensor label: 7
def loadEegData(dataPath):
    data, meta = loadarff(dataPath)
    nbSensors = len(data[0])-1 # NOTE: the labelling information is contained in the last column
    nbDataPoints = len(data) # NOTE: should be equal to 14,980
    resData = np.zeros((nbDataPoints,nbSensors))
    resLabels = -1*np.ones((nbDataPoints),dtype=int)
    for idx in range(nbDataPoints):
        tmpData = list(data[idx])
        resData[idx] = tmpData[:-1]
        resLabels[idx] = tmpData[-1]

    return resData, resLabels


### Load the data from the Gas-mixture dataset
# Two data files are provided (.txt format)
# Each data file has the same structure and contains
# gas concentrations (in ppm) and raw sensor readings 
# (conductances, in Intensity/Voltage) for two mixtures
# of gas (CO+Ethylene and Methane+Ethylene) 
def loadGasMixtureData(dataFolder):
    # Read the CO+Ethylene datafile
    fCo = open(dataFolder+'/ethylene_CO.txt', 'r')
    coLines = fCo.readlines()
    fCo.close()
    nbCoTimestamps = len(coLines)
    nbModalities = len(coLines[1].split())-1 # The first column is the time (in seconds)
    coData = np.zeros((nbCoTimestamps,nbModalities))
    for idx in range(1,nbCoTimestamps): # The first line is a header and doesn't contain any relevant data
        tmpData = [float(e) for e in coLines[idx].split()]
        coData[idx,:] = tmpData[1:]
    # Read the Methane+Ethylene datafile
    fMeth = open(dataFolder+'/ethylene_methane.txt', 'r')
    methLines = fMeth.readlines()
    fMeth.close()
    nbMethTimestamps = len(methLines)
    methData = np.zeros((nbMethTimestamps,nbModalities))
    for idx in range(1,nbMethTimestamps): # The first line is a header and doesn't contain any relevant data
        tmpData = [float(e) for e in methLines[idx].split()]
        methData[idx,:] = tmpData[1:]

    return coData, methData


### Load the data from the energy-preidction dataset
# The data in contained in a single .csv file
# The first column containing the date, as well as the 2 last columns containing random data are ignored
def loadEnergyPredictionData(dataPath):
    
    # Read the csv file
    data = np.genfromtxt(dataPath,delimiter=',',dtype=str) # NOTE: the data contained in the CSV is str only

    # Remove unneeded information
    data = data[1:] # Remove the first line which contains headers
    data = data[:,1:-2] # Remove the first column containing dates, and the last two containing random variables

    # Remove all spaces and " from the str data
    height, width = data.shape

    for hIdx in range(height):
        for wIdx in range(width):
            string = data[hIdx,wIdx]
            string = string.replace('"','')
            string = string.replace(' ','')
            data[hIdx,wIdx] = string

    # Convert the data to float
    data = data.astype(np.float32)

    return data


### Build 1D time windows of data from the EEG, gas-mixture, OPPORTUNITY and energy datasets
# Input argument dataPathList: contains the paths to the EEG, gas-mixture, OPPORTUNITY and energy datasets (in order)
# Input argument downsample: used to balance the amount of data used from each dataset
# Input argument offset: offset for the selection of downsampled examples
# Input argument datasets: which datasets to use. Options: 'all', 'opportunity', 'eeg', 'energy', 'gas-mixture', 'no-opportunity', 'no-gas-mixture'
def buildDataFrames(dataPathList,timeWindow,stride,savePath,downsample=15000,offset=0,datasets='all'):
    
    # Loading datasets
    # For all datasets except OPPORTUNITY, the raw data is loaded directly (matrices of size nbExamples x nbSensors)
    # For OPPORTUNITY, data separated by time windows and sensor channels are already provided (matrix of size )
    print('Loading EEG eye-state data ...')
    eegData, _ = loadEegData(dataPathList[0])
    print('Loading gas-mixture data ...')
    coData, methData = loadGasMixtureData(dataPathList[1])
    print('Loading the OPPORTUNITY dataset ...')
    oppData = loadOpportunityData(dataPathList[2])
    print('Loading the energy-prediction dataset ...')
    energyData = loadEnergyPredictionData(dataPathList[3])

    # Downsampling the source datasets to limit the amount of data obtained from them
    if downsample > 0:
        # NOTE: since the CO and METH datasets are merged after processing, only half of the number of examples are taken
        if len(coData) > downsample:
            coData = coData[offset:int(downsample/2.0)+offset]
        if len(methData) > downsample:
            methData = methData[offset:int(downsample/2.0)+offset]
        if len(energyData) > downsample:
            energyData = energyData[offset:downsample+offset]
        if len(eegData) > downsample:
            eegData = eegData[offset:downsample+offset]
        if len(oppData) > downsample:
            oppData = oppData[offset:downsample+offset]

    # Number of samples for each dataset
    nbSampEeg = len(eegData)
    nbSampCo = len(coData)
    nbSampMeth = len(methData)
    nbSampEnergy = len(energyData)
    nbSampOpp = len(oppData)

    # Number of sensor modalities for each dataset
    nbModEeg = len(eegData[0])
    nbModCo = len(coData[0])
    nbModMeth = len(methData[0])
    nbModEnergy = len(energyData[0])
    nbModOpp = len(oppData[0])

    # Number of time windows for each dataset
    nbEegFrames = int(floor((nbSampEeg-timeWindow)/stride))+1
    nbCoFrames = int(floor((nbSampCo-timeWindow)/stride))+1
    nbMethFrames = int(floor((nbSampMeth-timeWindow)/stride))+1
    nbEnergyFrames = int(floor((nbSampEnergy-timeWindow)/stride))+1
    nbOppFrames = int(floor((nbSampOpp-timeWindow)/stride))+1

    print('Building data frames with parameters T = %d and sigma = %d ...' % (timeWindow,stride))

    # Building the time windows of the EEG dataset
    eegFramesArray = np.empty((nbEegFrames*nbModEeg,timeWindow),dtype=np.float32)

    # Iteration on the EEG data to build examples of size timeWindow
    # NOTE: each example contains only one sensor modality
    idx = 0
    timeWindowCounter = 0
    while idx < nbSampEeg - timeWindow + 1:
        windowData = eegData[idx:idx+timeWindow]
        for modIdx in range(nbModEeg):
            eegFramesArray[timeWindowCounter] = windowData[:,modIdx]
            timeWindowCounter += 1
        # Iterate
        idx += stride 

    # Building the time windows of the CO-Methane dataset
    coFramesArray = np.empty((nbCoFrames*nbModCo,timeWindow),dtype=np.float32)

    # Iteration on the CO-Methane data to build examples of size timeWindow
    # NOTE: each example contains only one sensor modality
    idx = 0
    timeWindowCounter = 0
    while idx < nbSampCo - timeWindow + 1:
        windowData = coData[idx:idx+timeWindow]
        for modIdx in range(nbModCo):
            coFramesArray[timeWindowCounter] = windowData[:,modIdx]
            timeWindowCounter += 1
        # Iterate
        idx += stride    

    # Building the time windows of the CO-Methane dataset
    methFramesArray = np.empty((nbMethFrames*nbModMeth,timeWindow),dtype=np.float32)

    # Iteration on the CO-Methane data to build examples of size timeWindow
    # NOTE: each example contains only one sensor modality
    idx = 0
    timeWindowCounter = 0
    while idx < nbSampMeth - timeWindow + 1:
        windowData = methData[idx:idx+timeWindow]
        for modIdx in range(nbModMeth):
            methFramesArray[timeWindowCounter] = windowData[:,modIdx]
            timeWindowCounter += 1
        # Iterate
        idx += stride 

    # Building the time windows of the energy-prediction dataset
    energyFrameArray = np.empty((nbEnergyFrames*nbModEnergy,timeWindow),dtype=np.float32)

    # Iteration on the energy-use data to build examples of size timeWindow
    # NOTE: each example contains only one sensor modality
    idx = 0
    timeWindowCounter = 0
    while idx < nbSampEnergy - timeWindow + 1:
        windowData = energyData[idx:idx+timeWindow]
        for modIdx in range(nbModEnergy):
            energyFrameArray[timeWindowCounter] = windowData[:,modIdx]
            timeWindowCounter += 1
        # Iterate
        idx += stride 

    # Building the time windows of the OPPORTUNITY dataset
    oppFrameArray = np.empty((nbOppFrames*nbModOpp,timeWindow),dtype=np.float32)

    # Iteration on the OPPORTUNITY data to build examples of size timeWindow
    # NOTE: each example contains only one sensor modality
    idx = 0
    timeWindowCounter = 0
    while idx < nbSampOpp - timeWindow + 1:
        windowData = oppData[idx:idx+timeWindow]
        for modIdx in range(nbModOpp):
            oppFrameArray[timeWindowCounter] = windowData[:,modIdx]
            timeWindowCounter += 1
        # Iterate
        idx += stride 

    # Concantenate the data
    print('Data concatenation ...')
    print('Concatenating:')
    print('    - %d examples from the OPPORTUNITY dataset' % (nbOppFrames*nbModOpp))
    print('    - %d examples from the gas-mixture dataset' % (nbMethFrames*nbModMeth+nbCoFrames*nbModCo))
    print('    - %d examples from the EEG dataset' % (nbEegFrames*nbModEeg))
    print('    - %d examples from the Energy-appliance dataset' % (nbEnergyFrames*nbModEnergy))

    #st()

    if datasets == 'all':
        eegLabels = np.tile([7],nbEegFrames*nbModEeg)
        methLabels = np.repeat(np.asarray([8]*2+[9]*16),nbMethFrames)
        coLabels = np.tile(np.asarray([8]*2+[9]*16),nbCoFrames)
        energyLabels = np.repeat(np.asarray([10,10,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,13,12,14,15,11]),nbEnergyFrames)
        oppLabels = np.repeat(np.asarray([0]*30+[1,1,1,2,2,2,3,3,3]*5+[4,4,4,1,1,1,1,1,1,5,5,5,5,5,5,6]*2),nbOppFrames) 
        fullData = np.concatenate((eegFramesArray,coFramesArray,methFramesArray,oppFrameArray,energyFrameArray))
        fullLabels = np.concatenate((eegLabels,coLabels,methLabels,oppLabels,energyLabels))

    elif datasets == 'opportunity':
        fullLabels = np.repeat(np.asarray([0]*30+[1,1,1,2,2,2,3,3,3]*5+[4,4,4,1,1,1,1,1,1,5,5,5,5,5,5,6]*2),nbOppFrames) 
        fullData = oppFrameArray

    elif datasets == 'eeg':
        eegLabels = np.tile([0],nbEegFrames*nbModEeg)
        methLabels = np.repeat(np.asarray([1]*18),nbMethFrames)
        coLabels = np.tile(np.asarray([1]*18),nbCoFrames)
        energyLabels = np.repeat(np.asarray([1]*26),nbEnergyFrames) 
        oppLabels = np.repeat(np.asarray([1]*107),nbOppFrames)
        fullData = np.concatenate((eegFramesArray,coFramesArray,methFramesArray,oppFrameArray,energyFrameArray))
        fullLabels = np.concatenate((eegLabels,coLabels,methLabels,oppLabels,energyLabels))

    elif datasets == 'gas-mixture':
        methLabels = np.repeat(np.asarray([0]*2+[1]*16),nbMethFrames)
        coLabels = np.tile(np.asarray([0]*2+[1]*16),nbCoFrames)
        fullData = np.concatenate((coFramesArray,methFramesArray))
        fullLabels = np.concatenate((coLabels,methLabels))

    elif datasets == 'energy':
        fullData = energyFrameArray
        fullLabels = np.repeat(np.asarray([0,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,3,2,4,5,1]),nbEnergyFrames)

    elif datasets == 'no-opportunity':
        eegLabels = np.tile([0],nbEegFrames*nbModEeg)
        methLabels = np.repeat(np.asarray([1]*2+[2]*16),nbMethFrames)
        coLabels = np.tile(np.asarray([1]*2+[2]*16),nbCoFrames)
        energyLabels = np.repeat(np.asarray([3,3,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,6,5,7,8,4]),nbEnergyFrames) 
        fullData = np.concatenate((eegFramesArray,coFramesArray,methFramesArray,energyFrameArray))
        fullLabels = np.concatenate((eegLabels,coLabels,methLabels,energyLabels))

    elif datasets == 'no-gas-mixture':
        eegLabels = np.tile([7],nbEegFrames*nbModEeg)
        energyLabels = np.repeat(np.asarray([8,8,9,10,9,10,9,10,9,10,9,10,9,10,9,10,9,10,9,10,9,11,10,12,13,9]),nbEnergyFrames)
        oppLabels = np.repeat(np.asarray([0]*30+[1,1,1,2,2,2,3,3,3]*5+[4,4,4,1,1,1,1,1,1,5,5,5,5,5,5,6]*2),nbOppFrames) 
        fullData = np.concatenate((eegFramesArray,oppFrameArray,energyFrameArray))
        fullLabels = np.concatenate((eegLabels,oppLabels,energyLabels))


    assert len(fullData) == len(fullLabels)

    # Shuffle examples
    fullData, fullLabels, _ = shuffleInUnisson(fullData,fullLabels)

    # Save data and labels in the result folder as .npy files 
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    np.save(savePath+'/mixedData_t'+str(timeWindow)+'_s'+str(stride)+'_'+datasets+'.npy',fullData)
    np.save(savePath+'/mixedLabels_t'+str(timeWindow)+'_s'+str(stride)+'_'+datasets+'.npy',fullLabels)
        
    print('Mixed dataset saved in folder %s' % (savePath))



#------------------------------------------------------------------------------------------------
# Methods for the CogAge dataset (target)
#------------------------------------------------------------------------------------------------

# Small utilitary function to filter between state and behavioral examples
def filterLabels(labels,labelsToFilter):
    idxToKeep = []
    for idx in range(len(labels)):
        if labels[idx] in labelsToFilter:
            idxToKeep.append(idx)

    return idxToKeep


#--------------------------------------------------------------------------------
# Methods for the DEAP dataset (target)
#--------------------------------------------------------------------------------
### Build time windows of data from the DEAP dataset
def buildDeapDataset(dataPathList,timeWindow,stride,savePath,trainingSetIdx,testingSetIdx):
    
    # Listing the data folder contents
    dataFileList = os.listdir(dataPathList)
    dataFileList = [e for e in dataFileList if '.dat' in e]
    dataFileList = sorted(dataFileList)
    
    # Determining the size of the training and testing sets
    deapNbSubjects = 32
    nbTrainSubjects = len(trainingSetIdx)

    tmpData = np.load(dataPathList+'/'+dataFileList[0])

    nbRuns, nbSensors, nbTimestamps = tmpData['data'].shape

    nbDataFramesPerRun = int(floor((nbTimestamps-timeWindow)/stride)+1)
    
    trainingData = np.zeros((nbDataFramesPerRun*nbRuns*nbTrainSubjects,timeWindow,nbSensors),dtype=np.float32)
    trainingLabels = np.zeros((nbDataFramesPerRun*nbRuns*nbTrainSubjects,4),dtype=np.float32) # Columns in order: valence, arousal, dominance, liking
    testingData = np.zeros((nbDataFramesPerRun*nbRuns*1,timeWindow,nbSensors),dtype=np.float32)
    testingLabels = np.zeros((nbDataFramesPerRun*nbRuns*1,4),dtype=np.float32) # Columns in order: valence, arousal, dominance, liking

    print('%d subjects in the training set (%d data frames)' % (nbTrainSubjects,nbDataFramesPerRun*nbRuns*nbTrainSubjects))
    print('%d subjects in the testing set (%d data frames)' % (1,nbDataFramesPerRun*nbRuns*1))
    
    # Building the time windows of data
    print('Building data frames with parameters T = %d and sigma = %d ...' % (timeWindow,stride))

    trainFrameCounter = 0
    testFrameCounter = 0

    for trainIdx in range(1,deapNbSubjects+1):
        
        dataDict = np.load(dataPathList+'/s'+str(trainIdx).zfill(2)+'.dat')
        data = dataDict['data']
        labels = dataDict['labels']

        if trainIdx in trainingSetIdx:

            for runIdx in range(nbRuns):
                
                runCounter = 0
                labelRun = labels[runIdx] # Vector of size 4
                dataRun = data[runIdx] # Matrix of size nbSensors x nbTimestamps

                while runCounter < nbTimestamps-timeWindow+1:

                    trainingData[trainFrameCounter] = np.transpose(dataRun[:,runCounter:runCounter+timeWindow])
                    trainingLabels[trainFrameCounter] = labelRun
                    trainFrameCounter += 1
                    runCounter += stride

        elif trainIdx in testingSetIdx:

            for runIdx in range(nbRuns):
                
                runCounter = 0
                labelRun = labels[runIdx] # Matrix of size nbSensors x nbTimestamps
                dataRun = data[runIdx] # Vector of size 4

                while runCounter < nbTimestamps-timeWindow+1:

                    testingData[testFrameCounter] = np.transpose(dataRun[:,runCounter:runCounter+timeWindow])
                    testingLabels[testFrameCounter] = labelRun
                    testFrameCounter += 1
                    runCounter += stride 

    # Save data and labels in the result folder as .npy files 
    np.save(savePath+'/x_train.npy',trainingData)
    np.save(savePath+'/y_train.npy',trainingLabels)
    np.save(savePath+'/x_test.npy',testingData)
    np.save(savePath+'/y_test.npy',testingLabels)
        
    print('DEAP training and testing sets saved in folder %s' % (savePath))


### Convert the DEAP regression labels to categorical labels for a 3-class problem
# <3: negative
# 3<=.<7: neutral
# >=7 : positive
def regressionToClassification3Labels(pathToLabelArray,savePath,nameExtension=''):
    
    # Loading the regression labels array
    regressionLabels = np.load(pathToLabelArray)

    # Determine number of examples
    nbExamples = len(regressionLabels)

    valenceLabels = -1*np.ones((nbExamples),dtype=int)
    arousalLabels = -1*np.ones((nbExamples),dtype=int)
    dominanceLabels = -1*np.ones((nbExamples),dtype=int)
    likingLabels = -1*np.ones((nbExamples),dtype=int)

    # Determination of categorical labels
    for idx in range(nbExamples):

        if regressionLabels[idx,0] <3:
            valenceLabels[idx] = 0    
        elif regressionLabels[idx,0] >=7:
            valenceLabels[idx] = 2
        else:
            valenceLabels[idx] = 1

        if regressionLabels[idx,1] <3:
            arousalLabels[idx] = 0
        elif regressionLabels[idx,1] >=7:
            arousalLabels[idx] = 2
        else:
            arousalLabels[idx] = 1

        if regressionLabels[idx,2] <3:
            dominanceLabels[idx] = 0
        elif regressionLabels[idx,2] >=7:
            dominanceLabels[idx] = 2
        else:
            dominanceLabels[idx] = 1

        if regressionLabels[idx,3] <3:
            likingLabels[idx] = 0          
        elif regressionLabels[idx,3] >=7:
            likingLabels[idx] = 2
        else:
            likingLabels[idx] = 1

    # Save data and labels in the result folder as .npy files 
    np.save(savePath+'/y_valence'+nameExtension+'.npy',valenceLabels)
    np.save(savePath+'/y_arousal'+nameExtension+'.npy',arousalLabels)
    np.save(savePath+'/y_dominance'+nameExtension+'.npy',dominanceLabels)
    np.save(savePath+'/y_liking'+nameExtension+'.npy',likingLabels)
        

### Convert the DEAP regression labels to categorical labels for a 2-class problem
# <5: negative
# >=5: positive
def regressionToClassification2Labels(pathToLabelArray,savePath,nameExtension=''):
    
    # Loading the regression labels array
    regressionLabels = np.load(pathToLabelArray)

    # Determine number of examples
    nbExamples = len(regressionLabels)

    valenceLabels = -1*np.ones((nbExamples),dtype=int)
    arousalLabels = -1*np.ones((nbExamples),dtype=int)
    dominanceLabels = -1*np.ones((nbExamples),dtype=int)
    likingLabels = -1*np.ones((nbExamples),dtype=int)

    # Determination of categorical labels
    for idx in range(nbExamples):

        if regressionLabels[idx,0] < 5:
            valenceLabels[idx] = 0
        elif regressionLabels[idx,0] >= 5:
            valenceLabels[idx] = 1

        if regressionLabels[idx,1] < 5:
            arousalLabels[idx] = 0
        elif regressionLabels[idx,1] >= 5:
            arousalLabels[idx] = 1

        if regressionLabels[idx,2] < 5:
            dominanceLabels[idx] = 0
        elif regressionLabels[idx,2] >= 5:
            dominanceLabels[idx] = 1

        if regressionLabels[idx,3] < 5:
            likingLabels[idx] = 0
        elif regressionLabels[idx,3] >= 5:
            likingLabels[idx] = 1

    # Save data and labels in the result folder as .npy files 
    np.save(savePath+'/y_valence'+nameExtension+'.npy',valenceLabels)
    np.save(savePath+'/y_arousal'+nameExtension+'.npy',arousalLabels)
    np.save(savePath+'/y_dominance'+nameExtension+'.npy',dominanceLabels)
    np.save(savePath+'/y_liking'+nameExtension+'.npy',likingLabels)


### Build the 10 CV-fold DEAP dataset
def buildDeap10CvFolds(dataPathList,timeWindow,stride,savePath):
    
    # Listing the data folder contents
    dataFileList = os.listdir(dataPathList)
    dataFileList = [e for e in dataFileList if '.dat' in e]
    dataFileList = sorted(dataFileList)
    
    # Determining the size of the full dataset
    nbSubjects = len(dataFileList)
    deapNbSubjects = 32
    assert nbSubjects == deapNbSubjects

    tmpData = np.load(dataPathList+'/'+dataFileList[0])
    nbRuns, nbSensors, nbTimestamps = tmpData['data'].shape

    nbDataFramesPerRun = int(floor((nbTimestamps-timeWindow)/stride)+1)
    totalNbExamples = nbDataFramesPerRun*nbRuns*nbSubjects
    
    allData = np.zeros((totalNbExamples,timeWindow,nbSensors),dtype=np.float32)
    allLabels = np.zeros((totalNbExamples,4),dtype=np.float32) # Columns in order: valence, arousal, dominance, liking
    
    # Building the time windows of data
    print('Building data frames with parameters T = %d and sigma = %d for all subjects of the DEAP dataset ...' % (timeWindow,stride))

    frameCounter = 0

    for subjectIdx in range(1,nbSubjects+1):
        
        dataDict = np.load(dataPathList+'/s'+str(subjectIdx).zfill(2)+'.dat')
        data = dataDict['data']
        labels = dataDict['labels']

        for runIdx in range(nbRuns):
                
            runCounter = 0
            labelRun = labels[runIdx] # Vector of size 4
            dataRun = data[runIdx] # Matrix of size nbSensors x nbTimestamps = 40 x 8064

            while runCounter < nbTimestamps-timeWindow+1:

                allData[frameCounter] = np.transpose(dataRun[:,runCounter:runCounter+timeWindow])
                allLabels[frameCounter] = labelRun
                frameCounter += 1
                runCounter += stride

    # Shuffling the data and labels in unisson
    print('Shuffling data and labels ...')
    allData, allLabels, _ = shuffleInUnisson(allData,allLabels)

    # Building the 10 folds of the dataset
    print('Preparing the 10 cross validation folds ...')
    foldLength = int(floor(totalNbExamples/10))

    for foldIdx in range(1,11):
        foldSavePath = savePath+'/'+str(foldIdx).zfill(2)+'/'
        if not os.path.exists(foldSavePath):
            os.makedirs(foldSavePath)
        if foldIdx != 10:
            trainingData = allData[list(range((foldIdx-1)*foldLength))+list(range(foldIdx*foldLength,totalNbExamples))]
            trainingLabels = allLabels[list(range((foldIdx-1)*foldLength))+list(range(foldIdx*foldLength,totalNbExamples))]
            testingData = allData[(foldIdx-1)*foldLength:foldIdx*foldLength]
            testingLabels = allLabels[(foldIdx-1)*foldLength:foldIdx*foldLength]
        else:
            trainingData = allData[list(range((foldIdx-1)*foldLength))]
            trainingLabels = allLabels[list(range((foldIdx-1)*foldLength))]
            testingData = allData[(foldIdx-1)*foldLength:]
            testingLabels = allLabels[(foldIdx-1)*foldLength:]
        np.save(foldSavePath+'/x_train_'+str(foldIdx).zfill(2)+'.npy',trainingData)
        np.save(foldSavePath+'/y_train_'+str(foldIdx).zfill(2)+'.npy',trainingLabels)
        np.save(foldSavePath+'/x_test_'+str(foldIdx).zfill(2)+'.npy',testingData)
        np.save(foldSavePath+'/y_test_'+str(foldIdx).zfill(2)+'.npy',testingLabels)
        
    print('DEAP 10-fold cross validation data saved in folder %s' % (savePath))


#--------------------------------------------------------------------------------
# Methods for analysis of activations during the training of the model
#--------------------------------------------------------------------------------

### Get the activations of the layer of one specific model, given some input examples X_batch
def getActivations(model, layer, X_batch): # layer: index integer
    get_activations = K.function(
        [model.layers[0].input,
        model.layers[1].input,
        model.layers[2].input,
        model.layers[3].input,
        model.layers[4].input,
        model.layers[5].input,
        model.layers[6].input,
        model.layers[7].input,
        K.learning_phase()], 
        [model.layers[layer].output[1:],]) # Multi-input case
    activations = get_activations(X_batch+[0])
    return activations, model.layers[layer].name


### Get the derivatives of the loss wrt weights given one model and a labelled dataset
def getGradients(model,inputData,inputLabels):
    # Inspired from: https://github.com/keras-team/keras/issues/2226
    weights = model.trainable_weights
    optimizer = model.optimizer

    gradients = optimizer.get_gradients(model.total_loss, weights) # gradient tensors
    input_tensors = [
        model.inputs[0], 
        model.inputs[1],
        model.inputs[2],
        model.inputs[3],
        model.inputs[4],
        model.inputs[5],
        model.inputs[6],
        model.inputs[7],
        model.sample_weights[0],
        model.targets[0],
        K.learning_phase()]
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    inputs = inputData + [np.ones(len(inputData[0])),inputLabels,0]
    grads = get_gradients(inputs)
    return grads


### Train and model and save activations and gradients every epochStride epochs
def trainDnnAndSaveActivationsGradients(model,batchSize,epochs,epochStride,x_train,y_train,x_test,y_test,pathToReport=None,pathToCheckpoint=None):

    # Tensorboard reports and checkpoints
    if pathToCheckpoint is None and pathToReport is None:
        callbacksList = None
    else:
        callbacksList = []

    if pathToCheckpoint is not None:
        if not os.path.exists(pathToCheckpoint):
            os.makedirs(pathToCheckpoint)
        checkpoint = keras.callbacks.ModelCheckpoint(pathToCheckpoint+'checkpoint.hdf5',monitor='val_acc',save_best_only=True,save_weights_only=False)
        callbacksList += [checkpoint]

    # Get the activations and gradients of each layer every epochStride epochs
    nbTrainingLoops = int(epochs/epochStride)
    nbLayers = len(model.layers)
    averageActivations = np.zeros((nbLayers,nbTrainingLoops+1),dtype=np.float32)
    stdActivations = np.zeros((nbLayers,nbTrainingLoops+1),dtype=np.float32)

    # Get the activations for every layer before training
    layerNameList = []
    trainableLayerNameList = []

    for layerIdx in range(nbLayers):
        activations, layerName = getActivations(model,layerIdx,x_test)
        averageActivations[layerIdx,0] = np.mean(activations)
        stdActivations[layerIdx,0] = np.std(activations)
        layerNameList += [layerName]
        if model.layers[layerIdx].count_params() > 0:
            trainableLayerNameList += [layerName]

    # Array of average and std of gradients during training
    # Note: stores information about gradients for weights and biases
    nbTrainableLayers = len(trainableLayerNameList)
    averageGradients = np.zeros((2,nbTrainableLayers,nbTrainingLoops+1),dtype=np.float32)
    stdGradients = np.zeros((2,nbTrainableLayers,nbTrainingLoops+1),dtype=np.float32)

    # Get the gradients before any training
    layerGradients = getGradients(model,x_test,y_test)

    assert len(layerGradients) == 2*nbTrainableLayers

    for layerIdx in range(nbTrainableLayers):
        averageGradients[0,layerIdx,0] = np.mean(layerGradients[2*layerIdx]) # Gradients of W
        averageGradients[1,layerIdx,0] = np.mean(layerGradients[2*layerIdx+1]) # Gradients of b
        stdGradients[0,layerIdx,0] = np.std(layerGradients[2*layerIdx])
        stdGradients[1,layerIdx,0] = np.std(layerGradients[2*layerIdx+1])
   
    # Initiate training and get activations and gradient every epochStride epochs 
    print('Initiating the training phase ...')
    print('Saving activations and gradients of %d layer(s) every %d epochs' % (nbLayers,epochStride))

    for trainIdx in range(nbTrainingLoops):

        print('    Training between epochs %d/%d and %d/%d ...' % (epochStride*trainIdx,epochs,epochStride*(trainIdx+1),epochs))

        # Train the model
        model.fit(x_train, y_train,
                batch_size=batchSize,
                epochs=epochStride,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=callbacksList)

        # Get activation values for each layer
        print('        Computing activations and gradients of the model layers ...')
        for layerIdx in range(nbLayers):
            activations, layerName = getActivations(model,layerIdx,x_test)
            averageActivations[layerIdx,trainIdx+1] = np.mean(activations)
            stdActivations[layerIdx,trainIdx+1] = np.std(activations)

        layerGradients = getGradients(model,x_test,y_test)

        for layerIdx in range(nbTrainableLayers):
            averageGradients[0,layerIdx,trainIdx+1] = np.mean(layerGradients[2*layerIdx]) # Gradients of W
            averageGradients[1,layerIdx,trainIdx+1] = np.mean(layerGradients[2*layerIdx+1]) # Gradients of b
            stdGradients[0,layerIdx,trainIdx+1] = np.std(layerGradients[2*layerIdx])
            stdGradients[1,layerIdx,trainIdx+1] = np.std(layerGradients[2*layerIdx+1])


    return averageActivations, stdActivations, layerNameList, averageGradients, stdGradients, trainableLayerNameList


### Compute the Jacobian matrix of a trained model
def computeJacobian(model,data):

    # Note: check placement of this
    sess = tf.InteractiveSession()
    K.set_session(sess)
    sess.run(tf.initialize_all_variables())

    # Prepare the dictionnary containing the Jacobina with respect to all inputs
    jacobian = {}
    nbInputComponents = len(data)
    nbOutputComponents = model.output.get_shape().as_list()[1]

    feedDict = {}

    for idx in range(nbInputComponents):
        feedDict[model.input[idx]] = np.expand_dims(data[idx],axis=0)

    # Initialisation of the Jacobians
    for idx in range(nbInputComponents):
        jacobian[idx] = np.zeros((nbOutputComponents,)+data[idx].shape,dtype=np.float32) # Excludes the last dimension = nbSensor channels

    # Jacobian computation
    for idx in range(nbOutputComponents):
        gradFunction = tf.gradients(model.output[:,idx], model.input)
        gradients = sess.run(gradFunction, feed_dict=feedDict)
        for idx2 in range(len(gradients)): # len(gradients) should be equal to the number of input components
            jacobian[idx2][idx] = gradients[idx2][0,:]

    #res = np.array(jacobian)
    sess.close()
    return jacobian



################################################################################################################################
### Loading the trained model and data
################################################################################################################################
# # Note: for some reason, the traditional loading of architecture + weights doesn't work
# modelPath = '/hdd/test-PMC/models/Multichannel-DNN/CogAge/multichannel-'+transfer+'-'+classification+'.h5'
# idx = 0
# model = load_model(modelPath,custom_objects={'idx':idx})

# # Loading the structure of the network (contained in JSON files) followed by loading weights
# jsonFilePath = '/hdd/test-PMC/models/Multichannel-DNN/CogAge/multichannel-'+transfer+'-'+classification+'.json'
#weightFilePath = '/hdd/test-PMC/models/Multichannel-DNN/CogAge/multichannel-weights-'+transfer+'-'+classification+'.h5'

# jsonFile = open(jsonFilePath,'r')
# loadedJsonModel = jsonFile.read()
# jsonFile.close()
# model = model_from_json(loadedJsonModel)
# model.load_weights(weightFilePath)
def loadTrainedMultichannelDnn(weightFilePath,testingAcc,testingGrav,testingGyro,testingLinAcc,testingMSAcc,testingMSGyro,testingJinsAcc,testingJinsGyro):

    smartphoneModel = {
        'name': 'CNN',
        'nb_conv_blocks' : 3,
        'nb_conv_kernels' : [10,10,10],
        'conv_kernels_size' : [(45,),(49,),(46,)],
        'pooling_size' : [(2,),(2,),(2,)],
        'activation' : 'relu',
        }

    msbandModel = {
        'name': 'CNN',
        'nb_conv_blocks' : 3,
        'nb_conv_kernels' : [10,10,10],
        'conv_kernels_size' : [(9,),(11,),(11,)],
        'pooling_size' : [(2,),(2,),(2,)],
        'activation' : 'relu',
        }

    jinsModel= {
       'name': 'CNN',
       'nb_conv_blocks' : 3,
       'nb_conv_kernels' : [10,10,10],
       'conv_kernels_size' : [(5,),(5,),(5,)],
       'pooling_size' : [(2,),(2,),(2,)],
       'activation' : 'relu',
      }

    denseSize = [2000,2000,2000]
    denseActivation = 'relu'
    nbClasses = 55

    smartphoneTimeWindow = testingAcc.shape[1] #800
    msbandTimeWindow = testingMSAcc.shape[1] #268
    jinsTimeWindow = testingJinsAcc.shape[1] # 80

    accTimeWindow = testingAcc.shape[1]
    gravTimeWindow = testingGrav.shape[1]
    gyroTimeWindow = testingGyro.shape[1]
    linAccTimeWindow = testingLinAcc.shape[1]
    #magnTimeWindow = testingMagn.shape[1]
    msAccTimeWindow = testingMSAcc.shape[1]
    msGyroTimeWindow = testingMSGyro.shape[1]
    jinsAccTimeWindow = testingJinsAcc.shape[1]
    jinsGyroTimeWindow = testingJinsGyro.shape[1]


    inputAcc = Input(shape=testingAcc.shape[1:])
    bnAcc = BatchNormalization(axis=2)(inputAcc)  
    accChannels = []
    for idx in range(testingAcc.shape[2]):
        accChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(accTimeWindow,1))(bnAcc))

    for idx in range(testingAcc.shape[2]):
        accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(accChannels[idx])
        accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(accChannels[idx])
        accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(accChannels[idx])
        accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(accChannels[idx])
        accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(accChannels[idx])
        accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(accChannels[idx])
        accChannels[idx] = Flatten()(accChannels[idx])

    inputGrav = Input(shape=testingGrav.shape[1:]) 
    bnGrav = BatchNormalization(axis=2)(inputGrav)
    gravChannels = []
    for idx in range(testingGrav.shape[2]):
        gravChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(gravTimeWindow,1))(bnGrav))

    for idx in range(testingGrav.shape[2]):
        gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(gravChannels[idx])
        gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(gravChannels[idx])
        gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(gravChannels[idx])
        gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(gravChannels[idx])
        gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(gravChannels[idx])
        gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(gravChannels[idx])
        gravChannels[idx] = Flatten()(gravChannels[idx])

    inputGyro = Input(shape=testingGyro.shape[1:]) 
    bnGyro = BatchNormalization(axis=2)(inputGyro)
    gyroChannels = []
    for idx in range(testingGyro.shape[2]):
        gyroChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(gyroTimeWindow,1))(bnGyro))

    for idx in range(testingGyro.shape[2]):
        gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(gyroChannels[idx])
        gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(gyroChannels[idx])
        gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(gyroChannels[idx])
        gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(gyroChannels[idx])
        gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(gyroChannels[idx])
        gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(gyroChannels[idx])
        gyroChannels[idx] = Flatten()(gyroChannels[idx])

    inputLinAcc = Input(shape=testingLinAcc.shape[1:]) 
    bnLinAcc = BatchNormalization(axis=2)(inputLinAcc)
    linAccChannels = []
    for idx in range(testingLinAcc.shape[2]):
        linAccChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(linAccTimeWindow,1))(bnLinAcc))

    for idx in range(testingLinAcc.shape[2]):
        linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(linAccChannels[idx])
        linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(linAccChannels[idx])
        linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(linAccChannels[idx])
        linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(linAccChannels[idx])
        linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(linAccChannels[idx])
        linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(linAccChannels[idx])
        linAccChannels[idx] = Flatten()(linAccChannels[idx])

    inputMSAcc = Input(shape=testingMSAcc.shape[1:]) 
    bnMsAcc = BatchNormalization(axis=2)(inputMSAcc)
    msAccChannels = []
    for idx in range(testingMSAcc.shape[2]):
        msAccChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(msAccTimeWindow,1))(bnMsAcc))

    for idx in range(testingMSAcc.shape[2]):
        msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][0],kernel_size=msbandModel['conv_kernels_size'][0],activation=msbandModel['activation'])(msAccChannels[idx])
        msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][0])(msAccChannels[idx])
        msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][1],kernel_size=msbandModel['conv_kernels_size'][1],activation=msbandModel['activation'])(msAccChannels[idx])
        msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][1])(msAccChannels[idx])
        msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][2],kernel_size=msbandModel['conv_kernels_size'][2],activation=msbandModel['activation'])(msAccChannels[idx])
        msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][2])(msAccChannels[idx])
        msAccChannels[idx] = Flatten()(msAccChannels[idx])

    inputMSGyro = Input(shape=testingMSGyro.shape[1:]) 
    bnMsGyro = BatchNormalization(axis=2)(inputMSGyro)
    msGyroChannels = []
    for idx in range(testingMSGyro.shape[2]):
        msGyroChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(msGyroTimeWindow,1))(bnMsGyro))

    for idx in range(testingMSGyro.shape[2]):
        msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][0],kernel_size=msbandModel['conv_kernels_size'][0],activation=msbandModel['activation'])(msGyroChannels[idx])
        msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][0])(msGyroChannels[idx])
        msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][1],kernel_size=msbandModel['conv_kernels_size'][1],activation=msbandModel['activation'])(msGyroChannels[idx])
        msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][1])(msGyroChannels[idx])
        msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][2],kernel_size=msbandModel['conv_kernels_size'][2],activation=msbandModel['activation'])(msGyroChannels[idx])
        msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][2])(msGyroChannels[idx])
        msGyroChannels[idx] = Flatten()(msGyroChannels[idx])

    inputJinsAcc = Input(shape=testingJinsAcc.shape[1:]) 
    bnJinsAcc = BatchNormalization(axis=2)(inputJinsAcc)
    jinsAccChannels = []
    for idx in range(testingJinsAcc.shape[2]):
        jinsAccChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(jinsTimeWindow,1))(bnJinsAcc))

    for idx in range(testingJinsAcc.shape[2]):
        jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][0],kernel_size=jinsModel['conv_kernels_size'][0],activation=jinsModel['activation'])(jinsAccChannels[idx])
        jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][0])(jinsAccChannels[idx])
        jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][1],kernel_size=jinsModel['conv_kernels_size'][1],activation=jinsModel['activation'])(jinsAccChannels[idx])
        jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][1])(jinsAccChannels[idx])
        jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][2],kernel_size=jinsModel['conv_kernels_size'][2],activation=jinsModel['activation'])(jinsAccChannels[idx])
        jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][2])(jinsAccChannels[idx])
        jinsAccChannels[idx] = Flatten()(jinsAccChannels[idx])


    inputJinsGyro = Input(shape=testingJinsGyro.shape[1:]) 
    bnJinsGyro = BatchNormalization(axis=2)(inputJinsGyro)
    jinsGyroChannels = []
    for idx in range(testingJinsGyro.shape[2]):
        jinsGyroChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(jinsTimeWindow,1))(bnJinsGyro))

    for idx in range(testingJinsGyro.shape[2]):
        jinsGyroChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][0],kernel_size=jinsModel['conv_kernels_size'][0],activation=jinsModel['activation'])(jinsGyroChannels[idx])
        jinsGyroChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][0])(jinsGyroChannels[idx])
        jinsGyroChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][1],kernel_size=jinsModel['conv_kernels_size'][1],activation=jinsModel['activation'])(jinsGyroChannels[idx])
        jinsGyroChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][1])(jinsGyroChannels[idx])
        jinsGyroChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][2],kernel_size=jinsModel['conv_kernels_size'][2],activation=jinsModel['activation'])(jinsGyroChannels[idx])
        jinsGyroChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][2])(jinsGyroChannels[idx])
        jinsGyroChannels[idx] = Flatten()(jinsGyroChannels[idx])

    # Concatenation of all sensor channel outputs
    concatenation = concatenate(accChannels+gravChannels+gyroChannels+linAccChannels+msAccChannels+msGyroChannels+jinsAccChannels+jinsGyroChannels,axis=1)

    # Add a dense layer
    dense = Dense(denseSize[0],activation=denseActivation)(concatenation)
    if len(denseSize) > 1:
        for idx in range(1,len(denseSize)):
            dense = Dense(denseSize[idx],activation=denseActivation)(dense)

    # Softmax layer
    outputLayer = Dense(nbClasses,activation='softmax')(dense)
    model = Model(inputs=[inputAcc,inputGrav,inputGyro,inputLinAcc,inputMSAcc,inputMSGyro,inputJinsAcc,inputJinsGyro],outputs=outputLayer)


    ### Model compilation
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=0.1),
                  #optimizer=keras.optimizers.Adam(lr=learningRate),
                  metrics=['acc',fmeasure])

    model.load_weights(weightFilePath)
    #print('*******************************************')
    #print('Model correctly loaded!')
    #print('*******************************************')

    return model
