##############################################################################################
# Script to train and evaluate a Multichannel DNN on the CogAge datset
# Used for the study presented in [*].
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification
# F.Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek 
##############################################################################################

from utilitary import *

#------------------------------------------------------------------------------------------------
# Hyper-parameters
#------------------------------------------------------------------------------------------------
### Meta parameters
transfer = 'cnn-transfer' # 'tto' (= no transfer) or 'cnn-transfer' or 'vae-transfer'
classification = 'state' # 'state', 'blho', 'bbh'
targetDataPath = '../data/target/CogAge/'+classification+'/' # Path to the CogAge dataset
#foldIdx = 1
#targetDataPath = '../data/target/CogAge-SI/'+classification+'/'+str(foldIdx)+'/' # Path to the CogAge dataset
modelPath = '../mDNN/' # Path to save the trained Multimodal DNN
saveName = 'mCnn-CogAge-'+transfer+'-'+classification
verbose = 1 # Enable (1) or disable (0) training details 

### Training parameters
batchSize = 50
learningRate = 0.5 # NOTE: tested: 1, 0.5, 0.1, 0.01
epochs = 150
targetDataProportion = 1 # What percentage of the target domain should be used for the training? Default = 1. Tested values: 0.05, 0.25, 0.50, 0.75, 1 

### single-channel DNN parameters
# Note: input sizes of each sensor channels:
#  - smartphone: 800
#  - magnetometer: 200
#  - MS Band: 268
#  - JINS glasses: 80
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


### Parameters of the softmax MLP of the Multichannel DNN
denseSize = [2000,2000,2000]
denseActivation = 'relu'

### Path to the trained single-channel DNN
if transfer is not 'vae-transfer':
    trainedSmartphoneModelPath = '../sDNN/cnn-transfer-CogAgeSmartphone.h5'
    trainedMSBandModelPath = '../sDNN/cnn-transfer-CogAgeMSBand.h5'
    trainedJinsModelPath = '../sDNN/cnn-transfer-CogAgeJins.h5'
else:
    trainedSmartphoneModelPath = '../sDNN/cnnVae-CogAgeSmartphone.h5'
    trainedMSBandModelPath = '../sDNN/cnnVae-CogAgeMSBand.h5'
    trainedJinsModelPath = '../sDNN/cnnVae-CogAgeJins.h5'

### [Optional] Path to save the class average precisions
# Leave empty to disable
apPath = ''



#----------------------------------------------------------------------------------------------
# multichannelCnnClassify: train and evaluate a multichannel CNN on the CogAge dataset
#----------------------------------------------------------------------------------------------
def multichannelCnnClassify(
    targetDataPath,
    transfer,
    classification,
    batchSize,
    learningRate,
    epochs,
    smartphoneModel,
    msbandModel,
    jinsModel,
    denseSize,
    denseActivation,
    trainedSmartphoneModelPath,
    trainedMSBandModelPath,
    trainedJinsModelPath,
    saveName='',
    modelPath='',
    apPath='',
    verbose=verbose,
    targetDataProportion=1
    ):

    start = time.time()
 
    print('#############################################################')
    print('Training of the multichannel CNN on the CogAge dataset')
    print('Classification problem: %s' % (classification.upper()))
    print('Transfer technique: %s' % (transfer.upper()))
    print('#############################################################')

    print('----------------------------------------------------------------------------------------')

    ### load the data
    print('Loading the CogAge data (regular train/test splitting) ...')
    trainingAcc = np.load(targetDataPath+'/training/trainAccelerometer.npy')
    trainingGrav = np.load(targetDataPath+'/training/trainGravity.npy')
    trainingGyro = np.load(targetDataPath+'/training/trainGyroscope.npy')
    trainingLinAcc = np.load(targetDataPath+'/training/trainLinearAcceleration.npy')
    #trainingMagn = np.load(targetDataPath+'/training/trainMagnetometer.npy')
    trainingMSAcc = np.load(targetDataPath+'/training/trainMSAccelerometer.npy')
    trainingMSGyro = np.load(targetDataPath+'/training/trainMSGyroscope.npy')
    trainingLabels = np.load(targetDataPath+'/training/trainLabels.npy')

    testingAcc = np.load(targetDataPath+'/testing/testAccelerometer.npy')
    testingGrav = np.load(targetDataPath+'/testing/testGravity.npy')
    testingGyro = np.load(targetDataPath+'/testing/testGyroscope.npy')
    testingLinAcc = np.load(targetDataPath+'/testing/testLinearAcceleration.npy')
    #testingMagn = np.load(targetDataPath+'/testing/testMagnetometer.npy')
    testingMSAcc = np.load(targetDataPath+'/testing/testMSAccelerometer.npy')
    testingMSGyro = np.load(targetDataPath+'/testing/testMSGyroscope.npy')
    testingLabels = np.load(targetDataPath+'/testing/testLabels.npy')

    trainingJinsAcc = np.load(targetDataPath+'/training/trainJinsAccelerometer.npy')
    testingJinsAcc = np.load(targetDataPath+'/testing/testJinsAccelerometer.npy')
    trainingJinsGyro = np.load(targetDataPath+'/training/trainJinsGyroscope.npy')
    testingJinsGyro = np.load(targetDataPath+'/testing/testJinsGyroscope.npy')

    smartphoneTimeWindow = trainingAcc.shape[1] #800
    msbandTimeWindow = trainingMSAcc.shape[1] #268
    jinsTimeWindow = trainingJinsAcc.shape[1] # 80

    # Downsample the training set if needed
    assert targetDataProportion >0
    assert targetDataProportion <=1

    if targetDataProportion <1:

        print('Downsampling the training set to %.2f %%' % (100*targetDataProportion))
        random.seed(1) # Fix the random seed to always get the same data downsampling

        # Shuffle in unisson training data and labels
        trainingAcc, trainingLabels, permutation = shuffleInUnisson(trainingAcc,trainingLabels)
        trainingGrav = trainingGrav[permutation]
        trainingGyro = trainingGyro[permutation]
        trainingLinAcc = trainingLinAcc[permutation]
        trainingMSAcc = trainingMSAcc[permutation]
        trainingMSGyro = trainingMSGyro[permutation]
        trainingJinsAcc = trainingJinsAcc[permutation]
        trainingJinsGyro = trainingJinsGyro[permutation]

        # Downsample the training set
        upperBound = int(floor(len(trainingLabels)*targetDataProportion))

        # Due to the very low number of training examples for state classification, the building of the training set is done by picking N examples per class to ensure that examples from all classes are selected.
        if classification == 'state':
            nbClasses = len(set(trainingLabels))
            indices = []
            nbExamplesPerClass = int(floor(upperBound/nbClasses))
            for classIdx in range(nbClasses):
                tmpIdx = [i for i,e in enumerate(trainingLabels) if e==classIdx]
                indices += tmpIdx[:nbExamplesPerClass]

            trainingLabels = trainingLabels[indices]
            trainingAcc = trainingAcc[indices]
            trainingGrav = trainingGrav[indices]
            trainingGyro = trainingGyro[indices]
            trainingLinAcc = trainingLinAcc[indices]
            trainingMSAcc = trainingMSAcc[indices]
            trainingMSGyro = trainingMSGyro[indices]
            trainingJinsAcc = trainingJinsAcc[indices]
            trainingJinsGyro = trainingJinsGyro[indices]

        else:
            trainingLabels = trainingLabels[:upperBound]
            trainingAcc = trainingAcc[:upperBound]
            trainingGrav = trainingGrav[:upperBound]
            trainingGyro = trainingGyro[:upperBound]
            trainingLinAcc = trainingLinAcc[:upperBound]
            trainingMSAcc = trainingMSAcc[:upperBound]
            trainingMSGyro = trainingMSGyro[:upperBound]
            trainingJinsAcc = trainingJinsAcc[:upperBound]
            trainingJinsGyro = trainingJinsGyro[:upperBound]

    # Reshape the data to make it suitable for keras formatting
    trainingAcc = np.expand_dims(trainingAcc,3)
    trainingGrav = np.expand_dims(trainingGrav,3)
    trainingGyro = np.expand_dims(trainingGyro,3)
    trainingLinAcc = np.expand_dims(trainingLinAcc,3)
    #trainingMagn = np.expand_dims(trainingMagn,3)
    trainingMSAcc = np.expand_dims(trainingMSAcc,3)
    trainingMSGyro = np.expand_dims(trainingMSGyro,3)

    testingAcc = np.expand_dims(testingAcc,3)
    testingGrav = np.expand_dims(testingGrav,3)
    testingGyro = np.expand_dims(testingGyro,3)
    testingLinAcc = np.expand_dims(testingLinAcc,3)
    #testingMagn = np.expand_dims(testingMagn,3)
    testingMSAcc = np.expand_dims(testingMSAcc,3)
    testingMSGyro = np.expand_dims(testingMSGyro,3)

    trainingJinsAcc = np.expand_dims(trainingJinsAcc,3)
    testingJinsAcc = np.expand_dims(testingJinsAcc,3)
    trainingJinsGyro = np.expand_dims(trainingJinsGyro,3)
    testingJinsGyro = np.expand_dims(testingJinsGyro,3)

    groundTruthLabels = testingLabels
    nbTestingExamples = len(testingLabels)

    if classification == 'state':
        nbClasses = 6
    else:
        nbClasses = 55

    assert nbClasses >= len(set(trainingLabels))
    assert nbClasses >= len(set(testingLabels))

    print('    %d training examples' % (len(trainingLabels)))
    print('    %d testing examples' % (nbTestingExamples))

    accTimeWindow = trainingAcc.shape[1]
    gravTimeWindow = trainingGrav.shape[1]
    gyroTimeWindow = trainingGyro.shape[1]
    linAccTimeWindow = trainingLinAcc.shape[1]
    #magnTimeWindow = trainingMagn.shape[1]
    msAccTimeWindow = trainingMSAcc.shape[1]
    msGyroTimeWindow = trainingMSGyro.shape[1]
    jinsAccTimeWindow = trainingJinsAcc.shape[1]
    jinsGyroTimeWindow = trainingJinsGyro.shape[1]

    # Convert class vectors to binary class matrices
    trainingLabels = keras.utils.to_categorical(trainingLabels, nbClasses)
    testingLabels = keras.utils.to_categorical(testingLabels, nbClasses)
    
    ### Loading weights of the single-channel DNNs
    if transfer == 'cnn-transfer':

        smartphoneSourceModel = cnn(
            inputShape=(smartphoneTimeWindow,1),
            nkerns=smartphoneModel['nb_conv_kernels'],
            filterSizes=smartphoneModel['conv_kernels_size'],
            poolSizes=smartphoneModel['pooling_size'],
            activation=smartphoneModel['activation'],
            nbClasses=nbModalitiesAllDatasets)
        smartphoneSourceModel.load_weights(trainedSmartphoneModelPath,by_name=True) 
        smartphoneLayersDict = dict([(layer.name,layer) for layer in smartphoneSourceModel.layers])
        smartphoneLayersWeights = []
        for idx in range(smartphoneModel['nb_conv_blocks']):
            smartphoneLayersWeights += [smartphoneLayersDict['conv'+str(idx+1)].get_weights()]

        msbandSourceModel = cnn(
            inputShape=(msbandTimeWindow,1),
            nkerns=msbandModel['nb_conv_kernels'],
            filterSizes=msbandModel['conv_kernels_size'],
            poolSizes=msbandModel['pooling_size'],
            activation=msbandModel['activation'],
            nbClasses=nbModalitiesAllDatasets)
        msbandSourceModel.load_weights(trainedMSBandModelPath,by_name=True) 
        msbandLayersDict = dict([(layer.name,layer) for layer in msbandSourceModel.layers])
        msbandLayersWeights = []
        for idx in range(msbandModel['nb_conv_blocks']):
            msbandLayersWeights += [msbandLayersDict['conv'+str(idx+1)].get_weights()]

        jinsSourceModel = cnn(
            inputShape=(jinsTimeWindow,1),
            nkerns=jinsModel['nb_conv_kernels'],
            filterSizes=jinsModel['conv_kernels_size'],
            poolSizes=jinsModel['pooling_size'],
            activation=jinsModel['activation'],
            nbClasses=nbModalitiesAllDatasets)
        jinsSourceModel.load_weights(trainedJinsModelPath,by_name=True) 
        jinsLayersDict = dict([(layer.name,layer) for layer in jinsSourceModel.layers])
        jinsLayersWeights = []
        for idx in range(jinsModel['nb_conv_blocks']):
            jinsLayersWeights += [jinsLayersDict['conv'+str(idx+1)].get_weights()]


    elif transfer == 'vae-transfer':

        smartphoneSourceModel = convAutoencoder(
            inputShape=(smartphoneTimeWindow,1),
            nkerns=smartphoneModel['nb_conv_kernels'],
            filterSizes=smartphoneModel['conv_kernels_size'],
            poolSizes=smartphoneModel['pooling_size'],
            activation=smartphoneModel['activation'])
        smartphoneSourceModel.load_weights(trainedSmartphoneModelPath,by_name=True)
        smartphoneLayersDict = dict([(layer.name,layer) for layer in smartphoneSourceModel.layers])
        smartphoneLayersWeights = []
        for idx in range(smartphoneModel['nb_conv_blocks']):
            smartphoneLayersWeights += [smartphoneLayersDict['encoder'+str(idx+1)].get_weights()]

        msbandSourceModel = convAutoencoder(
            inputShape=(msbandTimeWindow,1),
            nkerns=msbandModel['nb_conv_kernels'],
            filterSizes=msbandModel['conv_kernels_size'],
            poolSizes=msbandModel['pooling_size'],
            activation=msbandModel['activation'])
        msbandSourceModel.load_weights(trainedMSBandModelPath,by_name=True)
        msbandLayersDict = dict([(layer.name,layer) for layer in msbandSourceModel.layers])
        msbandLayersWeights = []
        for idx in range(msbandModel['nb_conv_blocks']):
            msbandLayersWeights += [msbandLayersDict['encoder'+str(idx+1)].get_weights()]

        jinsSourceModel = convAutoencoder(
        inputShape=(jinsTimeWindow,1),
        nkerns=jinsModel['nb_conv_kernels'],
        filterSizes=jinsModel['conv_kernels_size'],
        poolSizes=jinsModel['pooling_size'],
        activation=jinsModel['activation'])
        jinsSourceModel.load_weights(trainedJinsModelPath,by_name=True)
        jinsLayersDict = dict([(layer.name,layer) for layer in jinsSourceModel.layers])
        jinsLayersWeights = []
        for idx in range(jinsModel['nb_conv_blocks']):
            jinsLayersWeights += [jinsLayersDict['encoder'+str(idx+1)].get_weights()]

    ############################ Model definition ##############################
    print('\n')
    print('Building the multichannel CNN ...')

    inputAcc = Input(shape=trainingAcc.shape[1:])
    bnAcc = BatchNormalization(axis=2)(inputAcc)  
    accChannels = []
    for idx in range(trainingAcc.shape[2]):
        accChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(accTimeWindow,1))(bnAcc))
    if transfer != 'tto':
        for idx in range(trainingAcc.shape[2]):
            accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[0])(accChannels[idx])
            accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(accChannels[idx])
            accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[1])(accChannels[idx])
            accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(accChannels[idx])
            accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[2])(accChannels[idx])
            accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(accChannels[idx])
            accChannels[idx] = Flatten()(accChannels[idx])
    else:
        for idx in range(trainingAcc.shape[2]):
            accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(accChannels[idx])
            accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(accChannels[idx])
            accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(accChannels[idx])
            accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(accChannels[idx])
            accChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(accChannels[idx])
            accChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(accChannels[idx])
            accChannels[idx] = Flatten()(accChannels[idx])

    inputGrav = Input(shape=trainingGrav.shape[1:]) 
    bnGrav = BatchNormalization(axis=2)(inputGrav)
    gravChannels = []
    for idx in range(trainingGrav.shape[2]):
        gravChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(gravTimeWindow,1))(bnGrav))
    if transfer != 'tto':
        for idx in range(trainingGrav.shape[2]):
            gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[0])(gravChannels[idx])
            gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(gravChannels[idx])
            gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[1])(gravChannels[idx])
            gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(gravChannels[idx])
            gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[2])(gravChannels[idx])
            gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(gravChannels[idx])
            gravChannels[idx] = Flatten()(gravChannels[idx])
    else:
        for idx in range(trainingGrav.shape[2]):
            gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(gravChannels[idx])
            gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(gravChannels[idx])
            gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(gravChannels[idx])
            gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(gravChannels[idx])
            gravChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(gravChannels[idx])
            gravChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(gravChannels[idx])
            gravChannels[idx] = Flatten()(gravChannels[idx])

    inputGyro = Input(shape=trainingGyro.shape[1:]) 
    bnGyro = BatchNormalization(axis=2)(inputGyro)
    gyroChannels = []
    for idx in range(trainingGyro.shape[2]):
        gyroChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(gyroTimeWindow,1))(bnGyro))
    if transfer != 'tto':
        for idx in range(trainingGyro.shape[2]):
            gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[0])(gyroChannels[idx])
            gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(gyroChannels[idx])
            gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[1])(gyroChannels[idx])
            gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(gyroChannels[idx])
            gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[2])(gyroChannels[idx])
            gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(gyroChannels[idx])
            gyroChannels[idx] = Flatten()(gyroChannels[idx])
    else:
        for idx in range(trainingGyro.shape[2]):
            gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(gyroChannels[idx])
            gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(gyroChannels[idx])
            gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(gyroChannels[idx])
            gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(gyroChannels[idx])
            gyroChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(gyroChannels[idx])
            gyroChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(gyroChannels[idx])
            gyroChannels[idx] = Flatten()(gyroChannels[idx])

    inputLinAcc = Input(shape=trainingLinAcc.shape[1:]) 
    bnLinAcc = BatchNormalization(axis=2)(inputLinAcc)
    linAccChannels = []
    for idx in range(trainingLinAcc.shape[2]):
        linAccChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(linAccTimeWindow,1))(bnLinAcc))
    if transfer != 'tto':
        for idx in range(trainingLinAcc.shape[2]):
            linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[0])(linAccChannels[idx])
            linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(linAccChannels[idx])
            linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[1])(linAccChannels[idx])
            linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(linAccChannels[idx])
            linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'],weights=smartphoneLayersWeights[2])(linAccChannels[idx])
            linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(linAccChannels[idx])
            linAccChannels[idx] = Flatten()(linAccChannels[idx])
    else:
        for idx in range(trainingLinAcc.shape[2]):
            linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][0],kernel_size=smartphoneModel['conv_kernels_size'][0],activation=smartphoneModel['activation'])(linAccChannels[idx])
            linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][0])(linAccChannels[idx])
            linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][1],kernel_size=smartphoneModel['conv_kernels_size'][1],activation=smartphoneModel['activation'])(linAccChannels[idx])
            linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][1])(linAccChannels[idx])
            linAccChannels[idx] = Conv1D(filters=smartphoneModel['nb_conv_kernels'][2],kernel_size=smartphoneModel['conv_kernels_size'][2],activation=smartphoneModel['activation'])(linAccChannels[idx])
            linAccChannels[idx] = MaxPooling1D(pool_size=smartphoneModel['pooling_size'][2])(linAccChannels[idx])
            linAccChannels[idx] = Flatten()(linAccChannels[idx])

    inputMSAcc = Input(shape=trainingMSAcc.shape[1:]) 
    bnMsAcc = BatchNormalization(axis=2)(inputMSAcc)
    msAccChannels = []
    for idx in range(trainingMSAcc.shape[2]):
        msAccChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(msAccTimeWindow,1))(bnMsAcc))
    if transfer != 'tto':
        for idx in range(trainingMSAcc.shape[2]):
            msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][0],kernel_size=msbandModel['conv_kernels_size'][0],activation=msbandModel['activation'],weights=msbandLayersWeights[0])(msAccChannels[idx])
            msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][0])(msAccChannels[idx])
            msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][1],kernel_size=msbandModel['conv_kernels_size'][1],activation=msbandModel['activation'],weights=msbandLayersWeights[1])(msAccChannels[idx])
            msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][1])(msAccChannels[idx])
            msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][2],kernel_size=msbandModel['conv_kernels_size'][2],activation=msbandModel['activation'],weights=msbandLayersWeights[2])(msAccChannels[idx])
            msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][2])(msAccChannels[idx])
            msAccChannels[idx] = Flatten()(msAccChannels[idx])
    else:
        for idx in range(trainingMSAcc.shape[2]):
            msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][0],kernel_size=msbandModel['conv_kernels_size'][0],activation=msbandModel['activation'])(msAccChannels[idx])
            msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][0])(msAccChannels[idx])
            msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][1],kernel_size=msbandModel['conv_kernels_size'][1],activation=msbandModel['activation'])(msAccChannels[idx])
            msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][1])(msAccChannels[idx])
            msAccChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][2],kernel_size=msbandModel['conv_kernels_size'][2],activation=msbandModel['activation'])(msAccChannels[idx])
            msAccChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][2])(msAccChannels[idx])
            msAccChannels[idx] = Flatten()(msAccChannels[idx])

    inputMSGyro = Input(shape=trainingMSGyro.shape[1:]) 
    bnMsGyro = BatchNormalization(axis=2)(inputMSGyro)
    msGyroChannels = []
    for idx in range(trainingMSGyro.shape[2]):
        msGyroChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(msGyroTimeWindow,1))(bnMsGyro))
    if transfer != 'tto':
        for idx in range(trainingMSGyro.shape[2]):
            msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][0],kernel_size=msbandModel['conv_kernels_size'][0],activation=msbandModel['activation'],weights=msbandLayersWeights[0])(msGyroChannels[idx])
            msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][0])(msGyroChannels[idx])
            msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][1],kernel_size=msbandModel['conv_kernels_size'][1],activation=msbandModel['activation'],weights=msbandLayersWeights[1])(msGyroChannels[idx])
            msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][1])(msGyroChannels[idx])
            msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][2],kernel_size=msbandModel['conv_kernels_size'][2],activation=msbandModel['activation'],weights=msbandLayersWeights[2])(msGyroChannels[idx])
            msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][2])(msGyroChannels[idx])
            msGyroChannels[idx] = Flatten()(msGyroChannels[idx])
    else:
        for idx in range(trainingMSGyro.shape[2]):
            msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][0],kernel_size=msbandModel['conv_kernels_size'][0],activation=msbandModel['activation'])(msGyroChannels[idx])
            msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][0])(msGyroChannels[idx])
            msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][1],kernel_size=msbandModel['conv_kernels_size'][1],activation=msbandModel['activation'])(msGyroChannels[idx])
            msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][1])(msGyroChannels[idx])
            msGyroChannels[idx] = Conv1D(filters=msbandModel['nb_conv_kernels'][2],kernel_size=msbandModel['conv_kernels_size'][2],activation=msbandModel['activation'])(msGyroChannels[idx])
            msGyroChannels[idx] = MaxPooling1D(pool_size=msbandModel['pooling_size'][2])(msGyroChannels[idx])
            msGyroChannels[idx] = Flatten()(msGyroChannels[idx])

    inputJinsAcc = Input(shape=trainingJinsAcc.shape[1:]) 
    bnJinsAcc = BatchNormalization(axis=2)(inputJinsAcc)
    jinsAccChannels = []
    for idx in range(trainingJinsAcc.shape[2]):
        jinsAccChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(jinsTimeWindow,1))(bnJinsAcc))
    if transfer != 'tto':
        for idx in range(trainingJinsAcc.shape[2]):
            jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][0],kernel_size=jinsModel['conv_kernels_size'][0],activation=jinsModel['activation'],weights=jinsLayersWeights[0])(jinsAccChannels[idx])
            jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][0])(jinsAccChannels[idx])
            jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][1],kernel_size=jinsModel['conv_kernels_size'][1],activation=jinsModel['activation'],weights=jinsLayersWeights[1])(jinsAccChannels[idx])
            jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][1])(jinsAccChannels[idx])
            jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][2],kernel_size=jinsModel['conv_kernels_size'][2],activation=jinsModel['activation'],weights=jinsLayersWeights[2])(jinsAccChannels[idx])
            jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][2])(jinsAccChannels[idx])
            jinsAccChannels[idx] = Flatten()(jinsAccChannels[idx])
    else:
        for idx in range(trainingJinsAcc.shape[2]):
            jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][0],kernel_size=jinsModel['conv_kernels_size'][0],activation=jinsModel['activation'])(jinsAccChannels[idx])
            jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][0])(jinsAccChannels[idx])
            jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][1],kernel_size=jinsModel['conv_kernels_size'][1],activation=jinsModel['activation'])(jinsAccChannels[idx])
            jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][1])(jinsAccChannels[idx])
            jinsAccChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][2],kernel_size=jinsModel['conv_kernels_size'][2],activation=jinsModel['activation'])(jinsAccChannels[idx])
            jinsAccChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][2])(jinsAccChannels[idx])
            jinsAccChannels[idx] = Flatten()(jinsAccChannels[idx])


    inputJinsGyro = Input(shape=trainingJinsGyro.shape[1:]) 
    bnJinsGyro = BatchNormalization(axis=2)(inputJinsGyro)
    jinsGyroChannels = []
    for idx in range(trainingJinsGyro.shape[2]):
        jinsGyroChannels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(jinsTimeWindow,1))(bnJinsGyro))
    if transfer != 'tto':
        for idx in range(trainingJinsGyro.shape[2]):
            jinsGyroChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][0],kernel_size=jinsModel['conv_kernels_size'][0],activation=jinsModel['activation'],weights=jinsLayersWeights[0])(jinsGyroChannels[idx])
            jinsGyroChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][0])(jinsGyroChannels[idx])
            jinsGyroChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][1],kernel_size=jinsModel['conv_kernels_size'][1],activation=jinsModel['activation'],weights=jinsLayersWeights[1])(jinsGyroChannels[idx])
            jinsGyroChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][1])(jinsGyroChannels[idx])
            jinsGyroChannels[idx] = Conv1D(filters=jinsModel['nb_conv_kernels'][2],kernel_size=jinsModel['conv_kernels_size'][2],activation=jinsModel['activation'],weights=jinsLayersWeights[2])(jinsGyroChannels[idx])
            jinsGyroChannels[idx] = MaxPooling1D(pool_size=jinsModel['pooling_size'][2])(jinsGyroChannels[idx])
            jinsGyroChannels[idx] = Flatten()(jinsGyroChannels[idx])
    else:
        for idx in range(trainingJinsGyro.shape[2]):
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
                  optimizer=keras.optimizers.Adadelta(lr=learningRate),
                  metrics=['acc',fmeasure])        


    # # Print a model summary
    # model.summary()

    # Tensorboard report
    pathToReport = '/tmp/tensorboard-report/multimodalCnn-'+transfer+'-CogAge-'+classification+'/'
    
    if not os.path.exists(pathToReport):
        os.makedirs(pathToReport)
    logFilesList= os.listdir(pathToReport)
    if logFilesList != []:
         for file in logFilesList:
             os.remove(pathToReport+file)
    if not os.path.exists('/tmp/checkpoint/'):
        os.makedirs('/tmp/checkpoint/')
    tensorboard = keras.callbacks.TensorBoard(log_dir=pathToReport, histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = keras.callbacks.ModelCheckpoint('/tmp/checkpoint/checkpoint.hdf5',monitor='val_acc',save_best_only=True,save_weights_only=False)

    print('Initiating the training phase using %d %% of the training set ...' % (targetDataProportion*100))

    model.fit([trainingAcc,trainingGrav,trainingGyro,trainingLinAcc,trainingMSAcc,trainingMSGyro,trainingJinsAcc,trainingJinsGyro], 
            trainingLabels,
            batch_size=batchSize,
            epochs=epochs,
            verbose=verbose,
            validation_data=([testingAcc,testingGrav,testingGyro,testingLinAcc,testingMSAcc,testingMSGyro,testingJinsAcc,testingJinsGyro], 
                testingLabels),
            callbacks=[tensorboard,checkpoint])

    model.load_weights('/tmp/checkpoint/checkpoint.hdf5')
    print('Model evaluation')

    score = model.evaluate([testingAcc,testingGrav,testingGyro,testingLinAcc,testingMSAcc,testingMSGyro,testingJinsAcc,testingJinsGyro],
        testingLabels,verbose=0,batch_size=batchSize)
    
    end = time.time()

    print('##############################################')
    print('Training summary:')
    for idx in range(len(score)):
        print('%s: %.4f' % (model.metrics_names[idx],score[idx]))
    print('##############################################')
    print('Tensorboard log file generated in the directory /tmp/logs/')
    print('Use the command')
    print('    tensorboard --logdir %s' % (pathToReport))
    print('to read it')
    print('##############################################')

   
    # Compute estimations on the testing set
    print('Computing DNN estimations on the testing set...')
    estimations = np.zeros((nbTestingExamples,nbClasses),dtype=np.float32)
    idx = 0
    while idx < nbTestingExamples:
        if idx + batchSize < nbTestingExamples:
            endIdx = idx+batchSize
            size = batchSize
        else:
            endIdx = nbTestingExamples
            size = nbTestingExamples-idx
    
        predictions = model.predict([testingAcc[idx:endIdx],testingGrav[idx:endIdx],testingGyro[idx:endIdx],testingLinAcc[idx:endIdx],
            testingMSAcc[idx:endIdx],testingMSGyro[idx:endIdx],testingJinsAcc[idx:endIdx],testingJinsGyro[idx:endIdx]],batch_size=size)
       
        estimations[idx:endIdx] = predictions
        idx += batchSize
   
    # Compute evaluation metrics
    estimatedLabels = np.argmax(estimations,axis=1)
    accuracy = accuracy_score(groundTruthLabels,estimatedLabels)
    confusionMatrix = confusion_matrix(groundTruthLabels,estimatedLabels)

    # Compute F1 score
    af1ScoreArray = f1_score(groundTruthLabels,estimatedLabels,average=None)
    af1Score = np.mean(af1ScoreArray)

    print('\n')
    #print('Overall accuracy on the %d classes: %.2f %%' % (nbClasses,accuracy*100))
    print('\n')

    # Computation of MAP
    if classification == 'state':
        MAP, classAPs = computeMeanAveragePrecision(groundTruthLabels,estimations) 
        print('Confusion Matrix for the 6 state activities:')
        print(confusionMatrix)
        print('\n')
        meanAcc, classAcc = computeAverageAccuracy(confusionMatrix)
        print('Average ACCURACY for state activities: %.2f %%' % (meanAcc*100))
        print('All accuracies for state activities:')
        print(classAcc*100)
        print('\n')
        print('Overall AF1-SCORE on the %d classes: %.2f %%' % (nbClasses,af1Score*100))
        print('Class F1-scores:')
        print(af1ScoreArray)
        print('\n')
        print('MAP for state activities: %.2f %%' % (MAP*100))    
        print('All average precisions for state activities:')    
        print(classAPs)

        # Indices of wrongly classified examples
        wronglyClassifiedExamples = [idx for idx in range(len(groundTruthLabels)) if groundTruthLabels[idx] != estimatedLabels[idx]]
        print('')
        print('Indices of misclassified testing examples:')
        print(wronglyClassifiedExamples)
        print('')
        np.save('misclassified-idx-'+classification+'-'+transfer+'.npy',wronglyClassifiedExamples)

    else:
        MAP, classAPs = computeMeanAveragePrecision(groundTruthLabels,estimations)
        meanAcc, classAcc = computeAverageAccuracy(confusionMatrix)
        print('Average ACCURACY for behavioral activities: %.2f %%' % (meanAcc*100))
        print('All accuracies for behavioral activities:')
        print(classAcc*100)
        print('\n')   
        print('Overall AF1-SCORE on the %d classes: %.2f %%' % (nbClasses,af1Score*100))
        print('Class F1-scores:')
        print(af1ScoreArray)
        print('\n')
        print('MAP for behavioral activities: %.2f %%' % (MAP*100))    
        print('All average precisions for behavioral activities:')    
        print(classAPs)

    print('\n')

    #np.savetxt('debug.txt',behavioralConfusionMatrix,delimiter=' ',fmt='%2d')
    if len(saveName) > 0:
        model.save(modelPath+'/'+saveName+'.h5')

    if len(apPath) > 0:
        apName = 'APs-cnn-'+classification+'-'+transfer+'.npy'
        np.save(apPath+apName,classAPs)

    end = time.time()
    print('Total time used: %.2f seconds' % (end-start))

    K.clear_session()


#----------------------------------------------------------------------------------------------
# Main
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':

    multichannelCnnClassify(
        targetDataPath=targetDataPath,
        transfer=transfer,
        classification=classification,
        batchSize=batchSize,
        learningRate=learningRate,
        epochs=epochs,
        smartphoneModel=smartphoneModel,
        msbandModel=msbandModel,
        jinsModel=jinsModel,
        denseSize=denseSize,
        denseActivation=denseActivation,
        trainedSmartphoneModelPath=trainedSmartphoneModelPath,
        trainedMSBandModelPath=trainedMSBandModelPath,
        trainedJinsModelPath=trainedJinsModelPath,
        saveName=saveName,
        modelPath=modelPath,
        apPath=apPath,
        targetDataProportion=targetDataProportion
        )