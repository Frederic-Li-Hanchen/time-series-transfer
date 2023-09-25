##############################################################################################
# Script to train and evaluate a Multichannel DNN on the DEAP datset
# Used for the study presented in [*]. 
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification
# F.Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek 
##############################################################################################

from utilitary import *


#-------------------------------------------------------------------------------------------------------
# Hyper-parameters 
#-------------------------------------------------------------------------------------------------------
### Input data parameters
classification = 'arousal' # 'valence' or 'arousal'
transfer = 'tto' # 'tto' or 'vae-transfer' or 'cnn-transfer'
#foldsToTest = list(range(1,11)) # Default: list(range(1,11))
foldsToTest = [1]

if len(foldsToTest) > 1:
    verbose = 0 
else:
    verbose = 1

### Path to the folder containing the 10 folds of the DEAP dataset
targetDataPath = '../data/target/DEAP/10-cv-folds/'

### Path to save the Multichannel DNN
modelPath = '../mDNN/'
saveName = 'mCnn-DEAP-'+transfer+'-'+classification

### Path to the trained single-channel DNN
if transfer == 'cnn-transfer':
    trainedModelPath = '../sDNN/cnn-transfer-DEAP.h5'
elif transfer == 'vae-transfer':
    trainedModelPath = '../sDNN/cnnVae-DEAP.h5'
else:
    trainedModelPath = ''

### Training parameters
batchSize = 500
epochs = 300
learningRate = 1 # NOTE: tested: 1, 0.1, 0.01
targetDataProportion = 1 # What percentage of the target domain should be used for the training? Default = 1. Tested values: 0.05, 0.25, 0.50, 0.75, 1 

### single-channel DNN parameters
selectedModel = {
    'name': 'CNN',
    'nb_conv_blocks' : 3,
    'nb_conv_kernels' : [10,10,10],
    'conv_kernels_size' : [(9,),(9,),(9,)],
    'pooling_size' : [(2,),(2,),(2,)],
    'activation' : 'relu',
    }

### Softmax MLP appended at the end of the multichannel DNN
denseSize = [1000,500,100] # Number of neurons per layer
denseActivation = 'relu'


##############################################################################################
# Main: training and evaluation of a Multichannel DNN on DEAP
##############################################################################################
if __name__ == '__main__':

    start = time.time()
  
    accuracies = np.zeros((10),dtype=np.float32)

    print('#############################################################')
    print('Binary classification problem for %s' % (classification.upper()))
    print('Folds tested: [%s]' % (','.join(map(str,foldsToTest))))
    print('Transfer strategy: %s' % (transfer))
    print('#############################################################') 

    timeWindow = 128 

    if transfer == 'cnn-transfer':

        sourceModel = cnn(
            inputShape=(timeWindow,1),
            nkerns=selectedModel['nb_conv_kernels'],
            filterSizes=selectedModel['conv_kernels_size'],
            poolSizes=selectedModel['pooling_size'],
            activation=selectedModel['activation'],
            nbClasses=nbModalitiesAllDatasets)

        sourceModel.load_weights(trainedModelPath,by_name=True) 

        sourceLayersDict = dict([(layer.name,layer) for layer in sourceModel.layers])
        sourceLayersWeights = []

        for idx in range(selectedModel['nb_conv_blocks']):
            sourceLayersWeights += [sourceLayersDict['conv'+str(idx+1)].get_weights()]


    elif transfer == 'vae-transfer':

        sourceModel = convAutoencoder(
            inputShape=(timeWindow,1),
            nkerns=selectedModel['nb_conv_kernels'],
            filterSizes=selectedModel['conv_kernels_size'],
            poolSizes=selectedModel['pooling_size'],
            activation=selectedModel['activation'])

        sourceModel.load_weights(trainedModelPath,by_name=True)
        sourceLayersDict = dict([(layer.name,layer) for layer in sourceModel.layers])
        sourceLayersWeights = []

        for idx in range(selectedModel['nb_conv_blocks']):
            sourceLayersWeights += [sourceLayersDict['encoder'+str(idx+1)].get_weights()]

    
    ### Loop on the DEAP folds
    nbClasses = 2

    for foldIdx in foldsToTest:

        strFold = str(foldIdx).zfill(2)

        print('----------------------------------------------------------------------------------------')
        print('Processing data for fold %s/10 ...' % (strFold))

        ### load the DEAP data
        print('    Loading the DEAP data ...')

        x_train = np.load(targetDataPath+'/'+strFold+'/x_train_'+strFold+'.npy')
        x_test = np.load(targetDataPath+'/'+strFold+'/x_test_'+strFold+'.npy')
        y_train = np.load(targetDataPath+'/'+strFold+'/y_'+classification+'_train_'+strFold+'.npy')
        y_test = np.load(targetDataPath+'/'+strFold+'/y_'+classification+'_test_'+strFold+'.npy')

        # Downsample the training set if needed
        assert targetDataProportion >0
        assert targetDataProportion <=1

        if targetDataProportion <1:

            print('    Downsampling the training set to %.2f %%' % (100*targetDataProportion))
            random.seed(0) # Fix the random seed to always get the same data downsampling

            # Shuffle in unisson traiing data and labels
            x_train, y_train, _ = shuffleInUnisson(x_train,y_train)

            # Downsample the training set
            upperBound = int(floor(len(y_train)*targetDataProportion))
            y_train = y_train[:upperBound]
            x_train = x_train[:upperBound]

        assert nbClasses >= len(set(y_train))
        assert nbClasses >= len(set(y_test))

        trainShape = x_train.shape
        testShape = x_test.shape

        timeWindow = trainShape[1]
        nbSensors = trainShape[2]

        assert trainShape[1] == testShape[1] # Size of the time window
        assert trainShape[2] == testShape[2] # Number of sensor channels
        assert trainShape[0] == y_train.shape[0] # Number of training examples
        assert testShape[0] == y_test.shape[0] # Number of testing examples

        x_train = x_train.reshape(trainShape[0],trainShape[1],trainShape[2], 1)
        x_test = x_test.reshape(testShape[0],testShape[1],testShape[2], 1)
        input_shape = (testShape[1], testShape[2], 1)
        print('    x_train shape: (%d, %d, %d)' % (trainShape[0], trainShape[1], trainShape[2]))

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, nbClasses)
        y_test = keras.utils.to_categorical(y_test, nbClasses)

   
        ############################ Model definition ##############################
        print('    Building the model ...')
        
        # Input layer
        inputLayer = Input(shape=input_shape) 

        # Batch norm layer
        bn = BatchNormalization()(inputLayer)

        # Separation by sensor 
        channels = []

        # Definition of sensor-specific network (either 3-layer MLP or CNN)

        for idx in range(nbSensors):
            channels.append(Lambda(lambda x: x[:,:,idx,:], output_shape=(timeWindow,1))(bn))

        if transfer == 'tto':
            for idx in range(nbSensors):
                for layerIdx in range(selectedModel['nb_conv_blocks']):
                    channels[idx] = Conv1D(filters=selectedModel['nb_conv_kernels'][layerIdx],kernel_size=selectedModel['conv_kernels_size'][layerIdx],activation=selectedModel['activation'])(channels[idx])
                    channels[idx] = MaxPooling1D(pool_size=selectedModel['pooling_size'][layerIdx])(channels[idx])
                channels[idx] = Flatten()(channels[idx])

        else:
            for idx in range(nbSensors):
                for layerIdx in range(selectedModel['nb_conv_blocks']):
                    channels[idx] = Conv1D(filters=selectedModel['nb_conv_kernels'][layerIdx],kernel_size=selectedModel['conv_kernels_size'][layerIdx],
                        activation=selectedModel['activation'],weights=sourceLayersWeights[layerIdx])(channels[idx])
                    channels[idx] = MaxPooling1D(pool_size=selectedModel['pooling_size'][layerIdx])(channels[idx])
                channels[idx] = Flatten()(channels[idx])


        # Concatenation of all sensor channel outputs
        concatenation = concatenate(channels)

        # Add dense layers
        dense = Dense(denseSize[0],activation=denseActivation)(concatenation)

        for idx in range(1,len(denseSize)):
           dense = Dense(denseSize[idx],activation=denseActivation)(dense)

        # Softmax layer
        outputLayer = Dense(nbClasses,activation='softmax')(dense)

        model = Model(inputs=inputLayer,outputs=outputLayer)


        ### Model compilation
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(lr=learningRate),
                      metrics=['acc', fmeasure])

        # Print a model summary
        #model.summary()

        # Tensorboard report
        pathToReport = '/home/prg/tensorboard-report/test/'
       
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

        # Early stopping condition
        #earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,min_delta=1e-5)
        earlyStop = CustomEarlyStopping(consecutiveEpochs=5)

        print('Initiating the training phase ...')

        # Supervised training
        model.fit(x_train, y_train,
                batch_size=batchSize,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_test, y_test),
                #callbacks=[tensorboard,checkpoint])
                callbacks=[tensorboard,checkpoint,earlyStop])
        model.load_weights('/tmp/checkpoint/checkpoint.hdf5')
        print('Model evaluation')
        score = model.evaluate(x_test, y_test, verbose=0)

        end = time.time()

        print('##############################################')
        print('Results for fold '+strFold+'/10:')
        for idx in range(len(score)):
            print('%s: %.4f' % (model.metrics_names[idx],score[idx]))
        print('##############################################')

        # Save the weights of the network
        model_json = model.to_json()

        savePath = modelPath +'/'+strFold+'/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        json_file = open(savePath + '/'+saveName+'.json', 'w+')
        json_file.write(model_json)
        json_file.close()

        # serialize weights to HDF5
        model.save_weights(savePath + '/'+saveName+'.h5')
        print('    Saved model to folder:' + savePath)

        # Save the performance metrics
        accuracies[foldIdx-1] = score[1]

        # Clear the memory to prevent resourceExhausted errors
        K.clear_session()

        np.save(modelPath+'/fold-accuracies.npy',accuracies)
    
    print('Average accuracy: %.2f' % (np.mean(accuracies)*100))

    end = time.time()
    print('Total time used: %.2f seconds' % (end-start))