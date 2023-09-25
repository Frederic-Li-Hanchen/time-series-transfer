##############################################################################################
# Script to train a sensor classifier on the source dataset
# Used for the study presented in [*].
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Moddality Classification
# F.Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek 
##############################################################################################

from utilitary import *

##############################################################################################
# Hyper-parameters
##############################################################################################

### Segmentation parameters
targetDataset = 'CogAgeMSBand' # 'DEAP' or 'CogAgeMSBand' or 'CogAgeSmartphone' or 'CogAgeJins'

### Model parameters
if targetDataset == 'DEAP':
    
    timeWindow = 128

    selectedModel = {
       'name': 'CNN',
       'nb_conv_blocks' : 3,
       'nb_conv_kernels' : [10,10,10],
       'conv_kernels_size' : [(9,),(9,),(9,)],
       'pooling_size' : [(2,),(2,),(2,)],
       'activation' : 'relu'
      }

elif targetDataset == 'CogAgeMSBand':
    
    timeWindow = 268
    device = 'smartwatch'

    selectedModel = {
       'name': 'CNN',
       'nb_conv_blocks' : 3,
       'nb_conv_kernels' : [10,10,10],
       'conv_kernels_size' : [(9,),(11,),(11,)],
       'pooling_size' : [(2,),(2,),(2,)],
       'activation' : 'relu',
      }


elif targetDataset == 'CogAgeSmartphone':
    
    timeWindow = 800
    device = 'smartphone'

    selectedModel = {
       'name': 'CNN',
       'nb_conv_blocks' : 3,
       'nb_conv_kernels' : [10,10,10],
       'conv_kernels_size' : [(45,),(49,),(46,)],
       'pooling_size' : [(2,),(2,),(2,)],
       'activation' : 'relu',
      }

elif targetDataset == 'CogAgeJins':
    
    timeWindow = 80
    device = 'smartglasses'

    selectedModel = {
       'name': 'CNN',
       'nb_conv_blocks' : 3,
       'nb_conv_kernels' : [10,10,10],
       'conv_kernels_size' : [(5,),(5,),(5,)],
       'pooling_size' : [(2,),(2,),(2,)],
       'activation' : 'relu',
      }


### Training parameters
split = 0.9 # Splitting factor between training and validation set (training = split * total number examples)
batchSize = 100
nbEpochs = 25 # Default: 100 for DEAP, 25 for CogAge
learningRate = 1 # NOTE: default ADADELTA learning rate = 1


### Mixed dataset path
# dataPath = '../data/source/processed/DEAP/mixedData_t128_s128_all.npy' # Path where the source data for DEAP are saved
# labelPath = '../data/source/processed/DEAP/mixedLabels_t128_s128_all.npy' # Path where the source labels for DEAP are saved

dataPath = '../data/source/processed/CogAge/'+device+'/mixedData_t268_s67_all.npy' # Path where the source data for CogAge are saved
labelPath = '../data/source/processed/CogAge/'+device+'/mixedLabels_t268_s67_all.npy' # Path where the source labels for DEAP are saved

### Model saving path
modelPath = '../sDNN/'



##############################################################################################
# Training of an autoencoder on mixed data
##############################################################################################
def trainSensorClassifier(dataPath,
                          modelPath,
                          learningRate,
                          epochs, 
                          batchSize,
                          selectedModel,
                          split,
                          timeWindow,
                          datasets='all'
                          ):

    start = time.time()

    # Load the mixed data
    print('Loading mixed data ...')

    mixedData = np.load(dataPath)
    mixedLabels = np.load(labelPath)

    nbClasses = nbModalitiesAllDatasets

    # Split training and validation sets
    x_train = mixedData[:int(floor(split*len(mixedData)))]
    x_test = mixedData[int(floor(split*len(mixedData))):]
    y_train = mixedLabels[:int(floor(split*len(mixedLabels)))]
    y_test = mixedLabels[int(floor(split*len(mixedLabels))):]

    if selectedModel['name'] == 'MLP':
        input_shape = (x_train.shape[1],)
    else:
        input_shape = (x_train.shape[1],1)
        x_train = np.expand_dims(x_train,axis=2) # Reshape to (nbExamples,timeWindow,1)
        x_test = np.expand_dims(x_test,axis=2) # Reshape to (nbExamples,timeWindow,1)
        #input_shape = x_train.shape[1:]
    print('-----------------------------------------------------------------')
    print('x_train shape: (%d, %d)' % (x_train.shape[0], x_train.shape[1]))
    print('x_test shape: (%d, %d)' % (x_test.shape[0], x_test.shape[1]))
    print('Model input shape:')
    print(input_shape)
    print('-----------------------------------------------------------------')

    y_train = keras.utils.to_categorical(y_train, nbClasses)
    y_test = keras.utils.to_categorical(y_test, nbClasses)
   

    ### Model definition
    print('Building the sensor modality classifier ...')

    model = Sequential()

    if selectedModel['name'] == 'MLP':
      
        model = bnMlp(inputShape=(timeWindow,),
                  denseLayersSize=selectedModel['dense_size'],
                  nbClasses=nbClasses,
                  activation=selectedModel['activation'])   
  

    elif selectedModel['name'] == 'CNN':

        model = bnCnn(inputShape=(timeWindow,1),
                nkerns=selectedModel['nb_conv_kernels'],
                filterSizes=selectedModel['conv_kernels_size'],
                poolSizes=selectedModel['pooling_size'],
                activation=selectedModel['activation'],
                nbClasses=nbClasses)

    else:
        print('--------------------------------------------------------------------------------')
        print('ERROR: Incorrect model parameters! Currently supported: MLP, CNN')
        print('--------------------------------------------------------------------------------')
        sys.exit()
    
    # Model compilation:
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=learningRate),
                  #optimizer=keras.optimizers.adagrad(lr=learningRate,decay=1e-5),
                  metrics=['acc', fmeasure])

    # Print a model summary
    model.summary()

    # Tensorboard report
    if not os.path.exists('/home/prg/sensor-classifier-tensorboard-report/test-PMC/'):
        os.makedirs('/home/prg/sensor-classifier-tensorboard-report/test-PMC/')
    logFilesList= os.listdir('/home/prg/sensor-classifier-tensorboard-report/test-PMC/')
    if logFilesList != []:
        for file in logFilesList:
            os.remove('/home/prg/sensor-classifier-tensorboard-report/test-PMC/'+file)
    if not os.path.exists('/tmp/checkpoint/'):
        os.makedirs('/tmp/checkpoint/')
    tensorboard = keras.callbacks.TensorBoard(log_dir='/home/prg/sensor-classifier-tensorboard-report/test-PMC/', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = keras.callbacks.ModelCheckpoint('/tmp/checkpoint/checkpoint.hdf5',monitor='val_loss',save_best_only=True,save_weights_only=False)

    print('Initiating the training phase ...')

    # Model training
    model.fit(x_train, y_train,
              batch_size=batchSize,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard,checkpoint])
    model.load_weights('/tmp/checkpoint/checkpoint.hdf5')
    print('Model evaluation')
    score = model.evaluate(x_test, y_test, verbose=0)

    end = time.time()

    print('##############################################')
    print('Total time used: %.2f seconds' % (end-start))
    print('Tensorboard log file generated in the directory /tmp/logs/')
    print('Use the command')
    print('    tensorboard --logdir /tmp/logs/')
    print('to read it')
    for idx in range(len(score)):
        print('%s: %.3f' % (model.metrics_names[idx],score[idx]))
    print('##############################################')

    # Save the weights of the network
    modelName = selectedModel['name'].lower()+'-transfer-'+targetDataset # Name under which the sensor classifier will be saved
    model_json = model.to_json()
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    with open(modelPath+'/'+modelName+'.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save(modelPath +'/'+modelName+'.h5')
    print('Saved model to folder:' + modelPath)

    K.clear_session()



##############################################################################################
# Main
##############################################################################################
if __name__ == '__main__':

    # Train the sensor classifier
    trainSensorClassifier(dataPath=dataPath,
                          modelPath=modelPath,
                          learningRate=learningRate,
                          epochs=nbEpochs, 
                          batchSize=batchSize,
                          selectedModel=selectedModel,
                          split=split,
                          timeWindow=timeWindow)