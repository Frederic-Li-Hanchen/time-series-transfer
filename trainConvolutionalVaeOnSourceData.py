##############################################################################################
# Script to train a convolutional autoencoder on the source dataset
# Used for the study presented in 
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification: 
# F.Li, K. Shirahama, A. M. Nisar, X. Huang, M. Grzegorzek 
##############################################################################################

from utilitary import *
from keras.losses import mse

##############################################################################################
# Hyper-parameters
##############################################################################################

### Segmentation parameters
targetDataset = 'CogAgeSmartphone' # 'DEAP' or CogAgeSmartphone' or 'CogAgeMSBand' or 'CogAgeJins'

### Training parameters
split = 0.9 # Splitting factor between training and validation set (training = split * total number examples)
batchSize = 10
nbEpochs = 10 
learningRate = 1 # NOTE: default ADADELTA learning rate = 1

### Encoder parameters
if targetDataset == 'CogAgeSmartphone':
    timeWindow = 800
    encoder = {'name': 'CNN',
               'nb_conv_blocks' : 3,
               'nb_conv_kernels' : [10,10,10],
               'conv_kernels_size' : [(45,),(49,),(46,)],
               'pooling_size' : [(2,),(2,),(2,)],
               'activation' : 'tanh', 
                }
    device = 'smartphone'

elif targetDataset == 'CogAgeMSBand':
    timeWindow = 268
    encoder = {
               'name': 'CNN',
               'nb_conv_blocks' : 3,
               'nb_conv_kernels' : [10,10,10],
               'conv_kernels_size' : [(9,),(11,),(11,)],
               'pooling_size' : [(2,),(2,),(2,)],
               'activation' : 'tanh', 
               }
    device = 'smartwatch'

elif targetDataset == 'DEAP':
    timeWindow = 128

    encoder = {
               'name': 'CNN',
               'nb_conv_blocks' : 3,
               'nb_conv_kernels' : [10,10,10],
               'conv_kernels_size' : [(9),(9),(9)],
               'pooling_size' : [(2),(2),(2)],
               'activation' : 'tanh',
              }

elif targetDataset == 'CogAgeJins':
    timeWindow = 80
    encoder = {
       'name': 'CNN',
       'nb_conv_blocks' : 3,
       'nb_conv_kernels' : [10,10,10],
       'conv_kernels_size' : [(5,),(5,),(5,)],
       'pooling_size' : [(2,),(2,),(2,)],
       'activation' : 'tanh', 
      }
    device = 'smartglasses'

### Data paths
dataPath = '../data/source/processed/DEAP/mixedData_t128_s128_all.npy' # Path where the source data for DEAP are saved
#dataPath = '../data/source/processed/CogAge/'+device+'/mixedData_t'+str(timeWindow)+'_s'+str(timeWindow/4)+'_all.npy' # Path where the source data for CogAge are saved
#modelPath = '../sDNN/' # Path to save the trained model
modelPath = '../test/' # Path to save the trained model


##############################################################################################
# Training of an autoencoder on mixed data
##############################################################################################
def trainConvolutionalAutoencoder(dataPath,
                                   modelPath,
                                   learningRate,
                                   epochs, 
                                   batchSize,
                                   selectedModel,
                                   split,
                                   timeWindow
                                  ):

    start = time.time()

    # Load the mixed data
    print('Loading mixed data ...')

    mixedData = np.load(dataPath)

    # Split training and validation sets
    x_train = mixedData[:int(floor(split*len(mixedData)))]
    x_test = mixedData[int(floor(split*len(mixedData))):]
    x_train = np.expand_dims(x_train,3)
    x_test = np.expand_dims(x_test,3)

    input_shape = (x_train.shape[1],1)
    print('x_train shape: %s' % (x_train.shape,))
    print('x_test shape: %s' % (x_test.shape,))
        

    ### Model definition
    # Note: implementation taken and modified from https://keras.io/examples/variational_autoencoder_deconv/
    print('Building the convolutional variational autoencoder ...')

    def sampling(args):
        """Reparameterization trick by sampling for an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # network parameters
    kernel_size = selectedModel['conv_kernels_size']
    filters = selectedModel['nb_conv_kernels']
    latent_dim = 10

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(len(filters)):
        x = Conv1D(filters=filters[i],
                   kernel_size=kernel_size[i],
                   activation=selectedModel['activation'],
                   padding='same',
                   name='encoder'+str(i+1))(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x) # TODO: change parameters 16?
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var',kernel_initializer='zeros')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2]))(x)

    for i in range(len(filters)):
        x = Conv1D(filters=filters[len(filters)-i-1],
                    kernel_size=kernel_size[len(filters)-i-1],
                    activation=selectedModel['activation'],
                    padding='same',
                    name='decoder'+str(i+1))(x)

    outputs = Conv1D(filters=1,
                      kernel_size=(1,), # TODO: check
                      activation='linear',
                      padding='same',
                      name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    model = Model(inputs, outputs, name='vae')

    # Define loss
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
 
    model.compile(optimizer=keras.optimizers.Adadelta(lr=learningRate))

    # Print a model summary
    model.summary()

    print('Initiating the training phase ...')

    model.fit(x_train,
              batch_size=batchSize,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, None))
   
    print('Model evaluation')
    score = model.evaluate(x_test, verbose=0)

    end = time.time()

    print('##############################################')
    print('Total time used: %.2f seconds' % (end-start))
    if len(model.metrics_names) > 1:
        for idx in range(len(score)):
            print('%s: %.3f' % (model.metrics_names[idx],score[idx]))
    else:
        print('%s: %.3f' % (model.metrics_names[0],score))
    print('##############################################')

    # Save the weights of the network
    model_json = model.to_json()
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    modelName = 'cnnVae-'+targetDataset # Save files names (.h5 and .json) | NOTE: the full model (encoder+decoder) is saved
    with open(modelPath + '/'+modelName+'.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save(modelPath + '/' + modelName + '.h5')
    print('Saved model to folder:' + modelPath)



##############################################################################################
# Main
##############################################################################################
if __name__ == '__main__':

    # Train the autoencoder
    trainConvolutionalAutoencoder(dataPath=dataPath,
                                   modelPath=modelPath,
                                   learningRate=learningRate,
                                   epochs=nbEpochs, 
                                   batchSize=batchSize,
                                   selectedModel=encoder,
                                   split=split,
                                   timeWindow=timeWindow)