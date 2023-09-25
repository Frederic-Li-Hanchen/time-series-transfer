##############################################################################################
# Script implementing the NISP and InfFS algorithms used in [*]
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Moddality Classification, 
# F.Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek 
##############################################################################################

from utilitary import *
import scipy.stats as sst
import pickle as pkl


#############################################################################
# Class InfFS for the implementation of the InfFS ranking algorithm
# Reference: [1] Infinite Feature Selection, G. Roffo et al., ICCV 2015
# #############################################################################
class InfFS:

    def __init__(self,data,alpha=0.5,coeff=0.9):

        self.alpha = alpha # InfFS loading coefficient

        try:
            self.alpha <= 1 and self.alpha >= 0
        except:
            print('ERROR: alpha must be in the interval [0,1]')
            exit()

        self.data = data # Set of features to compute the importance scores (2D matrix of size nbExamples x nbFeatures)
        self.nbFeatures = data.shape[1] # Number of features
        self.scores = np.zeros((data.shape[1]),dtype=np.float32) # Array to contain the scores for each features
        self.costMatrix = np.zeros((data.shape[1],data.shape[1]),dtype=np.float32)
        self.coefficient = coeff # Spectral radius coefficient
        self.energyMatrix = np.zeros((data.shape[1],data.shape[1]),dtype=np.float32)

    # Compute the cost matrix of InfFS
    def computeCostMatrix(self):

        # Compute variances of each feature
        standardDeviations = np.std(self.data,axis=0)

        # Compute cost matrix using Spearman correlation coefficients for each pair of features
        for i in range(self.nbFeatures):
            for j in range(self.nbFeatures):
                spearman, _ = sst.spearmanr(self.data[:,i],self.data[:,j]) ### TODO: possible problem with returning Nan values if std=0 for one channel
                self.costMatrix[i,j] = self.alpha*max(standardDeviations[i],standardDeviations[j]) + (1-self.alpha)*(1-abs(spearman)) 

        # Note: the cost matrix might contain Nan elements if some features have a standard-deviation of 0 (the Spearman correlation coefficient involves a division by the standard-deviation)
        # The correlation coefficient cannot be defined in that case. As a temporary fix, the values are set to 0
        if np.isnan(self.costMatrix).any():
            self.costMatrix = np.nan_to_num(self.costMatrix)

    # Compute the regularisation coefficient by checking the eigenvalues of the cost matrix
    def computeRegularisationCoefficient(self):
        eigenvalues, _ = np.linalg.eig(self.costMatrix)
        self.coefficient /= max(abs(eigenvalues))

    # Compute the energy matrix 
    def computeEnergyMatrix(self):
        identity = np.eye(self.nbFeatures,dtype=np.float32) 
        self.energyMatrix = np.linalg.inv(identity-self.coefficient*self.costMatrix) - identity

    # Compute the score for all features
    def computeFeatureScores(self):
        e = np.ones((self.nbFeatures),dtype=np.float32)
        self.computeCostMatrix()
        self.computeRegularisationCoefficient()
        self.computeEnergyMatrix()
        self.scores = np.dot(self.energyMatrix,e)
        return self.scores



############################################################################################
# Class Nisp for the implementation of the NISP neuron computation algorithm
# Reference: [2] NISP: Pruning Networks usign Neuron Importance Score Propagation, 
# R. Yu et al., CVPR 2018
# NOTE: current implementation specific to the CogAge dataset and mDNN architecture
############################################################################################
class Nisp:

    def __init__(self,pathToTrainedModel='',pathToData='',modalities=[],alpha=0.5): 

        self.alpha = alpha # InfFS loading coefficient

        try:
            self.alpha <= 1 and self.alpha >= 0
        except:
            print('ERROR: alpha must be in the interval [0,1]')
            exit()

        self.scores = {} # Dictionary to contain scores per layer
        self.pathToTrainedModel = pathToTrainedModel # Path to the trained mDNN
        self.pathToData = pathToData # Path to the folder containing data
        self.data = {} # Dictionary containing the various input data arrays from different sensors
        self.features = [] # Array to contain the features of the DNN (i.e. output of the penultimate layer computed on a dataset)
        self.modalities = modalities # List of numpy file names corresponding to the different CogAge sensor data
        self.concatenateRepartition = [] # Specific to mDNN: list of positive integers indicating how many neurons of the concat layer are connected to each branch

    def loadTrainedModel(self): # Note: for some reason, loading an architecture with lamda layers produces an error
        #model = load_model(self.pathToTrainedModel)
        model = loadTrainedMultichannelDnn(
            weightFilePath=self.pathToTrainedModel,
            testingAcc=self.data[self.modalities[0]],
            testingGrav=self.data[self.modalities[1]],
            testingGyro=self.data[self.modalities[2]],
            testingLinAcc=self.data[self.modalities[3]],
            testingMSAcc=self.data[self.modalities[4]],
            testingMSGyro=self.data[self.modalities[5]],
            testingJinsAcc=self.data[self.modalities[6]],
            testingJinsGyro=self.data[self.modalities[7]])
        modelWithoutSoftmax = Model(inputs=model.input,outputs=model.layers[-2].output)
        modelWithoutSoftmax.layers[-1]._outbound_nodes = []
        self.model = modelWithoutSoftmax

    def loadData(self):
        for file in self.modalities:
            self.data[file] = np.expand_dims(np.load(self.pathToData+file),3)

    def computeFeatures(self):
        dataList = [self.data[e] for e in self.modalities]
        self.features = self.model.predict(dataList)

    def computeConcatenateRepartition(self): # Compute the number of neurons that each branch contributes to in the concat layer
        flattenLayerNames = [layer.name for layer in self.model.layers if 'flatten' in layer.name]
        repartition = np.zeros((len(flattenLayerNames)),dtype=int)
        for idx in range(len(flattenLayerNames)):
            repartition[idx] = self.model.get_layer(flattenLayerNames[idx]).output_shape[1]
        self.concatenateRepartition = repartition

    # Compute the matrix of correspondancies between the input and output tensor of a pooling layer
    # dimension1 and dimension2 are the length of the time dimension of the input and output respectively
    def definePoolingMatrix(self,poolingSize,dimension1,dimension2):
        matrix = np.zeros((dimension1,dimension2),dtype=np.float32)
        # Note: dimension2 should be dimension1/poolingSize
        idx1 = 0
        for idx2 in range(dimension2):
            matrix[idx1:idx1+poolingSize,idx2] = 1
            idx1 += poolingSize # By default, non-overlapping pooling windows
        # Return the pooling matrix
        return matrix

    # Compute the matrix of correspondancies between the input and output tensors of a conv layer
    # The weights given as input are assumed to connect one specific input neuron to a specific output one
    # dimension1 and dimension2 are the length of the time dimension of the input and output respectively
    def defineConvMatrix(self,filterWeights,dimension1,dimension2):
        nbWeights = len(filterWeights) # i.e. filter length in the time dimension
        matrix = np.zeros((dimension1,dimension2),dtype=np.float32)
        # Note: in the no padding case, dimension2 = dimension1 - nbWeights + 1
        idx1 = 0
        for idx2 in range(dimension2):
            matrix[idx1:idx1+nbWeights,idx2] = filterWeights
            idx1 += 1 # By default, convolutional stride of 1
        return matrix

    # Compute the score of a layer given by layerName
    # NOTE: implementation specific to the mDNN architecture
    def computeScoresLayer(self,layerName):    

        if len(self.model.get_layer(layerName)._outbound_nodes) == 0: # Last layer of the model: doesn't have any outbound nodes
            print('Computing InfFS scores for the final layer ...')
            self.computeFeatures()
            infFs = InfFS(self.features,self.alpha)
            #st()
            score = infFs.computeFeatureScores()
            ### DEBUG
            #score = np.random.rand(self.model.get_layer(layerName).output_shape[1])
            self.scores[layerName] = score

        else:

            # Treat the case where the layer is a conv or pooling layer
            # Get the name of the next layer
            node = self.model.get_layer(layerName)._outbound_nodes[0]
            nextLayer = node.outbound_layer
            nextLayerName = nextLayer.name

            if ('dense' in layerName or 'concatenate' in layerName) and 'dense' in nextLayerName: # Multiply with weight matrix of the previous layer
                # Use neuron importance of the next layer to compute importance scores of the current layer (should already be computed)
                nextLayerScores = self.scores[nextLayerName]
                # Get weight matrix
                weightMatrix = self.model.get_layer(nextLayerName).get_weights()[0] # Biases are not used
                # Compute score importance of the current layer
                self.scores[layerName] = np.dot(np.absolute(weightMatrix),nextLayerScores) # TODO: check if sizes fit, should matrix be transposed?

            elif 'flatten' in layerName and 'concatenate' in nextLayerName:
                # Get flatten layer index
                layerIdx = int(layerName.replace('flatten_',''))-1
                # Get the corresponding importance scores in the concatenate layer
                if layerIdx != 0:
                    startingIdx = np.cumsum(self.concatenateRepartition[:layerIdx])[-1] 
                else:
                    startingIdx = 0
                endingIdx = startingIdx + self.concatenateRepartition[layerIdx]
                self.scores[layerName] = self.scores[nextLayerName][startingIdx:endingIdx]

            elif 'pooling1d' in layerName and 'flatten' in nextLayerName: # Scores are not changed in this configuration
                # Reshape the scores in a matricial format
                _, timeDimension, nbKerns = self.model.get_layer(layerName).output_shape
                self.scores[layerName] = np.transpose(np.reshape(self.scores[nextLayerName],(timeDimension,nbKerns)))

            #elif 'pooling1d' in layerName and 'conv1d' in nextLayerName:
            elif 'conv1d' in layerName and 'pooling1d' in nextLayerName: # 3D tensor to 3D tensor
                # Get the number of features of the pooling layer
                nbFeatures = self.model.get_layer(layerName).input_shape[2]
                # Get the time dimension of the input and output
                inputDimension = self.model.get_layer(nextLayerName).input_shape[1]
                outputDimension = self.model.get_layer(nextLayerName).output_shape[1]
                # Get the pooling factor
                poolSize = self.model.get_layer(nextLayerName).pool_size[0]
                # Compute the pooling back-propagation matrix
                matrix = self.definePoolingMatrix(poolSize,inputDimension,outputDimension)
                # Use neuron importance of the next layer to compute importance scores of the current layer (should already be computed)
                # Should be a matrix of size nbFeatures x outputDimension
                nextLayerScores = self.scores[nextLayerName]
                # Compute the new scores for the layer
                scoreMatrix = np.zeros((nbFeatures,inputDimension),dtype=np.float32)
                for idx in range(nbFeatures):
                    scoreMatrix[idx,:] = np.dot(matrix,nextLayerScores[idx,:])
                self.scores[layerName] = scoreMatrix
                
            #elif 'conv1d' in layerName and 'pooling1d' in nextLayerName: # 3D tensor to 3D tensor
            elif 'pooling1d' in layerName and 'conv1d' in nextLayerName:
                # Get weight matrix in absolute value
                weightMatrix = np.absolute(self.model.get_layer(nextLayerName).get_weights()[0]) # Note: biases are not used
                filterSize, nbKernsIn, nbKernsOut = weightMatrix.shape
                # Get the size of the input and output along the time dimension
                inputDimension = self.model.get_layer(nextLayerName).input_shape[1]
                outputDimension = self.model.get_layer(nextLayerName).output_shape[1]
                # Prepare the score matrix
                scoreMatrix = np.zeros((nbKernsIn,inputDimension),dtype=np.float32)
                # Score matrix of the next layer (should be of size nbKernsOut x outputDimension)
                nextLayerScoreMatrix = self.scores[nextLayerName]

                # Fill the score matrix row by row
                for idx1 in range(nbKernsIn):
                    for idx2 in range(nbKernsOut):
                        convMatrix = self.defineConvMatrix(weightMatrix[:,idx1,idx2],inputDimension,outputDimension)
                        scoreMatrix[idx1,:] += np.dot(convMatrix,nextLayerScoreMatrix[idx2,:])
                # Save the score matrix
                self.scores[layerName] = scoreMatrix

            else:
                print('NISP for the connection between %s and %s not implemented yet!' % (layerName,nextLayerName))

        # Get the name(s) of the previous layer(s) and return it
        previousNode = self.model.get_layer(layerName)._inbound_nodes[0]

        if len(previousNode.inbound_layers)>1: # What happens if there are more than 1 previous layers? E.g. concatenate layer
            previousLayerNames = [previousNode.inbound_layers[idx].name for idx in range(len(previousNode.inbound_layers))]
        elif len(previousNode.inbound_layers)==1: # There is exactly one parent layer 
            previousLayerNames = [previousNode.inbound_layers[0].name]
        else: # There is no parent layer, i.e. the current layer is an input layer
            previousLayerNames = []
        
        return previousLayerNames

    # Compute importance scores for all layers of the mDNN
    def computeScoresAllLayers(self):
        ## Compute the number of layers for which a score computation is needed
        # Load the trained model and data
        self.loadData()
        self.loadTrainedModel()

        # Compute the concatenation repartition (for multibranches architectures)
        self.computeConcatenateRepartition()

        # Get the name of the last layer and put it in a list
        layersToComputeScore = [l.name for l in self.model.layers if len(l._outbound_nodes) == 0]

        # Loop on the layers to compute the scores and backpropagate them
        while len(layersToComputeScore) > 0:
            print('------------------------------------------------------------------------------------')
            print('Computing importance scores for layer %s ...' % (layersToComputeScore[0]))
            previousLayerNames = self.computeScoresLayer(layersToComputeScore[0]) # Stopping condition: when an input layer is reached, the returned list is empty
            # Remove the already processed layer from the list of layers to process
            layersToComputeScore.remove(layersToComputeScore[0])
            # Append the list of new layers
            layersToComputeScore += previousLayerNames

        return self.scores


##################################
# Main function
##################################
if __name__ == '__main__':
    
    ### InfFS test
    # testData = np.random.rand(10000,20)
    # infFS = InfFS(testData,alpha=0.1)
    # infFS.computeFeatureScores()
    # print(infFS.scores)

    ### Nisp test
    classification = 'bbh' # 'bbh' or 'blho'
    transfer = 'tto' # 'cnn-transfer' or 'tto'

    pathToData = '../data/target/CogAge/'+classification+'/testing/'
    modalities = [
    'testAccelerometer.npy',
    'testGravity.npy',
    'testGyroscope.npy',
    'testLinearAcceleration.npy',
    'testMSAccelerometer.npy',
    'testMSGyroscope.npy',
    'testJinsAccelerometer.npy',
    'testJinsGyroscope.npy']
    pathToTrainedModel = '../mDNN/mCnn-CogAge-'+transfer+'-'+classification+'.h5'

    nisp = Nisp(pathToTrainedModel=pathToTrainedModel,pathToData=pathToData,modalities=modalities)

    start = time.time()
    scores = nisp.computeScoresAllLayers()
    end = time.time()

    print('Computation of the neural importance scores performed in %.2f seconds' % (end-start))

    f = open('../nisp-inffs/'+transfer+'-'+classification+'.pkl','wb')
    pkl.dump(scores,f)
    f.close()