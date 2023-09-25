##############################################################################################
# Script to plot importance scores obtained on different mDNNs used in [*].
# [*] Deep Transfer Learning for Time Series Data Based on Sensor Moddality Classification, 
# F.Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek 
##############################################################################################


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import norm
from scipy.stats import entropy

from pdb import set_trace as st


#########################################################################################################################
# Function to compare importance scores of two different DNNs (trained with TTO and CNN-transfer).
# - [str] pathToFile1: path to a dictionary containing the neuron importance scores of a mDNN
# - [str] pathToFile2: path to a dictionary containing the neuron importance scores of a mDNN with the same architecture
# - [bool] normalise: apply or not a min-max normalisation per scores of a layer
# - [bool] display: plot a histogram of differences computed on all layers of the mDNN or not
#
# Outputs:
# - [str list] layerNames: list of layer names ordered by decreasing difference
# - [float list] sortedDiff: list of layer differences ordered by decreasing value
# - [int list] permutation: NOTE: for debug mainly. List of the original layer indices before ordering
#########################################################################################################################
def computeLayerDifferences(pathToFile1,pathToFile2,normalise=True,display=True):

    # Load files
    file = open(pathToFile1,'r')
    scores1 = pkl.load(file)
    file.close()

    file = open(pathToFile2,'r')
    scores2 = pkl.load(file)
    file.close()

    # Check if the architectures of both DNNs are comparable:
    assert sorted(scores1.keys()) == sorted(scores2.keys())

    # Compare the differences of the scores of both DNNs layer by layer
    # Note: scores of earlier layers tend to be bigger: should the scores be normalised?
    nbLayers = len(scores1.keys())
    differences = {}

    # Normalise the score values per layer between 0 and 1 (min max normalisation)
    if normalise:
        for layerName in scores1.keys():
            scoresToNormalise = scores1[layerName]
            maximum = np.amax(scoresToNormalise)
            minimum = np.amin(scoresToNormalise)
            scores1[layerName] = (scoresToNormalise - minimum)/(maximum-minimum)

        for layerName in scores2.keys():
            scoresToNormalise = scores2[layerName]
            maximum = np.amax(scoresToNormalise)
            minimum = np.amin(scoresToNormalise)
            scores2[layerName] = (scoresToNormalise - minimum)/(maximum-minimum)

    # Compute Euclidean difference between scores of a same layer
    for layerName in scores1.keys():
        differences[layerName] = np.linalg.norm(scores1[layerName]-scores2[layerName])
    xlabel = 'L2 norm of the difference of scores'

    # Return the sorted keys by decreasing scores
    sortedDiff = differences.values()
    layerNames = differences.keys()
    permutation = np.flip(np.argsort(sortedDiff))
    sortedDiff = sorted(sortedDiff,reverse=True)
    layerNames = [layerNames[i] for i in permutation.tolist()]

    # Plot histogram of differences 
    if display:
        nbBins = 100
        plt.figure()
        plt.hist(sortedDiff,bins=nbBins)
        plt.title(str(nbBins)+'-bin histogram of score differences per layer between CNN-transfer and TTO')

        plt.xlabel(xlabel)
        plt.show()

    return layerNames, sortedDiff, permutation


####################################################################################################################
# Function to plot the score of a layer
# - [str] pathToFile: path to a dictionary containing the neuron importance scores for all layers of a trained mDNN
# - [str] layerName: name of the layer of a mDNN to plot the importance scores
####################################################################################################################
def plotLayerScore(pathToFile,layerName):

    # Load scores
    file = open(pathToFile,'r')
    scores = pkl.load(file)
    file.close()

    if layerName not in scores.keys():
        print('%s is not a layer of the loaded model!' % layerName)
        exit()

    # Get importance scores of the layer
    layerScores = scores[layerName]

    # Check the size of the scores (2D for conv/pooling layers, 1D for dense and concatenate)
    fig = plt.figure()
    plt.title('Importance scores for layer '+layerName)

    if len(layerScores.shape) == 1:
        plt.bar(list(range(1,len(layerScores)+1)),layerScores)

    elif len(layerScores.shape) == 2:
        plt.imshow(layerScores,cmap='gray')
        plt.colorbar()

    else:
        print('%dD score matrix found for layer %s: plotting not implemented!' % (len(layerScores.shape),layerName))

    plt.show()


##########################################################################################################################
# Function to plot a comparison of scores of a layer for two DNNs
# - [str] pathToFile1: path to a dictionary containing the neuron importance scores of a mDNN
# - [str] pathToFile2: path to a dictionary containing the neuron importance scores of a mDNN with the same architecture
# - [str] layerName: name of the layer of the mDNN for which the score should be plotted
# - [bool] normalise: apply or not a min-max normalisation per scores of a layer
##########################################################################################################################
def plotComparativeLayerScore(pathToFile1,pathToFile2,layerName,normalise=True):

    # Load scores
    file = open(pathToFile1,'r')
    scores1 = pkl.load(file)
    file.close()

    file = open(pathToFile2,'r')
    scores2 = pkl.load(file)
    file.close()

    if layerName not in scores1.keys():
        print('%s is not a layer of the loaded model!' % layerName)
        exit()

    # Get importance scores of the layer
    layerScores1 = scores1[layerName]
    layerScores2 = scores2[layerName]

    # Score normalisation
    if normalise:
        maximum = np.amax(layerScores1)
        minimum = np.amin(layerScores1)
        layerScores1 = (layerScores1 - minimum)/(maximum-minimum)

        maximum = np.amax(layerScores2)
        minimum = np.amin(layerScores2)
        layerScores2 = (layerScores2 - minimum)/(maximum-minimum)


    # Get the transfer method corresponding to each score dictionary
    if 'tto' in pathToFile1:
        transferMethod1 = 'tto'
    elif 'cnn-transfer' in pathToFile1:
        transferMethod1 = 'cnn-transfer'
    else:
        transferMethod1 = 'unknown'

    if 'tto' in pathToFile2:
        transferMethod2 = 'tto'
    elif 'cnn-transfer' in pathToFile2:
        transferMethod2 = 'cnn-transfer'
    else:
        transferMethod2 = 'unknown'

    # Remove dimensions equal to 1 from scores
    layerScores1 = np.squeeze(layerScores1)
    layerScores2 = np.squeeze(layerScores2)

    # Check the size of the scores (2D for conv/pooling layers, 1D for dense and concatenate)
    fig,ax = plt.subplots(nrows=1,ncols=2)

    if len(layerScores1.shape) == 1:
        ax[0].bar(list(range(1,len(layerScores1)+1)),layerScores1)
        ax[0].set_title(transferMethod1)
        ax[1].bar(list(range(1,len(layerScores2)+1)),layerScores2)
        ax[1].set_title(transferMethod2)

    elif len(layerScores1.shape) == 2:
        ax[0].imshow(layerScores1,cmap='gray')
        ax[0].set_title(transferMethod1)
        ax[0].set_xlabel('Time dimension')
        ax[0].set_ylabel('Channel dimension')
        #ax[0].colorbar()
        ax[1].imshow(layerScores2,cmap='gray')
        ax[1].set_title(transferMethod2)
        ax[1].set_xlabel('Time dimension')
        ax[1].set_ylabel('Channel dimension')
        #ax[1].colorbar()

    else:
        print('%dD score matrix found for layer %s: plotting not implemented!' % (len(layerScores.shape),layerName))

    fig.suptitle(layerName)
    plt.show()


###########################################################################################################################################
# Function to plot the bar graph of layer differences by decreasing values
# - [str list] layerNames: list of layer names ordered by decreasing difference
# - [float list] sortedDiff: list of layer differences ordered by decreasing value
# - [str] xlabels: either 'index' or 'branches' or 'names'. Indicates how to indicate the layer names on the plot
# - [int list] permutation: NOTE: for debug mainly. List of the original layer indices before ordering, only needed if xlabels == 'index'
# - [str] plotType: 'vertical' or 'horizontal'. Bar graph plotting mode
#
# NOTE: unclean implementation
###########################################################################################################################################
def plotBarGraph(layerNames,sortedDiff,xlabels='index',permutation=[],plotType='vertical'):

    nbLayers = len(layerNames)
    indices = np.arange(nbLayers)

    fig = plt.figure()
    
    layerAnnotations = []

    displayOffset = 0.5 # Factor to space out bars to make ticks easier to read  
    barWidth = 1

    # Prepare annotations containing branch ID, depth and device
    for idx in range(nbLayers):

        # Extract the layer index
        underscorePositions = [pos for pos, char in enumerate(layerNames[idx]) if char == '_']
        layerIdx = int(layerNames[idx][underscorePositions[-1]+1:])

        # Get the layer type:
        if 'conv' in layerNames[idx]:
            layerType = 'conv'
        elif 'pooling' in layerNames[idx]:
            layerType = 'pool'
        elif 'dense' in layerNames[idx]:
            layerType = 'dense'
        elif 'flatten' in layerNames[idx]:
            layerType = 'flat'
        else:
            layerType = ''

        # Get the branch index
        # Note: only works for mDNN with branch depth of 3!
        if layerType != 'dense':
            depth = layerIdx%3 # Note: between 0 and 2
            if depth == 0:
                depth = 3
            tmpIdx = layerIdx - depth
            if layerType == 'flat':
                depth = 4
            branchIdx = tmpIdx/3 + 1
        elif layerType == 'dense':
            branchIdx = -1
            depth = layerIdx

        ### Horizontal bar plot
        if plotType == 'horizontal':
            if branchIdx in list(range(1,13)): # smartphone channel
                plt.barh(nbLayers-(barWidth*idx+1),sortedDiff[idx],color='b',label='smartphone',height=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_B'+str(branchIdx)+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:
                    layerAnnotations += [str(permutation[idx]+1)]

        
            elif branchIdx in list(range(13,19)): # smartwatch channel
                plt.barh(nbLayers-(barWidth*idx+1),sortedDiff[idx],color='r',label='smartwatch',height=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_B'+str(branchIdx)+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:
                    layerAnnotations += [str(permutation[idx]+1)]


            elif branchIdx in list(range(19,25)): # smartglasses channel
                plt.barh(nbLayers-(barWidth*idx+1),sortedDiff[idx],color='limegreen',label='smartglasses',height=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_B'+str(branchIdx)+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:  
                    layerAnnotations += [str(permutation[idx]+1)]

            else:
                plt.barh(nbLayers-(barWidth*idx+1),sortedDiff[idx],color='k',height=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_B'+str(branchIdx)+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:
                    layerAnnotations += [str(permutation[idx]+1)]

        ### Vertical bar plot
        else:
            
            if branchIdx in list(range(1,13)): # smartphone channel
                plt.bar(barWidth*idx+1,sortedDiff[idx],color='b',label='smartphone',width=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_B'+str(branchIdx)+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:
                    layerAnnotations += [str(permutation[idx]+1)]

            elif branchIdx in list(range(13,19)): # smartwatch channel
                plt.bar(barWidth*idx+1,sortedDiff[idx],color='r',label='smartwatch',width=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_B'+str(branchIdx)+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:
                    layerAnnotations += [str(permutation[idx]+1)]

            elif branchIdx in list(range(19,25)): # smartglasses channel
                plt.bar(barWidth*idx+1,sortedDiff[idx],color='limegreen',label='smartglasses',width=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_B'+str(branchIdx)+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:
                    layerAnnotations += [str(permutation[idx]+1)]

            else:
                plt.bar(barWidth*idx+1,sortedDiff[idx],color='k',width=0.8*barWidth)
                if xlabels=='branches':
                    layerAnnotations += [layerType+'_D'+str(depth)]
                elif xlabels=='names':
                    layerAnnotations += [layerNames[idx]]
                else:
                    layerAnnotations += [str(permutation[idx]+1)]


    # Annotate the graph with layer names
    # Space layer annotations to make it easier to read
    for idx in range(len(layerAnnotations)):
        if idx %2 ==1:
            layerAnnotations[idx] = layerAnnotations[idx]+r' $\longleftarrow$  '

    if plotType == 'horizontal':  

        plt.yticks(barWidth*indices+1,layerAnnotations,fontsize=8,rotation=0)
        if xlabels != 'index':
            plt.ylabel('Layer name',fontsize=30)
        else:
            plt.ylabel(r'$k$',fontsize=30)
        plt.xlabel(r'$D^{(k)}$',fontsize=25)
    else:

        #plt.xticks(barWidth*indices+1,layerAnnotations,rotation='vertical',fontsize=8)
        if xlabels != 'index':
            plt.xlabel('Layer name',fontsize=30)
        else:
            plt.xlabel(r'$k$',fontsize=30)
        plt.ylabel(r'$D^{(k)}$',fontsize=25)

    ## Remove the tick lines
    #for tick in fig.axes[0].xaxis.get_major_ticks():
    #    tick.tick1On = tick.tick2On = False

    leg1 = mpatches.Patch(color='b',label='smartphone')
    leg2 = mpatches.Patch(color='r',label='smartwatch')
    leg3 = mpatches.Patch(color='limegreen',label='smartglasses')
    leg4 = mpatches.Patch(color='k',label='other')
    plt.legend(handles=[leg1,leg2,leg3,leg4],fontsize=30)
    plt.show()



################
# Main function
################
if __name__ == '__main__':

    #plt.rcParams['axes.grid'] = True
        
    ### Plot the score of one layer
    transfer = 'tto' # 'cnn-transfer' or 'tto'
    classification = 'bbh' # 'bbh' or 'blho'
    pathToFile = '../nisp-inffs/'+transfer+'-'+classification+'.pkl'
    layerName = 'max_pooling1d_40'
    plotLayerScore(pathToFile,layerName)

    ### Compute layer differences
    classification = 'bbh' # 'bbh' or 'blho'
    pathToFile1 = '../nisp-inffs/cnn-transfer-'+classification+'.pkl'
    pathToFile2 = '../nisp-inffs/tto-'+classification+'.pkl'
    layerNames, sortedDiff, permutation = computeLayerDifferences(pathToFile1,pathToFile2,display=False)

    ### Plot bar graph of layer differences ordered by decreasing value
    plotBarGraph(layerNames,sortedDiff,xlabels='index',permutation=permutation,plotType='vertical')

    ### Plot comparative scores between TTO and CNN-transfer for one layer of the mDNN
    plotComparativeLayerScore(pathToFile1,pathToFile2,'max_pooling1d_40',normalise=True)
