# Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification

This repository hosts the code that was used for the publication:

**[*]** Frédéric Li, Kimiaki Shirahama, Muhammad Adeel Nisar, Xinyu Huang and Marcin Grzegorzek, **Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification**, _Sensors_ (MDPI), Vol. 20, Issue 15, 2020, https://doi.org/10.3390/s20154271

It contains the following Python codes used for the study in **[*]**: 
 - **./trainSensorClassifierOnSourceData.py**: script to train on the source domain single-channel DNNs in a supervised way (sensor modality classification)
 - **./trainConvolutionalVaeOnSourceData.py**: script to train a CNN Variational Autoencoder on the source domain (to be transferred on the target domain)
 - **./trainMultichannelDnnDeap.py**: script to train and evaluate a Multichannel DNN on the DEAP dataset
 - **./trainMultichannelDnnCogAge.py**: script to train and evaluate a Multichannel DNN on the CogAge dataset
 - **./computeNeuronImportanceScores.py**: script to compute neuron importance scores of a trained mDNN. Contains implementations of NISP and InfFS.
 - **./nispPlottingScripts.py**: script containing various functions to plot NISP+InfFS scores or layer differences between TTO and CNN-transfer.
 - **./utilitary.py**: contains various functions and attributes used by the other scripts

All codes were used and tested with Python 2.7.12, Keras 2.2.4 with Tensorflow 1.13.1 as backend. The additional following Python libraries are required for a proper execution of the code:
- numpy
- scipy
- sklearn

The scripts are intended to be used in the following order:
- 1- Build and save the source dataset: buildSourceDataset.py
- 2- Train a single-channel DNN (using either the sensor-based classification or the variational autoencoder approach) and save it: trainSensorClassifierOnSourceData.py or trainConvolutionalVaeOnSourceData.py
- 3- Train a Multichannel DNN on DEAP or CogAge (with or without a transfer of weights): trainMultichannelDnnDeap.py or trainMultichannelDnnCogAge.py
    
To use a script in particular:
- 1- Update the hyper-parameter section at the beginning of every script with the desired values
- 2- Execute the script


The data used in the experiments of [*] can be found at the following webpage: https://ccilab.doshisha.ac.jp/shirahama/research/transfer/
