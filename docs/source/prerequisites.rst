.. highlight:: shell

=====
Prerequisites
=====

Data preprocessing
==========================================

`deepbiome` packages takes microbiome abundance data as input and uses the phylogenetic taxonomy to guide the decision of the optimal number of layers and neurons in the deep learning architecture.


To use `deepbiome`, you can use the microbiome abundance data with the form:

    1. A list of input files for `k` times repeatition or `k`-fold cross validation.
    2. One input file for `k` times cross validation.


With the list of inputs
------------------------------------------------------




With the one input file
------------------------------------------------------






Configuration
===================================

Preparing the configuration about the network information
------------------------------------------------------

You can build the configuration information for the network training by dictionary format or the configufation file (.cfg).
Your configuration for the network training should include the information about `model_info`, `architecture_info`,  and `tensorboard_info`.


Details about the `model_info` are as follow:

optimizer:
    optimization methods for training the network.
    possible options:
        adam
        sgd
lr:
    learning rate for the optimizor.
    float between 0 ~ 1
    
decay:
     learning late decay ratio for the optimizer.
     float between 0 ~ 1
loss:
    loss functions for training the network
    possible options:
        mean_squared_error
        binary_crossentropy
        categorical_crossentropy
        
metrics:
    additional metrics to check the model perforance
    possible options:
        correlation_coefficient
        binary_accuracy
        sensitivity
        specificity
        gmeasure
        auc
        precision
        recall
        f1
        
texa_selection_metrics:
    metrics to calculate the texa selection performance
    possible options:
        accuracy
        sensitivity
        specificity
        gmeasure














This is the example of the configuration dictionary: `network_info` dictionary




This is the example of the configuration file: `network_info.cfg`




Preparing the configuration about the path information
------------------------------------------------------




This is the example of the `path_info` dictionary.