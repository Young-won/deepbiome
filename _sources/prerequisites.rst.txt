.. highlight:: shell

==============
Prerequisites
==============

Data preprocessing
==========================================

`deepbiome` packages takes microbiome abundance data as input and uses the phylogenetic taxonomy to guide the decision of the optimal number of layers and neurons in the deep learning architecture.


To use `deepbiome`, you can use the microbiome abundance data with the form:

    1. A list of input files for `k` times repeatition or `k`-fold cross validation.
    2. One input file for `k` times cross validation.


With the list of inputs
------------------------------------------------------

Dataset should have the format as follow:

tree information (.csv)
    Example of the genus dictionary

    .. csv-table:: Example of genus dictionary
        :class: longtable
        :file: data/genus48_dic.csv

lists of the name of input files (.csv)
    Example with 4 sample, 4 repeatition (4-fold cross validation)
  
    .. csv-table:: Example of input file name list
        :class: longtable
        :file: data/gcount_list.csv


directory contain the list of the input files
    Example with 4 sample, 4 repeatition (4-fold cross validation)
    
    .. csv-table:: Example of each input file (.csv)
        :class: longtable
        :file: data/gcount_0001.csv

y (.csv)
    Example with 4 sample, 4 repeatition (4-fold cross validation)
    
    One column for one repeatition

    .. csv-table:: Example of y file (.csv)
        :class: longtable
        :file: data/y.csv

index for training set for each repetation (.csv)
    Example with 4 sample, 4 repeatition (4-fold cross validation)
    
    One column for one repeatition.
    
    .. csv-table:: Example of index file (.csv)
        :class: longtable
        :file: data/idx.csv

In the example above, we used the first 3 row of the first column in `y.csv` for the training set in the first repeatition.


With the one input file
------------------------------------------------------

Instead of the lists of the name of input files and the directory contain the list of the input files, 
you can use the input file (.csv).

Dataset should have the format as follow:

input (.csv)
    Example with 4 sample
    
    .. csv-table:: Example of input file (.csv)
        :class: longtable
        :file: data/X_onefile.csv

y (.csv)
    Example with 4 sample
    
    .. csv-table:: Example of y file (.csv)
        :class: longtable
        :file: data/y_onefile.csv


Configuration
===================================

For detailed configuration, we used python dictionary as inputs for the main training function.

Preparing the configuration about the network information (`network_info`)
----------------------------------------------------------------------------

For giving the information about the training hyper-parameter, you have to provide the dictionary for configuration to `netowrk_info` field. 

You can build the configuration information for the network training by dictionary format or the configufation file (.cfg).

Your configuration for the network training should include the information about:

:model_info: about the training method and metrics
:architecture_info: about the architecture options
:training_info: about the hyper-parameter for training
:validation_info: about the hyper-parameter for validation
:test_info: about the hyper-parameter for testing

.. note:: You don't have to fill the options if it has a default value.


network_info['model_info']
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed options for the `model_info` field are as follows.

:network_clas: deepbiome network class (default='DeepBiomeNetwork').

:reader_class: reader classes

    ===================================  ================================================================================
    possible options                     explanation
    ===================================  ================================================================================
    "MicroBiomeRegressionReader"         Microbiome adandunce data reader for regression problem
    "MicroBiomeClassificationReader"     Microbiome adandunce data reader for classification problem
    ===================================  ================================================================================


:optimizer: optimization methods for training the network. We used the optimizers implemented in Keras (See Optimizer_).


:optimizer: optimization methods for training the network. We used the optimizers implemented in Keras (See Optimizer_).

    ====================  ================================================================================
    possible options      explanation
    ====================  ================================================================================
    "adam"                Adam optimizer
    "sgd"                 stocastic gradient decent optimizer
    ====================  ================================================================================

:lr: learning rate for the optimizor. (float between 0 ~ 1)
:decay: learning late decay ratio for the optimizer. (float between 0 ~ 1)
:loss: loss functions for training the network

    ============================  ================================================================================
    possible options              explanation
    ============================  ================================================================================
    "mean_squared_error"          for regression problem
    "binary_crossentropy"         for binary classification problem
    "categorical_crossentropy"    for multi-class classification problem
    ============================  ================================================================================

:metrics: additional metrics to check the model performance

    ============================  ================================================================================
    possible options              explanation
    ============================  ================================================================================
    "correlation_coefficient"     Pearson correlation coefficient (-1 ~ 1)
    "binary_accuracy"             Accuracy for binary classification problem (0 ~ 1)
    "categorical_accuracy"        Accuracy for multi-class classification problem (0 ~ 1)
    "sensitivity"                 Sensitivity (0 ~ 1)
    "specificity"                 Specificity (0 ~ 1)
    "gmeasure"                    (Sensitivity * Specificity) ^ (0.5) (0 ~ 1)
    "auc"                         Area under the receiver operating characteristics (0 ~ 1)
    "precision"                   Precision (0 ~ 1)
    "recall"                      Recall (0 ~ 1)
    "f1"                          F1 score (0 ~ 1)
    ============================  ================================================================================
         
:texa_selection_metrics: metrics for the texa selection performance

    ============================  ================================================================================
    possible options              explanation
    ============================  ================================================================================
    "accuracy"                    Accuracy (-1 ~ 1)
    "sensitivity"                 Sensitivity (0 ~ 1)
    "specificity"                 Specificity (0 ~ 1)
    "gmeasure"                    (Sensitivity * Specificity) ^ (0.5) (0 ~ 1)
    ============================  ================================================================================
    
:normalizer: normalizer for the input data (default=`normalize_minmax`)


network_info['architecture_info']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed options for the `architecture_info` field are as follows.

Combination of the options below will provide you the network training method `DNN`, `DNN+L1` and `Deepbiome` in the reference (url. TBD)


:weight_initial: network weight initialization

    ==================================  ========================================================================================================
    possible options                    explanation
    ==================================  ========================================================================================================
    "glorot_uniform"                    Glorot uniform initializer (defualt)
    "he_normal"                         He normal initializer
    "phylogenetic_tree"                 weight within the tree connection: 1; weight without the tree connection: 0
    "phylogenetic_tree_glorot_uniform"  weight within the tree connection: `glorot_uniform`; weight without the tree connection: 0
    "phylogenetic_tree_he_normal"       weight within the tree connection: `he_normal`; weight without the tree connection: 0
    ==================================  ========================================================================================================
    
:weight_l1_penalty: :math:`\lambda` for l1 penalty (float. defaut = 0)
:weight_l2_penalty: :math:`\lambda` for l2 penalty (float. defaut = 0)
:weight_deacy: **DeepBiome with the phylogenetic tree based weight decay method** (default = "": without deepbiome weight decay method)

    ==================================  ===========================================================================================================
    possible options                    explanation
    ==================================  ===========================================================================================================
    "phylogenetic_tree"                 weight decay method based on the phylogenetic tree information with small amout of noise (:math:`\epsilon \le 1e-2`)
    "phylogenetic_tree_wo_noise"        weight decay method based on the phylogenetic tree information without any noise outside the tree
    ==================================  ===========================================================================================================
    
:batch_normalization: options for adding the batch normalization for each convolutional layer (default = `False`)
:drop_out: options for adding the drop out for each convolutional layer with given ratio (default = 0)

.. hint::  Example of the combination of the options in the reference paper (url TBD):

    ==================================  ===========================================================================================================
    training method                     combination of the options
    ==================================  ===========================================================================================================
    DNN                                 "weight_initial"="glorot_uniform"
    DNN+L1                              "weight_initial"="glorot_uniform", "weight_l1_penalty"="0.01"
    Deepbiome                           "weight_initial"="glorot_uniform", "weight_deacy"="phylogenetic_tree"
    ==================================  ===========================================================================================================


network_info['training_info']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed options for the `training_info` field are as follows.

:epochs: number of the epoch for training (integer)
:batch_size: number of the batch size for each mini-batch (integer)
:callbacks: callback class implemented in Keras (See Callbacks_)

    ============================  ===============================================================================================================
    possible options              explanation
    ============================  ===============================================================================================================
    "ModelCheckpoint"             save the best model weight based on the monitor (See ModelCheckpoint_)
    "EarlyStopping"               early stopping the training before the number of epochs `epochs` based on the monitor (See EarlyStopping_)
    ============================  ===============================================================================================================
    
:monitor: monitor value for the `ModelCheckpoint`, `EarlyStoppoing` callbacks (e.g.  `val_loss`, `val_accuray`)
:mode: how to use the monitor value for the `ModelCheckpoint`, `EarlyStopping` callbacks 

    ============================  ================================================================================
    possible options              explanation
    ============================  ================================================================================
    "min"                         for example: when using the monitor `val_loss`
    "max"                         for example: when using the monitor `val_accuray`
    ============================  ================================================================================
    
:patience: patient for the EarlyStopping callback (integer; default = 20)
:min_delta: the minimum threshold for the ModelCheckpoint, EarlyStopping callbacks (float; default = 1e-4)


network_info['validation_info']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Detailed options for the `validation_info` field are as follows.

:validation_size: the ratio of the number of the sample in the validation set / the number of the sample in the training set(e.g. "0.2") 
:batch_size: the batch size for each mini-batch. If "None", use the whole number of the sample as one mini-batch. (defualt = "None")

network_info['test_info']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed options for the `test_info` field are as follows.

:batch_size: the batch size for each mini-batch. If "None", use the whole number of the sample as one mini-batch. (defualt = "None")

Example for the `network_info`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the example of the configuration dictionary: `network_info` dictionary


.. code-block:: python

    network_info = {
        'architecture_info': {
            'batch_normalization': 'False',
            'drop_out': '0',
            'weight_initial': 'glorot_uniform',
            'weight_l1_penalty':'0.01',
            'weight_decay': 'phylogenetic_tree',
        },
        'model_info': {
            'decay': '0.001',
            'loss': 'binary_crossentropy',
            'lr': '0.01',
            'metrics': 'binary_accuracy, sensitivity, specificity, gmeasure, auc',
            'network_class': 'DeepBiomeNetwork',
            'normalizer': 'normalize_minmax',
            'optimizer': 'adam',
            'reader_class': 'MicroBiomeClassificationReader',
            'texa_selection_metrics': 'accuracy, sensitivity, specificity, gmeasure'
        },
        'training_info': {
            'batch_size': '200', 'epochs': '10'
        },
        'validation_info': {
            'batch_size': 'None', 'validation_size': '0.2'
        },
        'test_info': {
            'batch_size': 'None'
        }
    }


This is the example of the configuration file: `network_info.cfg`

.. code-block:: cfg

    [model_info]
    network_class = DeepBiomeNetwork
    optimizer   = adam
    lr          = 0.01
    decay       = 0.0001
    loss        = binary_crossentropy
    metrics     = binary_accuracy, sensitivity, specificity, gmeasure, auc
    texa_selection_metrics = accuracy, sensitivity, specificity, gmeasure
    reader_class = MicroBiomeClassificationReader
    normalizer  = normalize_minmax

    [architecture_info]
    weight_initial = glorot_uniform
    weight_decay = phylogenetic_tree
    batch_normalization = False
    drop_out = 0

    [training_info]
    epochs          = 1000
    batch_size      = 200 
    callbacks       = ModelCheckpoint
    monitor         = val_binary_accuracy
    mode            = max
    patience        = 20
    min_delta       = 1e-4

    [validation_info]
    validation_size = 0.2 
    batch_size = None

    [test_info]
    batch_size = None

.. hint::  See Example_ for reference about the configuration file example for various problems.



Preparing the configuration about the path information (`path_info`)
------------------------------------------------------------------------

For giving the information about the path of dataset, paths for saving the trained weight and the evaluation results, you have to provide the dictionary for configuration to `path_info` feild.

You can build the configuration information for the network training by dictionary format or the configufation file (.cfg).

Your configuration for the paths should include the information about:

:data_info: about the path information of the dataset
:model_info: about the path information for saving the trained weight and the evaluation results

.. note:: Every paths are the relative path based on the directory that code will run.


path_info['data_info']
~~~~~~~~~~~~~~~~~~~~~~~~~

If you provide the list of inputs, you can use the option below:

:tree_info_path: tree information file (.csv)
:count_list_path: lists of the name of input files (.csv)
:count_path: directory path of the input files
:y_path: y path (.csv)
:idx_path: index path for repetation (.csv)
:data_path: directory path of the index and y file

If you provide the one input file, you can use the option below:

:tree_info_path: tree information file (.csv)
:x_path: input path (.csv)
:y_path: y path (.csv)
:data_path: directory path of the index, x and y file


path_info['model_info']
~~~~~~~~~~~~~~~~~~~~~~~~~

:weight: weight file name (.h5)
:evaluation: evaluation file name (.npy)
:model_dir: base directory path for the model (weight, evaluation)

.. warning:: If you want to use sub-directory in the path (for example, "weight"="weight/weight.h5", "model_dir"="./"), you should have to make the sub-directory "./weight" before running the code.


Example for the `path_info` for the list of inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the example of the configuration dictionary: `path_info` dictionary


.. code-block:: python

    path_info = {
        'data_info': {
            'count_list_path': 'data/simulation/gcount_list.csv',
            'count_path': 'data/simulation/count',
            'data_path': 'data/simulation/s2/',
            'idx_path': 'data/simulation/s2/idx.csv',
            'tree_info_path': 'data/genus48/genus48_dic.csv',
            'x_path': '',
            'y_path': 'y.csv'
        },
        'model_info': {
            'model_dir': './simulation_s2/simulation_s2_deepbiome/',
            'weight': 'weight/weight.h5',
            'evaluation': 'eval.npy'
        }
    }


This is the example of the configuration file: `path_info.cfg`

.. code-block:: cfg

    [data_info]
    data_path = data/simulation/s2/
    tree_info_path = data/genus48/genus48_dic.csv
    idx_path = data/simulation/s2/idx.csv
    count_list_path = data/simulation/gcount_list.csv
    count_path = data/simulation/count
    y_path = y.csv

    [model_info]
    model_dir = ./simulation_s2/simulation_s2_deepbiome/
    weight = weight/weight.h5
    evaluation = eval.npy


Example for the `path_info` for the one input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the example of the configuration dictionary: `path_info` dictionary


.. code-block:: python

    path_info = {
        'data_info': {
            'data_path': '../../data/pulmonary/',
            'tree_info_path': '../../data/genus48/genus48_dic.csv',
            'x_path': 'X.csv',
            'y_path': 'y.csv'
        },
        'model_info': {
            'model_dir': './',
            'weight': 'weight/weight.h5',
            'evaluation': 'eval.npy',
        }
    }


This is the example of the configuration file: `path_info.cfg`

.. code-block:: cfg

    [data_info]
    data_path = ../../data/pulmonary/
    tree_info_path = ../../data/genus48/genus48_dic.csv
    x_path = X.csv
    y_path = y.csv

    [model_info]
    model_dir = ./
    weight = weight/weight.h5
    evaluation = eval.npy


.. hint::  See Example_ for reference about the configuration file example for various problems.



.. _Example: https://github.com/Young-won/deepbiome/tree/master/examples

.. _Optimizer: https://keras.io/optimizers/

.. _Callbacks: https://keras.io/callbacks/

.. _ModelCheckpoint: https://keras.io/callbacks/#modelcheckpoint

.. _EarlyStopping: https://keras.io/callbacks/#earlystopping

.. _Tensorboard: https://keras.io/callbacks/#tensorboard