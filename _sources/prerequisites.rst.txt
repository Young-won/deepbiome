.. highlight:: shell

==============
Prerequisites
==============

Data preprocessing
==========================================

DeepBiome package takes microbiome abundance data as input and uses the phylogenetic taxonomy to guide the decision of the optimal number of layers and neurons in the deep learning architecture.


To use DeepBiome, you can experiment (1) **`k` times repetition** or (2) **`k` fold cross-validation**.
For each experiment, we asuume that the dataset is given by

    1. **A list of `k` input files for `k` times repetition.**, or
    2. **One input file for `k` fold cross-validation.**

With a list of `k` inputs for `k` times repetition
------------------------------------------------------

DeepBiome needs 4 data files as follows:

    1. **the tree information**  
    2. **the list of the input files** (each file has all sample's information for one repetition)  
    3. **the list of the names of input files**  
    4. **y**

For `k` times repetition, we can use the list of `k` input files. Each file has all sample's information for one repetition.
In addition, we can set **the training index for each repetition**. If we set the index file, DeepBiome builds the training set for each repetition based on each fold index in the index file. If not, DeepBiome will generate the index file locally.


Eath data should have the csv format as follows:

tree information (.csv)
    A file about the phylogenetic tree information.
    Below is an example file of the phylogenetic tree information dictionary

    .. csv-table:: Example of genus dictionary
        :class: longtable
        :file: data/genus48_dic.csv

list of the names of `k` input files (.csv)
    If we want to use the list of the input files, we need the make a list of the names of each input file.
    Below is an example file for `k=4` repetition. 
  
    .. csv-table:: Example of the list of `k` input file names
        :class: longtable
        :file: data/gcount_list.csv

list of `k` input files
    Each file should have each repetition's sample microbiome abandunce.
    Below is an example file for `k=4` repetition. This example is `gcount_0001.csv` for the first repetition in the list of the names of input files above. This file has the 4 samples' microbiome abandunce.
    
    .. csv-table:: Example of one input file (.csv) of `k` inputs.
        :class: longtable
        :file: data/gcount_0001.csv

y (.csv)
    One column contains y samples for one repetition. 
    Below is an example file for `k=4` repetition. For each repetition (column) has outputs of 4 samples for each repeatition.

    .. csv-table:: Example of y file (.csv)
        :class: longtable
        :file: data/y.csv

index for training set for each repetition (.csv)
    For each repetition, we have to set the training and test set. If the index file is given, DeepBiome sets the training set and test set based on the index file. Below is the example of the index file. Each column has the training indices for each repetition. DeepBiome will only use the samples in this index set for training. Below is an example for `k=4` repetition
    
    .. csv-table:: Example of index file (.csv)
        :class: longtable
        :file: data/idx.csv

In the example above, we used the first 3 rows of the first column in `y.csv` for the training set in the first repetition.



With one input file for `k` fold cross-validation
------------------------------------------------------

DeepBiome needs 3 data files as follows:

    1. **the tree information**  
    2. **the input file**  
    3. **y**

For `k` fold cross-validation, we can use an input file.
In addition, we can set **the training index for each fold**. If we set the index file, DeepBiome builds the training set for each fold based on each fold index in the index file. If not, DeepBiome will generate the index file locally.
        
Eath data should have the csv format as follows:

tree information (.csv)
    A file about the phylogenetic tree information.
    Below is an example file of the phylogenetic tree information dictionary

    .. csv-table:: Example of genus dictionary
        :class: longtable
        :file: data/genus48_dic.csv

input file
    Input file has the microbiome abandunce of each sample.
    Below is an example file with the 4 samples' microbiome abandunce.
    
    .. csv-table:: Example of input file (.csv)
        :class: longtable
        :file: data/X_onefile.csv

y (.csv)
    Below is an example file of the outputs of 4 samples.

    .. csv-table:: Example of y file (.csv)
        :class: longtable
        :file: data/y_onefile.csv

index for training set for each fold (.csv)
    For each fold, we have to set the training and test set. If the index file is given, DeepBiome sets the training set and test set based on the index file. Below is an example of the index file. Each column has the training indices for each fold. DeepBiome will only use the samples in this index set for training. Below is an example for `k=4` fold
    
    .. csv-table:: Example of index file (.csv)
        :class: longtable
        :file: data/idx.csv

In the example above, we used the first 3 rows of the first column in `y.csv` for the training set in the first fold.



Configuration
===================================

For detailed configuration, we used python dictionary as inputs for the main training function.

Preparing the configuration about the network information (`network_info`)
----------------------------------------------------------------------------

To give the information about the training hyper-parameter, we provide a dictionary of configuration to the `netowrk_info` field. Alternatively  we can use the configufation file (.cfg).

Configuration for the network training should include the information about:

:model_info: about the training method and metrics
:architecture_info: about the architecture options
:training_info: about the hyper-parameter for training (not required for testing and prediction)
:validation_info: about the hyper-parameter for validation (not required for testing and prediction)
:test_info: about the hyper-parameter for testing

.. note:: You don't have to fill the options if it has a default value.


network_info['model_info']
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed options for the `model_info` field are as follows.

:network_clas: DeepBiome network class (default='DeepBiomeNetwork').

:reader_class: reader classes

    ===================================  ================================================================================
    possible options                     explanation
    ===================================  ================================================================================
    "MicroBiomeRegressionReader"         Microbiome adandunce data reader for regression problem
    "MicroBiomeClassificationReader"     Microbiome adandunce data reader for classification problem
    ===================================  ================================================================================


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
         
:taxa_selection_metrics: metrics for the texa selection performance

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
:weight_decay: **DeepBiome with the phylogenetic tree based weight decay method** (default = "": without deepbiome weight decay method)

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
    DeepBiome                           "weight_initial"="glorot_uniform", "weight_deacy"="phylogenetic_tree"
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
    
:patience: patience for the EarlyStopping callback (integer; default = 20)
:min_delta: the minimum threshold for the ModelCheckpoint, EarlyStopping callbacks (float; default = 1e-4)


network_info['validation_info']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Detailed options for the `validation_info` field are as follows.

:validation_size: the ratio of the number of the samples in the validation set / the number of the samples in the training set(e.g. "0.2") 
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
            'taxa_selection_metrics': 'accuracy, sensitivity, specificity, gmeasure'
        },
        'training_info': {
            'batch_size': '200',
            'epochs': '10',
            'callbacks': 'ModelCheckpoint',
            'monitor': 'val_binary_accuracy',
            'mode': 'max',
            'min_delta': '1e-4',
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
    min_delta       = 1e-4

    [validation_info]
    validation_size = 0.2 
    batch_size = None

    [test_info]
    batch_size = None

.. hint::  See Example_ for reference about the configuration file example for various problems.



Preparing the configuration about the path information (`path_info`)
------------------------------------------------------------------------

To give the information about the path to dataset, paths for saving the trained weights and the evaluation results, we provide a dictionary of configurations to the `path_info` feild. Alternatively we can also use the configufation file (.cfg).

Your configuration for the paths should include the information about:

:data_info: about the path information of the dataset
:model_info: about the path information for saving the trained weights and the evaluation results

.. note:: All paths are the relative path based on the directory where code will run.


path_info['data_info']
~~~~~~~~~~~~~~~~~~~~~~~~~

To provide the dictionary as input, we can use the option below:

:tree_info_path: tree information file (.csv)
:count_list_path: lists of the name of input files (.csv)
:count_path: directory path of the input files
:y_path: y path (.csv)  (not required for prediction)
:idx_path: index path for repetation (.csv)
:data_path: directory path of the index and y file

To provide one configuration file, we can use the options below:

:tree_info_path: tree information file (.csv)
:x_path: input path (.csv)
:y_path: y path (.csv)  (not required for prediction)
:data_path: directory path of the index, x and y file


path_info['model_info']
~~~~~~~~~~~~~~~~~~~~~~~~~

:weight: weight file name (.h5)
:evaluation: evaluation file name (.npy)  (not required for prediction)
:model_dir: base directory path for the model (weight, evaluation)
:history: history file name for the history value of each evaluation metric from the training (.json). If not setted, `deepbiome` will not save the history of the network training.
        
.. warning:: If you want to use sub-directories in the path (for example, "weight"="weight/weight.h5", "history"="history/hist.h5", "model_dir"="./"), you should have to make the sub-directories "./weight" and "./history" before running the code.


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
            'history': 'hist.json',
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
    history = historys/hist.json
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
            'history':'history/hist.json',
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
    history = history/hist.json
    evaluation = eval.npy


.. hint::  See Example_ for reference about the configuration file example for various problems.



.. _Example: https://github.com/Young-won/deepbiome/tree/master/examples

.. _Optimizer: https://keras.io/optimizers/

.. _Callbacks: https://keras.io/callbacks/

.. _ModelCheckpoint: https://keras.io/callbacks/#modelcheckpoint

.. _EarlyStopping: https://keras.io/callbacks/#earlystopping

.. _Tensorboard: https://keras.io/callbacks/#tensorboard