######################################################################
## DeepBiome
## - Main code
##
## July 10. 2019
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings("ignore")
import logging
from sklearn.model_selection import KFold

from . import logging_daily
from . import configuration
from . import loss_and_metric
from . import readers
from . import build_network
from .utils import file_path_fold, argv_parse, taxa_selection_accuracy

import keras.backend as k  
import tensorflow as tf

import copy
from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle, CircleFace
import matplotlib.colors as mcolors


def deepbiome_train(log, network_info, path_info, number_of_fold=None, 
                    max_queue_size=10, workers=1, use_multiprocessing=False):
    """
    Function for training the deep neural network with phylogenetic tree weight regularizer.
    
    It uses microbiome abundance data as input and uses the phylogenetic taxonomy to guide the decision of the optimal number of layers and neurons in the deep learning architecture.

    See ref url (TODO: update)

    Parameters
    ----------
    log (logging instance) :
        python logging instance for logging
    network_info (dictionary) :
        python dictionary with network_information
    path_info (dictionary):
        python dictionary with path_information
    number_of_fold (int):
        default=None
    max_queue_size (int):
        default=10
    workers (int):
        default=1
    use_multiprocessing (boolean):
        default=False

    Returns
    -------
    test_evaluation (numpy array):
        numpy array of the evaluation using testset from all fold
    train_evaluation (numpy array):
        numpy array of the evaluation using training from all fold
    network (deepbiome network instance):
        deepbiome class instance
    
    Examples
    --------
    Training the deep neural network with phylogenetic tree weight regularizer.

    test_evaluation, train_evaluation, network = deepbiome_train(log, network_info, path_info)
    """
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    ### Argument #########################################################################################
    model_save_dir = path_info['model_info']['model_dir']
    model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
    try:
        hist_path = os.path.join(model_save_dir, path_info['model_info']['history'])
        is_save_hist = True
    except:
        is_save_hist = False
        
    try:
        warm_start = network_info['training_info']['warm_start'] == 'True'
        warm_start_model = network_info['training_info']['warm_start_model']
    except:
        warm_start = False
    # try: save_frequency=int(network_info['training_info']['save_frequency'])
    # except: save_frequency=None

    ### Reader ###########################################################################################
    log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    # TODO: fix path_info
    reader = reader_class(log, path_info, verbose=True)

    data_path = path_info['data_info']['data_path']
    y_path = '%s/%s'%(data_path, path_info['data_info']['y_path'])
    
    ############################################
    # Set the cross-validation
    try:
        idxs = np.array(pd.read_csv(path_info['data_info']['idx_path'])-1, dtype=np.int)
        if number_of_fold == None:
            number_of_fold = idxs.shape[1] 
    except:
        nsample = pd.read_csv(y_path).shape[0]
        if number_of_fold == None:
            number_of_fold = nsample
        kf = KFold(n_splits=number_of_fold, shuffle=True, random_state=12)
        cv_gen = kf.split(range(nsample))
        idxs = np.array([train_idx for train_idx, test_idx in cv_gen]).T
     ############################################
    
    try:
        count_path = path_info['data_info']['count_path']
        x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None).iloc[:,0])
        x_path = np.array(['%s/%s'%(count_path, x_list[fold]) for fold in range(x_list.shape[0]) if '.csv' in x_list[fold]])
    except:
        x_path = np.array(['%s/%s'%(data_path, path_info['data_info']['x_path']) for fold in range(number_of_fold)])

    ### Simulations #################################################################################
    # if is_save_hist: history = []
    train_evaluation = []
    test_evaluation = []
    # train_tot_idxs = []
    # test_tot_idxs = []
    starttime = time.time()
    for fold in range(number_of_fold):
        log.info('-------%d simulation start!----------------------------------' % (fold+1))
        foldstarttime = time.time()

        ### Read datasets ####################################################################################
        reader.read_dataset(x_path[fold], y_path, fold)
        x_train, x_test, y_train, y_test = reader.get_dataset(idxs[:,fold])
        num_classes = reader.get_num_classes()

        ### Bulid network ####################################################################################
        if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
        log.info('-----------------------------------------------------------------')
        log.info('Build network for %d simulation' % (fold+1))
        network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
        network = network_class(network_info, path_info, log, fold, num_classes=num_classes,
                                is_covariates=reader.is_covariates, covariate_shape = reader.covariate_shape)
        network.model_compile() ## TODO : weight clear only (no recompile)
        if warm_start:
            network.load_weights(file_path_fold(warm_start_model, fold))
        sys.stdout.flush()

        ### Training #########################################################################################
        log.info('-----------------------------------------------------------------')
        log.info('%d fold computing start!----------------------------------' % (fold+1))
        hist = network.fit(x_train, y_train, 
                           max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
                           model_path=file_path_fold(model_path, fold))
        # if is_save_hist: history.append(hist.history)
        sys.stdout.flush()

        network.save_weights(file_path_fold(model_path, fold))
        log.debug('Save weight at {}'.format(file_path_fold(model_path, fold)))
        if is_save_hist: 
            network.save_history(file_path_fold(hist_path, fold), hist.history)
            log.debug('Save history at {}'.format(file_path_fold(hist_path, fold)))
        sys.stdout.flush()

        # Evaluation
        train_eval_res = network.evaluate(x_train, y_train)
        train_evaluation.append(train_eval_res)
        
        test_eval_res = network.evaluate(x_test, y_test)
        test_evaluation.append(test_eval_res)

        if not tf.__version__.startswith('2'): k.clear_session()
        log.info('Compute time : {}'.format(time.time()-foldstarttime))
        log.info('%d fold computing end!---------------------------------------------' % (fold+1))

    ### Summary #########################################################################################
    log.info('-----------------------------------------------------------------')
    train_evaluation = np.vstack(train_evaluation)
    log.info('Train Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
    mean = np.mean(train_evaluation, axis=0)
    std = np.std(train_evaluation, axis=0)
    log.info('      mean : %s',mean)
    log.info('       std : %s',std)
    log.info('-----------------------------------------------------------------')
    test_evaluation = np.vstack(test_evaluation)
    log.info('Test Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
    mean = np.mean(test_evaluation, axis=0)
    std = np.std(test_evaluation, axis=0)
    log.info('      mean : %s',mean)
    log.info('       std : %s',std)
    log.info('-----------------------------------------------------------------')

    ### Save #########################################################################################
    np.save(os.path.join(model_save_dir, 'train_%s'%path_info['model_info']['evaluation']),train_evaluation)
    np.save(os.path.join(model_save_dir, 'test_%s'%path_info['model_info']['evaluation']),test_evaluation)
    # np.save(os.path.join(model_save_dir, path_info['model_info']['train_tot_idxs']), np.array(train_tot_idxs))
    # np.save(os.path.join(model_save_dir, path_info['model_info']['test_tot_idxs']), np.array(test_tot_idxs))

    ### Exit #########################################################################################
    log.info('Total Computing Ended')
    log.info('-----------------------------------------------------------------')
    gc.collect()
    return test_evaluation, train_evaluation, network


def deepbiome_test(log, network_info, path_info, number_of_fold=None,
                   max_queue_size=10, workers=1, use_multiprocessing=False):
    """
    Function for testing the pretrained deep neural network with phylogenetic tree weight regularizer. 
    
    If you use the index file, this function provide the evaluation using test index (index set not included in the index file) for each fold. If not, this function provide the evaluation using the whole samples.

    See ref url (TODO: update)

    Parameters
    ----------
    log (logging instance) :
        python logging instance for logging
    network_info (dictionary) :
        python dictionary with network_information
    path_info (dictionary):
        python dictionary with path_information
    number_of_fold (int):
        If `number_of_fold` is set as `k`, the function will test the model only with first `k` folds.
        default=None
    max_queue_size (int):
        default=10
    workers (int):
        default=1
    use_multiprocessing (boolean):
        default=False
    
    Returns
    -------
    evaluation (numpy array):
        evaluation result using testset as a numpy array with a shape of (number of fold, number of evaluation measures)
    
    Examples
    --------
    Test the pre-trained deep neural network with phylogenetic tree weight regularizer.
    
    evaluation = deepbiome_test(log, network_info, path_info)
    """
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    ### Argument #########################################################################################
    model_save_dir = path_info['model_info']['model_dir']
    model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
    
    ### Reader ###########################################################################################
    log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=True)

    data_path = path_info['data_info']['data_path']
    y_path = '%s/%s'%(data_path, path_info['data_info']['y_path'])
    
    ############################################
    # Set the cross-validation
    try:
        idxs = np.array(pd.read_csv(path_info['data_info']['idx_path'])-1, dtype=np.int)
        if number_of_fold == None:
            number_of_fold = idxs.shape[1] 
    except:
        nsample = pd.read_csv(y_path).shape[0]
        if number_of_fold == None:
            number_of_fold = pd.read_csv(y_path).shape[1]
        try: idxs = np.array([np.arange(nsample) for i in range(number_of_fold)]).T
        except: idxs = np.array([np.arange(nsample)]).T
     ############################################
    
    try:
        count_path = path_info['data_info']['count_path']
        x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None).iloc[:,0])
        x_path = np.array(['%s/%s'%(count_path, x_list[fold]) for fold in range(x_list.shape[0]) if '.csv' in x_list[fold]])
    except:
        x_path = np.array(['%s/%s'%(data_path, path_info['data_info']['x_path']) for fold in range(number_of_fold)])

    ### Simulations #################################################################################
    train_evaluation = []
    test_evaluation = []
    starttime = time.time()
    log.info('Test Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
    for fold in range(number_of_fold):
        log.info('-------%d fold test start!----------------------------------' % (fold+1))
        foldstarttime = time.time()

        ### Read datasets ####################################################################################
        reader.read_dataset(x_path[fold], y_path, fold)
        x_train, x_test, y_train, y_test = reader.get_dataset(idxs[:,fold])
        num_classes = reader.get_num_classes()

        ### Bulid network ####################################################################################
        if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
        log.info('-----------------------------------------------------------------')
        log.info('Build network for %d fold testing' % (fold+1))
        network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
        network = network_class(network_info, path_info, log, fold, num_classes=num_classes, 
                                is_covariates=reader.is_covariates, covariate_shape = reader.covariate_shape)
        network.model_compile() ## TODO : weight clear only (no recompile)
        network.load_weights(file_path_fold(model_path, fold), verbose=False)
        sys.stdout.flush()

        ### Training #########################################################################################
        log.info('-----------------------------------------------------------------')
        log.info('%d fold computing start!----------------------------------' % (fold+1))
        test_eval_res = network.evaluate(x_test, y_test)
        test_evaluation.append(test_eval_res)
        log.info('' % test_eval_res)
        if not tf.__version__.startswith('2'): k.clear_session()
        log.info('Compute time : {}'.format(time.time()-foldstarttime))
        log.info('%d fold computing end!---------------------------------------------' % (fold+1))

    ### Summary #########################################################################################
    log.info('-----------------------------------------------------------------')
    test_evaluation = np.vstack(test_evaluation)
    log.info('Test Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
    mean = np.mean(test_evaluation, axis=0)
    std = np.std(test_evaluation, axis=0)
    log.info('      mean : %s',mean)
    log.info('       std : %s',std)
    log.info('-----------------------------------------------------------------')

    ### Exit #########################################################################################
    log.info('Total Computing Ended')
    log.info('-----------------------------------------------------------------')
    gc.collect()
    return test_evaluation


def deepbiome_prediction(log, network_info, path_info, num_classes, number_of_fold=None,
                         change_weight_for_each_fold=False, get_y = False,
                         max_queue_size=10, workers=1, use_multiprocessing=False):
    """
    Function for prediction by the pretrained deep neural network with phylogenetic tree weight regularizer. 
    
    See ref url (TODO: update)

    Parameters
    ----------
    log (logging instance) :
        python logging instance for logging
    network_info (dictionary) :
        python dictionary with network_information
    path_info (dictionary):
        python dictionary with path_information
    num_classes (int):
        number of classes for the network. 0 for regression, 1 for binary classificatin.
    number_of_fold (int):
        1) For the list of input files for repeatitions, the function will predict the output of the first `number_of_fold` repetitions. If `number_of_fold` is None, then the function will predict the output of the whole repetitions.
        
        2) For the one input file for cross-validation, the function will predict the output of the `k`-fold cross validatoin. If `number_of_fold` is None, then the function will predict the output of the LOOCV.
        
        default=None
    change_weight_for_each_fold (boolean):
        If `True`, weight will be changed for each fold (repetition). For example, if the given weight's name is `weight.h5` then `weight_0.h5` will loaded for the first fold (repetition). If `False`, weight path in the path_info will used for whole prediction. For example, if the given weight's name is `weight_0.h5` then `weight_0.h5` will used for whole fold (repetition).
        default=False
    get_y (boolean):
        If 'True', the function will provide a list of tuples (prediction, true output) as a output.
        degault=False
    max_queue_size (int):
        default=10
    workers (int):
        default=1
    use_multiprocessing (boolean):
        default=False
    
    Returns
    -------
    prediction (numpy array):
        prediction using whole dataset in the data path
    
    Examples
    --------
    Prediction by the pre-trained deep neural network with phylogenetic tree weight regularizer.
    
    prediction = deepbiome_predictoin(log, network_info, path_info, num_classes)
    
    For LOOCV prediction, we can use this options.
    prediction = deepbiome_predictoin(log, network_info, path_info, num_classes, number_of_fold=None, change_weight_for_each_fold=True)
    """
    
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    ### Argument #########################################################################################
    model_save_dir = path_info['model_info']['model_dir']
    model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
    
    ### Reader ###########################################################################################
    log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=True)

    data_path = path_info['data_info']['data_path']
    if get_y: y_path = '%s/%s'%(data_path, path_info['data_info']['y_path'])
    ############################################################
    # Set the cross-validation
    try:
        idxs = np.array(pd.read_csv(path_info['data_info']['idx_path'])-1, dtype=np.int)
        if number_of_fold == None:
            number_of_fold = idxs.shape[1]
    except:
        # TODO: check
        if number_of_fold == None:
            number_of_fold = 1
        idxs = None
     ############################################################
    try:
        count_path = path_info['data_info']['count_path']
        x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None).iloc[:,0])
        x_path = np.array(['%s/%s'%(count_path, x_list[fold]) for fold in range(x_list.shape[0]) if '.csv' in x_list[fold]])
    except:
        x_path = np.array(['%s/%s'%(data_path, path_info['data_info']['x_path']) for fold in range(number_of_fold)])
    ############################################################
    
    starttime = time.time()
    if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
    prediction = []
    for fold in range(number_of_fold):
        log.info('-------%d th repeatition prediction start!----------------------------------' % (fold+1))
        foldstarttime = time.time()

        ### Read datasets ####################################################################################
        if get_y:
            reader.read_dataset(x_path[fold], y_path, fold)
            if not np.all(idxs == None):
                x_train, x_test, y_train, y_test = reader.get_dataset(idxs[:,fold])
            else:
                x_test, _, y_test, _ = reader.get_dataset()
        else:
            reader.read_dataset(x_path[fold], None, fold)
            if not np.all(idxs == None):
                x_train, x_test = reader.get_input(idxs[:,fold])
            else:
                x_test, _ = reader.get_input()

        ### Bulid network ####################################################################################
        log.info('-----------------------------------------------------------------')
        network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
        network = network_class(network_info, path_info, log, fold=fold, num_classes=num_classes,
                                is_covariates=reader.is_covariates, covariate_shape = reader.covariate_shape)
        network.model_compile()
        if change_weight_for_each_fold:network.load_weights(file_path_fold(model_path, fold), verbose=False)
        else: network.load_weights(model_path, verbose=False)
        sys.stdout.flush()

        ### Training #########################################################################################
        log.info('-----------------------------------------------------------------')
        pred = network.predict(x_test)
        if get_y: prediction.append(np.array(list(zip(pred, y_test))))
        else: prediction.append(pred)
        log.info('Compute time : {}'.format(time.time()-foldstarttime))
        log.info('%d fold computing end!---------------------------------------------' % (fold+1))

    ### Exit #########################################################################################
    if not tf.__version__.startswith('2'): k.clear_session()
    prediction = np.array(prediction)
    log.info('Total Computing Ended')
    log.info('-----------------------------------------------------------------')
    gc.collect()
    return prediction

def deepbiome_get_trained_weight(log, network_info, path_info, num_classes, weight_path):
    """
    Function for prediction by the pretrained deep neural network with phylogenetic tree weight regularizer. 
    
    See ref url (TODO: update)

    Parameters
    ----------
    log (logging instance) :
        python logging instance for logging
    network_info (dictionary) :
        python dictionary with network_information
    path_info (dictionary):
        python dictionary with path_information
    num_classes (int):
        number of classes for the network. 0 for regression, 1 for binary classificatin.
    weight_path (string):
        path of the model weight
    
    Returns
    -------
    list of pandas dataframe:
        the trained model's weight
    
    Examples
    --------
    Trained weight of the deep neural network with phylogenetic tree weight regularizer.
    
    tree_weight_list = deepbiome_get_trained_weight(log, network_info, path_info, num_classes, weight_path)
    """
    
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=True)
    data_path = path_info['data_info']['data_path']
    try:
        count_path = path_info['data_info']['count_path']
        x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None).iloc[:,0])
        x_path = np.array(['%s/%s'%(count_path, x_list[fold]) for fold in range(x_list.shape[0]) if '.csv' in x_list[fold]])
    except:
        x_path = np.array(['%s/%s'%(data_path, path_info['data_info']['x_path']) for fold in range(1)])
    
    reader.read_dataset(x_path[0], None, 0)
    
    network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
    network = network_class(network_info, path_info, log, fold=0, num_classes=num_classes, 
                            is_covariates=reader.is_covariates, covariate_shape = reader.covariate_shape,
                            verbose=False)
    network.fold = ''
    network.load_weights(weight_path, verbose=False)
    tree_weight_list = network.get_trained_weight()  
    if not tf.__version__.startswith('2'): k.clear_session()
        
    if reader.is_covariates:
        if len(tree_weight_list[-1].index) - len(reader.covariates_names) > 0:
            tree_weight_list[-1].index = list(tree_weight_list[-1].index)[:-len(reader.covariates_names)] + list(reader.covariates_names)
    return tree_weight_list


def deepbiome_taxa_selection_performance(log, network_info, path_info, num_classes,
                                         true_tree_weight_list, trained_weight_path_list):
    """
    Function for prediction by the pretrained deep neural network with phylogenetic tree weight regularizer. 
    
    See ref url (TODO: update)

    Parameters
    ----------
    log (logging instance) :
        python logging instance for logging
    network_info (dictionary) :
        python dictionary with network_information
    path_info (dictionary):
        python dictionary with path_information
    num_classes (int):
        number of classes for the network. 0 for regression, 1 for binary classificatin.
    true_tree_weight_list (ndarray):
        lists of the true weight information with the shape of (k folds, number of weights)
        `true_tree_weight_list[0][0]` is the true weight information between the first and second layers for the first fold. It is a numpy array with the shape of (number of nodes for the first layer, number of nodes for the second layer).
    trained_weight_path_list (list):
        lists of the path of trained weight for each fold.
    max_queue_size (int):
        default=10
    workers (int):
        default=1
    use_multiprocessing (boolean):
        default=False
    
    Returns
    -------
    summary (numpy array):
        summary of the taxa selection performance
    
    Examples
    --------
    The taxa selection performance of the trained deep neural network with phylogenetic tree weight regularizer.
    
    summary = deepbiome_taxa_selection_performance(log, network_info, path_info, num_classes)
    """
    
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    ### Argument #########################################################################################
    model_save_dir = path_info['model_info']['model_dir']
    model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
    
    data_path = path_info['data_info']['data_path']
        
    starttime = time.time()
    if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=True)
    data_path = path_info['data_info']['data_path']
    try:
        count_path = path_info['data_info']['count_path']
        x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None).iloc[:,0])
        x_path = np.array(['%s/%s'%(count_path, x_list[fold]) for fold in range(x_list.shape[0]) if '.csv' in x_list[fold]])
    except:
        x_path = np.array(['%s/%s'%(data_path, path_info['data_info']['x_path']) for fold in range(1)])
    
    reader.read_dataset(x_path[0], None, 0)
    network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
    network = network_class(network_info, path_info, log, fold=0, num_classes=num_classes, 
                            is_covariates=reader.is_covariates, covariate_shape = reader.covariate_shape,
                            verbose=False)
    
    prediction = []
    accuracy_list = []
    for fold in range(len(trained_weight_path_list)):
        foldstarttime = time.time()
        network.load_weights(trained_weight_path_list[fold], verbose=False)
        tree_weight_list = network.get_trained_weight()
        # true_tree_weight_list = network.load_true_tree_weight_list(path_info['data_info']['data_path'])
        accuracy_list.append(np.array(taxa_selection_accuracy(tree_weight_list, true_tree_weight_list[fold])))
    accuracy_list = np.array(accuracy_list)[:,:,1:]
    
    for fold in range(len(trained_weight_path_list)):
        tree_level = []
        selected = []
        true_node = []
        for i in range(accuracy_list.shape[1]):    
            tree_tw = true_tree_weight_list[fold][i].astype(np.int32)
            tree_level.append(network.tree_level_list[i])
            selected.append(np.sum(np.sum(tree_tw, axis=1)>0))
            true_node.append(tree_tw.shape[0])

        taxa_metrics = [ms.strip() for ms in network_info['model_info']['taxa_selection_metrics'].split(',')]
        metrics_names = list(np.array([['%s_mean' % ms.capitalize(), '%s_std' % ms.capitalize()] for ms in taxa_metrics]).flatten())
        summary = pd.DataFrame(columns=['Model','PhyloTree','No. true taxa', 'No. total taxa'] + metrics_names)
        for i, (mean, std) in enumerate(zip(np.mean(accuracy_list, axis=0), np.std(accuracy_list, axis=0))):
            args = ['', tree_level[i], selected[i], true_node[i]] + np.stack([mean, std]).T.flatten().tolist()
            summary.loc[i] = args
        summary.iloc[0,0] = model_save_dir
    if not tf.__version__.startswith('2'): k.clear_session()
    gc.collect()
    return summary


def deepbiome_draw_phylogenetic_tree(log, network_info, path_info, num_classes,
                                     file_name = "%%inline", img_w = 500, branch_vertical_margin = 20, 
                                     arc_start = 0, arc_span = 360,
                                     node_name_on = True, name_fsize = 10,
                                     tree_weight_on = True, tree_weight=None, weight_opacity = 0.4, weight_max_radios = 10, 
                                     background_color_on = True, phylumn_color = []):
    """
    Draw phylogenetic tree
    
    See ref url (TODO: update)

    Parameters
    ----------
    log (logging instance) :
        python logging instance for logging
    network_info (dictionary) :
        python dictionary with network_information
    path_info (dictionary):
        python dictionary with path_information
    num_classes (int):
        number of classes for the network. 0 for regression, 1 for binary classificatin.
    file_name (str):
        name of the figure for save.
        - "*.png", "*.jpg"
        - "%%inline" for notebook inline output.
        default="%%inline"
    img_w (int):
        image width (pt)
        default=500
    branch_vertical_margin (int):
        default=20
    arc_start (int):
        default=0
    arc_span (int):
        default=360
    node_name_on (boolean):
        default=False
    name_fsize (int):
        default=10
    tree_weight_on (boolean):
        default=True
    tree_weight (ndarray)
        default=None
    weight_opacity  (float)
        default= 0.4
    weight_max_radios (int)
        default= 10
    background_color_on (boolean)
        default= True
    phylumn_color (list)
        default= []
    
    Returns
    -------
    
    Examples
    --------
    Draw phylogenetic tree
    
    deepbiome_draw_phylogenetic_tree(log, network_info, path_info, num_classes, file_name = "%%inline")
    """
    
    os.environ['QT_QPA_PLATFORM']='offscreen' # for tree figure (https://github.com/etetoolkit/ete/issues/381)
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=True)
    data_path = path_info['data_info']['data_path']
    try:
        count_path = path_info['data_info']['count_path']
        x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None).iloc[:,0])
        x_path = np.array(['%s/%s'%(count_path, x_list[fold]) for fold in range(x_list.shape[0]) if '.csv' in x_list[fold]])
    except:
        x_path = np.array(['%s/%s'%(data_path, path_info['data_info']['x_path']) for fold in range(1)])
    
    reader.read_dataset(x_path[0], None, 0)
    
    network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
    network = network_class(network_info, path_info, log, fold=0, num_classes=num_classes, 
                            is_covariates=reader.is_covariates, covariate_shape = reader.covariate_shape,
                            verbose=False)
    
    if len(phylumn_color) == 0:
        colors = mcolors.CSS4_COLORS
        colors_name = list(colors.keys())
        phylumn_color = np.random.choice(colors_name, network.phylogenetic_tree_info['Phylum'].unique().shape[0])
    
    basic_st = NodeStyle()
    basic_st['size'] = weight_max_radios * 0.5
    basic_st['shape'] = 'circle'
    basic_st['fgcolor'] = 'black'
    
    t = Tree()
    root_st = NodeStyle()
    root_st["size"] = 0
    t.set_style(root_st)
    for i, phylum_val in enumerate(network.phylogenetic_tree_info['Phylum'].unique()):
        t.add_child(name=phylum_val)
        phylum_t = t.get_leaves_by_name(name=phylum_val)[0]
        if background_color_on:
            phylum_st = copy.deepcopy(basic_st)
            phylum_st["bgcolor"] = phylumn_color[i]
            phylum_t.set_style(phylum_st)
        else:
            phylum_t.set_style(basic_st)
            
        child_class = network.phylogenetic_tree_info['Class'][network.phylogenetic_tree_info['Phylum'] == phylum_val].unique()
        for class_val in child_class:
            phylum_t.add_child(name=class_val)
            class_t = t.get_leaves_by_name(name=class_val)[0]
            class_t.set_style(basic_st)
            if tree_weight_on:
                class_weights = np.array(tree_weight[-1])
                class_weights *= (weight_max_radios / np.max(class_weights))
                phylum_num = network.phylogenetic_tree_dict['Phylum'][phylum_val]
                class_num = network.phylogenetic_tree_dict['Class'][class_val]
                class_t.add_features(weight=class_weights[class_num,phylum_num])

            child_order = network.phylogenetic_tree_info['Order'][network.phylogenetic_tree_info['Class'] == class_val].unique()
            for order_val in child_order:
                class_t.add_child(name=order_val)
                order_t = t.get_leaves_by_name(name=order_val)[0]
                order_t.set_style(basic_st)
                if tree_weight_on:
                    order_weights = np.array(tree_weight[-2]) 
                    order_weights *= (weight_max_radios / np.max(order_weights))
                    class_num = network.phylogenetic_tree_dict['Class'][class_val]
                    order_num = network.phylogenetic_tree_dict['Order'][order_val]
                    order_t.add_features(weight=order_weights[order_num,class_num])
                    
                child_family = network.phylogenetic_tree_info['Family'][network.phylogenetic_tree_info['Order'] == order_val].unique()
                for family_val in child_family:
                    order_t.add_child(name=family_val)
                    family_t = t.get_leaves_by_name(name=family_val)[0]
                    family_t.set_style(basic_st)
                    if tree_weight_on:
                        family_weights = np.array(tree_weight[-3])
                        family_weights *= (weight_max_radios / np.max(family_weights))
                        order_num = network.phylogenetic_tree_dict['Order'][order_val]
                        family_num = network.phylogenetic_tree_dict['Family'][family_val]
                        family_t.add_features(weight=family_weights[family_num, order_num])
                        
                    child_genus = network.phylogenetic_tree_info['Genus'][network.phylogenetic_tree_info['Family'] == family_val].unique()
                    for genus_val in child_genus:
                        family_t.add_child(name=genus_val)
                        genus_t = t.get_leaves_by_name(name=genus_val)[0]
                        genus_t.set_style(basic_st)
                        if tree_weight_on:
                            genus_weights = np.array(tree_weight[-4])
                            genus_weights *= (weight_max_radios / np.max(genus_weights))
                            family_num = network.phylogenetic_tree_dict['Family'][family_val]
                            genus_num = network.phylogenetic_tree_dict['Genus'][genus_val]
                            genus_t.add_features(weight=genus_weights[genus_num, family_num])
                    
    def layout(node):
        if "weight" in node.features:
            # Creates a sphere face whose size is proportional to node's
            # feature "weight"
            color = {1:"RoyalBlue", 0: "Red"}[int(node.weight > 0)]
            C = CircleFace(radius=node.weight, color=color, style="circle")
            # Let's make the sphere transparent
            C.opacity = weight_opacity
            # And place as a float face over the tree
            faces.add_face_to_node(C, node, 0, position="float")
        
        if node_name_on & node.is_leaf():
            # Add node name to laef nodes
            N = AttrFace("name", fsize=name_fsize, fgcolor="black")
            faces.add_face_to_node(N, node, 0)
        
        
    ts = TreeStyle()

    ts.show_leaf_name = False
    ts.mode = "c"
    ts.arc_start = arc_start
    ts.arc_span = arc_span
    ts.layout_fn = layout
    ts.branch_vertical_margin = branch_vertical_margin
    ts.show_scale = False

    return t.render(file_name=file_name, w=img_w, tree_style=ts)

# #########################################################################################################################
# if __name__ == "__main__":  
#     argdict = argv_parse(sys.argv)
#     try: gpu_memory_fraction = float(argdict['gpu_memory_fraction'][0]) 
#     except: gpu_memory_fraction = None
#     try: max_queue_size=int(argdict['max_queue_size'][0])
#     except: max_queue_size=10
#     try: workers=int(argdict['workers'][0])
#     except: workers=1
#     try: use_multiprocessing=argdict['use_multiprocessing'][0]=='True'      
#     except: use_multiprocessing=False

#     ### Logger ############################################################################################
#     logger = logging_daily.logging_daily(argdict['log_info'][0])
#     logger.reset_logging()
#     log = logger.get_logging()
#     log.setLevel(logging_daily.logging.INFO)

#     log.info('Argument input')
#     for argname, arg in argdict.items():
#         log.info('    {}:{}'.format(argname,arg))

#     ### Configuration #####################################################################################
#     config_data = configuration.Configurator(argdict['path_info'][0], log)
#     config_data.set_config_map(config_data.get_section_map())
#     config_data.print_config_map()

#     config_network = configuration.Configurator(argdict['network_info'][0], log)
#     config_network.set_config_map(config_network.get_section_map())
#     config_network.print_config_map()

#     path_info = config_data.get_config_map()
#     network_info = config_network.get_config_map()
#     test_evaluation, train_evaluation, network = deepbiome_train(log, network_info, path_info, number_of_fold=20)