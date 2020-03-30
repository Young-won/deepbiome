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
from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle, CircleFace, TextFace, RectFace
import matplotlib.colors as mcolors

pd.set_option('display.float_format', lambda x: '%.03f' % x)
np.set_printoptions(formatter={'float_kind':lambda x: '%.03f' % x})

def deepbiome_train(log, network_info, path_info, number_of_fold=None, 
                    tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                    max_queue_size=10, workers=1, use_multiprocessing=False, 
                    verbose=True):
    """
    Function for training the deep neural network with phylogenetic tree weight regularizer.
    
    It uses microbiome abundance data as input and uses the phylogenetic taxonomy to guide the decision of the optimal number of layers and neurons in the deep learning architecture.

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
    tree_level_list (list):
        name of each level of the given reference tree weights
        default=['Genus', 'Family', 'Order', 'Class', 'Phylum']
    max_queue_size (int):
        default=10
    workers (int):
        default=1
    use_multiprocessing (boolean):
        default=False
    verbose (boolean):
        show the log if True
        default=True

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
    if verbose: log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    # TODO: fix path_info
    reader = reader_class(log, path_info, verbose=verbose)

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
        if verbose: log.info('-------%d simulation start!----------------------------------' % (fold+1))
        foldstarttime = time.time()

        ### Read datasets ####################################################################################
        reader.read_dataset(x_path[fold], y_path, fold)
        x_train, x_test, y_train, y_test = reader.get_dataset(idxs[:,fold])
        num_classes = reader.get_num_classes()

        ### Bulid network ####################################################################################
        if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
        if verbose:
            log.info('-----------------------------------------------------------------')
            log.info('Build network for %d simulation' % (fold+1))
        network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
        network = network_class(network_info, path_info, log, fold, num_classes=num_classes,
                                tree_level_list = tree_level_list,
                                is_covariates=reader.is_covariates, covariate_names = reader.covariate_names, verbose=verbose)
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
        if verbose: log.debug('Save weight at {}'.format(file_path_fold(model_path, fold)))
        if is_save_hist: 
            network.save_history(file_path_fold(hist_path, fold), hist.history)
            if verbose: log.debug('Save history at {}'.format(file_path_fold(hist_path, fold)))
        sys.stdout.flush()

        # Evaluation
        train_eval_res = network.evaluate(x_train, y_train)
        train_evaluation.append(train_eval_res)
        
        test_eval_res = network.evaluate(x_test, y_test)
        test_evaluation.append(test_eval_res)

        if not tf.__version__.startswith('2'): k.clear_session()
        if verbose: log.info('Compute time : {}'.format(time.time()-foldstarttime))
        if verbose: log.info('%d fold computing end!---------------------------------------------' % (fold+1))

    ### Summary #########################################################################################
    train_evaluation = np.vstack(train_evaluation)
    mean = np.mean(train_evaluation, axis=0)
    std = np.std(train_evaluation, axis=0)
    
    test_evaluation = np.vstack(test_evaluation)
    test_mean = np.mean(test_evaluation, axis=0)
    test_std = np.std(test_evaluation, axis=0)
    if verbose:
        log.info('-----------------------------------------------------------------')
        log.info('Train Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
        log.info('      mean : %s',mean)
        log.info('       std : %s',std)
        log.info('-----------------------------------------------------------------')
        log.info('Test Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
        log.info('      mean : %s',test_mean)
        log.info('       std : %s',test_std)
        log.info('-----------------------------------------------------------------')

    ### Save #########################################################################################
    np.save(os.path.join(model_save_dir, 'train_%s'%path_info['model_info']['evaluation']),train_evaluation)
    np.save(os.path.join(model_save_dir, 'test_%s'%path_info['model_info']['evaluation']),test_evaluation)
    # np.save(os.path.join(model_save_dir, path_info['model_info']['train_tot_idxs']), np.array(train_tot_idxs))
    # np.save(os.path.join(model_save_dir, path_info['model_info']['test_tot_idxs']), np.array(test_tot_idxs))

    ### Exit #########################################################################################
    gc.collect()
    if verbose:
        log.info('Total Computing Ended')
        log.info('-----------------------------------------------------------------')
    return test_evaluation, train_evaluation, network


def deepbiome_test(log, network_info, path_info, number_of_fold=None,
                   tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                   max_queue_size=10, workers=1, use_multiprocessing=False,
                   verbose=True):
    """
    Function for testing the pretrained deep neural network with phylogenetic tree weight regularizer. 
    
    If you use the index file, this function provide the evaluation using test index (index set not included in the index file) for each fold. If not, this function provide the evaluation using the whole samples.

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
    tree_level_list (list):
        name of each level of the given reference tree weights
        default=['Genus', 'Family', 'Order', 'Class', 'Phylum']
    max_queue_size (int):
        default=10
    workers (int):
        default=1
    use_multiprocessing (boolean):
        default=False
    verbose (boolean):
        show the log if True
        default=True
        
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
    if verbose: log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=verbose)

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
    if verbose: log.info('Test Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
    for fold in range(number_of_fold):
        if verbose: log.info('-------%d fold test start!----------------------------------' % (fold+1))
        foldstarttime = time.time()

        ### Read datasets ####################################################################################
        reader.read_dataset(x_path[fold], y_path, fold)
        x_train, x_test, y_train, y_test = reader.get_dataset(idxs[:,fold])
        num_classes = reader.get_num_classes()

        ### Bulid network ####################################################################################
        if not tf.__version__.startswith('2'): k.set_session(tf.Session(config=config))
        if verbose:
            log.info('-----------------------------------------------------------------')
            log.info('Build network for %d fold testing' % (fold+1))
        network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
        network = network_class(network_info, path_info, log, fold, num_classes=num_classes, 
                                tree_level_list = tree_level_list,
                                is_covariates=reader.is_covariates, covariate_names = reader.covariate_names, verbose=verbose)
        network.model_compile() ## TODO : weight clear only (no recompile)
        network.load_weights(file_path_fold(model_path, fold), verbose=verbose)
        sys.stdout.flush()

        ### Training #########################################################################################
        if verbose:
            log.info('-----------------------------------------------------------------')
            log.info('%d fold computing start!----------------------------------' % (fold+1))
        test_eval_res = network.evaluate(x_test, y_test)
        test_evaluation.append(test_eval_res)
        if not tf.__version__.startswith('2'): k.clear_session()
        if verbose: 
            log.info('' % test_eval_res)
            log.info('Compute time : {}'.format(time.time()-foldstarttime))
            log.info('%d fold computing end!---------------------------------------------' % (fold+1))
    
    ### Summary #########################################################################################
    test_evaluation = np.vstack(test_evaluation)
    mean = np.mean(test_evaluation, axis=0)
    std = np.std(test_evaluation, axis=0)
    gc.collect()
    
    if verbose:
        log.info('-----------------------------------------------------------------')
        log.info('Test Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
        log.info('      mean : %s',mean)
        log.info('       std : %s',std)
        log.info('-----------------------------------------------------------------')
        log.info('Total Computing Ended')
        log.info('-----------------------------------------------------------------')
    
    return test_evaluation


def deepbiome_prediction(log, network_info, path_info, num_classes, number_of_fold=None,
                         change_weight_for_each_fold=False, get_y = False,
                         tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                         max_queue_size=10, workers=1, use_multiprocessing=False, 
                         verbose=True):
    """
    Function for prediction by the pretrained deep neural network with phylogenetic tree weight regularizer. 
    
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
    tree_level_list (list):
        name of each level of the given reference tree weights
        default=['Genus', 'Family', 'Order', 'Class', 'Phylum']
    max_queue_size (int):
        default=10
    workers (int):
        default=1
    use_multiprocessing (boolean):
        default=False
    verbose (boolean):
        show the log if True
        default=True
    
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
    if verbose: log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=verbose)

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
        if verbose: log.info('-------%d th repeatition prediction start!----------------------------------' % (fold+1))
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
        if verbose: log.info('-----------------------------------------------------------------')
        network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
        network = network_class(network_info, path_info, log, fold=fold, num_classes=num_classes,
                                tree_level_list = tree_level_list,
                                is_covariates=reader.is_covariates, covariate_names = reader.covariate_names, verbose=verbose)
        network.model_compile()
        if change_weight_for_each_fold:network.load_weights(file_path_fold(model_path, fold), verbose=verbose)
        else: network.load_weights(model_path, verbose=verbose)
        sys.stdout.flush()

        ### Training #########################################################################################
        if verbose: log.info('-----------------------------------------------------------------')
        pred = network.predict(x_test)
        if get_y: prediction.append(np.array(list(zip(pred, y_test))))
        else: prediction.append(pred)
        if verbose: log.info('Compute time : {}'.format(time.time()-foldstarttime))
        if verbose: log.info('%d fold computing end!---------------------------------------------' % (fold+1))

    ### Exit #########################################################################################
    if not tf.__version__.startswith('2'): k.clear_session()
    prediction = np.array(prediction)
    if verbose: log.info('Total Computing Ended')
    if verbose: log.info('-----------------------------------------------------------------')
    gc.collect()
    return prediction

def deepbiome_get_trained_weight(log, network_info, path_info, num_classes, weight_path,
                                 tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                                 verbose=True):
    """
    Function for prediction by the pretrained deep neural network with phylogenetic tree weight regularizer. 
    
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
    tree_level_list (list):
        name of each level of the given reference tree weights
        default=['Genus', 'Family', 'Order', 'Class', 'Phylum']
    verbose (boolean):
        show the log if True
        default=True
    
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
    reader = reader_class(log, path_info, verbose=verbose)
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
                            tree_level_list = tree_level_list,
                            is_covariates=reader.is_covariates, covariate_names = reader.covariate_names,
                            verbose=verbose)
    network.fold = ''
    network.load_weights(weight_path, verbose=False)
    tree_weight_list = network.get_trained_weight()  
    if not tf.__version__.startswith('2'): k.clear_session()
        
    if reader.is_covariates:
        if len(tree_weight_list[-1].index) - len(reader.covariate_names) > 0:
            tree_weight_list[-1].index = list(tree_weight_list[-1].index)[:-len(reader.covariate_names)] + list(reader.covariate_names)
    return tree_weight_list


def deepbiome_taxa_selection_performance(log, network_info, path_info, num_classes,
                                         true_tree_weight_list, trained_weight_path_list, 
                                         tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                                         lvl_category_dict = None,
                                         verbose=True):
    """
    Function for prediction by the pretrained deep neural network with phylogenetic tree weight regularizer. 

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
    tree_level_list (list):
        name of each level of the given reference tree weights
        default=['Genus', 'Family', 'Order', 'Class', 'Phylum']
    verbose (boolean):
        show the log if True
        default=True
    
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
    reader = reader_class(log, path_info, verbose=verbose)
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
                            tree_level_list = tree_level_list,
                            is_covariates=reader.is_covariates, covariate_names = reader.covariate_names, 
                            # lvl_category_dict = lvl_category_dict,
                            verbose=False)
    
    prediction = []
    accuracy_list = []
    for fold in range(len(trained_weight_path_list)):
        foldstarttime = time.time()  
        network.load_weights(trained_weight_path_list[fold], verbose=verbose)
        tree_weight_list = network.get_trained_weight()
        # true_tree_weight_list = network.load_true_tree_weight_list(path_info['data_info']['data_path'])
        try: 
            accuracy_list.append(np.array(taxa_selection_accuracy(tree_weight_list, true_tree_weight_list[fold])))
        except:
            for tree_level in range(len(tree_weight_list)):
                tw = tree_weight_list[tree_level]
                row_setdiff = np.setdiff1d(lvl_category_dict[tree_level], tw.index)
                if len(row_setdiff) > 0:
                    for new_row in row_setdiff:
                        tw = tw.append(pd.Series(0, index=tw.columns, name=new_row))
                    tree_weight_list[tree_level] = tw.loc[lvl_category_dict[tree_level],:]

                if tree_level+1 < len(tree_weight_list):
                    tw = tree_weight_list[tree_level]
                    col_setdiff = np.setdiff1d(lvl_category_dict[tree_level+1], tw.columns)
                    if len(col_setdiff) > 0:
                        for new_col in col_setdiff:
                            tw[new_col] = 0
                        tree_weight_list[tree_level] = tw.loc[:, lvl_category_dict[tree_level+1]]
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
                                     tree_weight_on = True, tree_weight=None,
                                     tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                                     weight_opacity = 0.4, weight_max_radios = 10, 
                                     phylum_background_color_on = True, phylum_color = [], phylum_color_legend=False,
                                     show_covariates = True,
                                     verbose=True):
    """
    Draw phylogenetic tree

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
        vertical margin for branch
        default=20
    arc_start (int):
        angle that arc start
        default=0
    arc_span (int):
        total amount of angle for the arc span
        default=360
    node_name_on (boolean):
        show the name of the last leaf node if True
        default=False
    name_fsize (int):
        font size for the name of the last leaf node
        default=10
    tree_weight_on (boolean):
        show the amount and the direction of the weight for each edge in the tree by circle size and color.
        default=True
    tree_weight (ndarray):
        reference tree weights
        default=None
    tree_level_list (list):
        name of each level of the given reference tree weights
        default=['Genus', 'Family', 'Order', 'Class', 'Phylum']
    weight_opacity  (float):
        opacity for weight circle
        default= 0.4
    weight_max_radios (int):
        maximum radios for weight circle
        default= 10
    phylum_background_color_on (boolean):
        show the background color for each phylum based on `phylumn_color`.
        default= True
    phylum_color (list):
        specify the list of background colors for phylum level. If `phylumn_color` is empty, it will arbitrarily assign the color for each phylum.
        default= []
    phylum_color_legend (boolean):
        show the legend for the background colors for phylum level
        default= False
    show_covariates (boolean):
        show the effect of the covariates
        default= True
    verbose (boolean):
        show the log if True
        default=True
    Returns
    -------
    
    Examples
    --------
    Draw phylogenetic tree
    
    deepbiome_draw_phylogenetic_tree(log, network_info, path_info, num_classes, file_name = "%%inline")
    """
    
    os.environ['QT_QPA_PLATFORM']='offscreen' # for tree figure (https://github.com/etetoolkit/ete/issues/381)
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, verbose=verbose)
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
                            tree_level_list = tree_level_list,
                            is_covariates=reader.is_covariates, covariate_names = reader.covariate_names,
                            verbose=False)
    
    if len(phylum_color) == 0:
        colors = mcolors.CSS4_COLORS
        colors_name = list(colors.keys())
        if reader.is_covariates and show_covariates:
            phylum_color = np.random.choice(colors_name, network.phylogenetic_tree_info['Phylum_with_covariates'].unique().shape[0])
        else:
            phylum_color = np.random.choice(colors_name, network.phylogenetic_tree_info['Phylum'].unique().shape[0])
            
    basic_st = NodeStyle()
    basic_st['size'] = weight_max_radios * 0.5
    basic_st['shape'] = 'circle'
    basic_st['fgcolor'] = 'black'

    t = Tree()
    root_st = NodeStyle()
    root_st["size"] = 0
    t.set_style(root_st)

    tree_node_dict ={}
    tree_node_dict['root'] = t

    upper_class = 'root'
    lower_class = tree_level_list[-1]
    lower_layer_names = tree_weight[-1].columns.to_list()

    layer_tree_node_dict = {}
    phylum_color_dict = {}
    for j, val in enumerate(lower_layer_names):
        t.add_child(name=val)
        leaf_t = t.get_leaves_by_name(name=val)[0]
        leaf_t.set_style(basic_st)
        layer_tree_node_dict[val] = leaf_t
        if lower_class == 'Phylum' and phylum_background_color_on:
            phylum_st = copy.deepcopy(basic_st)
            phylum_st["bgcolor"] = phylum_color[j]
            phylum_color_dict[val] = phylum_color[j]
            leaf_t.set_style(phylum_st)
    tree_node_dict[lower_class] = layer_tree_node_dict
    upper_class = lower_class
    upper_layer_names = lower_layer_names

    for i in range(len(tree_level_list)-1):
        lower_class = tree_level_list[-2-i]
        if upper_class == 'Disease' and show_covariates == False: 
            lower_layer_names = network.phylogenetic_tree_info[lower_class].unique()
        else: 
            lower_layer_names =  tree_weight[-i-1].index.to_list()

        layer_tree_node_dict = {}
        for j, val in enumerate(upper_layer_names):
            parient_t = tree_node_dict[upper_class][val]
            if upper_class == 'Disease':
                child_class = lower_layer_names
            else:
                child_class = network.phylogenetic_tree_info[lower_class][network.phylogenetic_tree_info[upper_class] == val].unique()
            
            for k, child_val in enumerate(child_class):
                parient_t.add_child(name=child_val)
                leaf_t = parient_t.get_leaves_by_name(name=child_val)[0]
                if lower_class == 'Phylum' and phylum_background_color_on:
                    phylum_st = copy.deepcopy(basic_st)
                    phylum_st["bgcolor"] = phylum_color[k]
                    phylum_color_dict[child_val] = phylum_color[k]
                    leaf_t.set_style(phylum_st)
                else:
                    leaf_t.set_style(basic_st)
                if tree_weight_on:
                    edge_weights = np.array(tree_weight[-1-i])
                    edge_weights *= (weight_max_radios / np.max(edge_weights))
                    if upper_class == 'Disease':
                        upper_num = 0
                    else:
                        upper_num = network.phylogenetic_tree_dict[upper_class][val]
                    if upper_class == 'Disease' and reader.is_covariates == True and show_covariates:
                        lower_num = network.phylogenetic_tree_dict['%s_with_covariates' % lower_class][child_val]
                    else: lower_num = network.phylogenetic_tree_dict[lower_class][child_val]
                    leaf_t.add_features(weight=edge_weights[lower_num,upper_num])
                layer_tree_node_dict[child_val] = leaf_t
        tree_node_dict[lower_class] = layer_tree_node_dict
        upper_class = lower_class
        upper_layer_names = lower_layer_names

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

    if phylum_color_legend:
        for phylum_name in np.sort(list(phylum_color_dict.keys())):
            color_name = phylum_color_dict[phylum_name]
            ts.legend.add_face(CircleFace(weight_max_radios * 1, color_name), column=0)
            ts.legend.add_face(TextFace(" %s" % phylum_name, fsize=name_fsize), column=1)

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