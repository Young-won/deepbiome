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

from . import logging_daily
from . import configuration
from . import loss_and_metric
from . import readers
from . import build_network
from .utils import file_path_fold, argv_parse

import keras.backend as k  
import tensorflow as tf
    
def deepbiome_train(log, network_info, path_info, number_of_fold=None, 
                    max_queue_size=10, workers=1, use_multiprocessing=False):
    """
    Training the deep neural network with phylogenetic tree weight regularizer.
    See ref url (TODO: update)

    Parameters
    ----------
    log : logging instance
        python logging instance for logging
    network_info : dictionary
        python dictionary with network_information
    path_info : dictionary
        python dictionary with path_information
    number_of_fold : int
        default=None
    max_queue_size : int
        default=10
    workers : int
        default=1
    use_multiprocessing : boolean
        default=False

    Returns
    -------
    test_evaluation : numpy array
        numpy array of the evaluation using testset from all fold
    train_evaluation : numpy array
        numpy array of the evaluation using training from all fold
    network : deepbiome network instance
        deepbiome class instance
    
    Examples
    --------
    Training the deep neural network with phylogenetic tree weight regularizer.

    >>> deepbiome_train(log, network_info, path_info)
    """
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.get_visible_devices(device_type='GPU')
        try: tf.config.experimental.set_memory_growth(gpus, True)
        except: pass
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
    ### Argument #########################################################################################
    model_save_dir = path_info['model_info']['model_dir']
    model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
    # hist_path = os.path.join(model_save_dir, path_info['model_info']['history'

    # TODO : Warm start
    # warm_start= network_info['training_info']['warm_start'] == 'True'
    # warm_start_model = network_info['training_info']['warm_start_model']
    # try: save_frequency=int(network_info['training_info']['save_frequency'])
    # except: save_frequency=None

    ### Reader ###########################################################################################
    log.info('-----------------------------------------------------------------')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, verbose=True)

    data_path = path_info['data_info']['data_path']
    # TODO: fix
    idxs = np.array(pd.read_csv(path_info['data_info']['idx_path'])-1, dtype=np.int)
    try:
        count_path = path_info['data_info']['count_path']
        x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None).iloc[:,0])
        x_path = np.array(['%s/%s'%(count_path, x_list[fold]) for fold in range(idxs.shape[1])])
    except:
        x_path = np.array(['%s/%s'%(data_path, path_info['data_info']['x_path']) for fold in range(idxs.shape[1])])
    y_path = '%s/%s'%(data_path, path_info['data_info']['y_path'])

    ### Simulations #################################################################################
    # history = []
    train_evaluation = []
    test_evaluation = []
    # train_tot_idxs = []
    # test_tot_idxs = []
    if number_of_fold == None:
        number_of_fold = idxs.shape[1]
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
        network = network_class(network_info, path_info['data_info'], log, fold, num_classes=num_classes)
        network.model_compile() ## TODO : weight clear only (no recompile)
        sys.stdout.flush()

        ### Training #########################################################################################
        log.info('-----------------------------------------------------------------')
        log.info('%d fold computing start!----------------------------------' % (fold+1))
        hist = network.fit(x_train, y_train, 
                           max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
                           model_path=file_path_fold(model_path, fold))
        # history.append(hist.history)
        sys.stdout.flush()

        network.save_weights(file_path_fold(model_path, fold))
        log.debug('Save weight at {}'.format(file_path_fold(model_path, fold)))
        # network.save_history(file_path_fold(hist_path, fold), history)
        # log.debug('Save history at {}'.format(file_path_fold(hist_path, fold)))
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


#########################################################################################################################
if __name__ == "__main__":  
    argdict = argv_parse(sys.argv)
    try: gpu_memory_fraction = float(argdict['gpu_memory_fraction'][0]) 
    except: gpu_memory_fraction = None
    try: max_queue_size=int(argdict['max_queue_size'][0])
    except: max_queue_size=10
    try: workers=int(argdict['workers'][0])
    except: workers=1
    try: use_multiprocessing=argdict['use_multiprocessing'][0]=='True'      
    except: use_multiprocessing=False

    ### Logger ############################################################################################
    logger = logging_daily.logging_daily(argdict['log_info'][0])
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)

    log.info('Argument input')
    for argname, arg in argdict.items():
        log.info('    {}:{}'.format(argname,arg))

    ### Configuration #####################################################################################
    config_data = configuration.Configurator(argdict['path_info'][0], log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(argdict['network_info'][0], log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    test_evaluation, train_evaluation, network = deepbiome_train(log, network_info, path_info, number_of_fold=2)