######################################################################
## DeepBiome
## - Main code for simulation
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

import logging_daily
import configuration
import loss_and_metric
import readers
import build_network
from utils import file_path_fold, argv_parse

import keras.backend as k
config = k.tf.ConfigProto()

def deepbiome_train(log, network_info, path_info, number_of_fold=None, 
                    max_queue_size=10, workers=1, use_multiprocessing=False):
    model_save_dir = path_info['model_info']['model_dir']
    # TODO : Warm start
    # warm_start= network_info['training_info']['warm_start'] == 'True'
    # warm_start_model = network_info['training_info']['warm_start_model']
    # try: save_frequency=int(network_info['training_info']['save_frequency'])
    # except: save_frequency=None

    model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
    # hist_path = os.path.join(model_save_dir, path_info['model_info']['history'
    
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
        k.set_session(k.tf.Session(config=config))
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

        k.clear_session()
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