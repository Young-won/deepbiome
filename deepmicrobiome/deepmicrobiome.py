# -*- coding: utf-8 -*-

"""Main module."""
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

import logging_daily
import configuration
import loss_and_metric
import readers
import build_network
from utils import file_path_fold, argv_parse

import keras.backend as k
argdict = argv_parse(sys.argv)

config = k.tf.ConfigProto()
if 'gpu_memory_fraction' in argdict: config.gpu_options.per_process_gpu_memory_fraction = float(argdict['gpu_memory_fraction'][0])
else: config.gpu_options.allow_growth=True

max_queue_size=int(argdict['max_queue_size'][0])
workers=int(argdict['workers'][0])
use_multiprocessing=argdict['use_multiprocessing'][0]=='True'      

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

### Training hyperparameter ##########################################################################
model_save_dir = path_info['model_info']['model_dir']
# TODO : Warm start
# warm_start= network_info['training_info']['warm_start'] == 'True'
# warm_start_model = network_info['training_info']['warm_start_model']
# try: save_frequency=int(network_info['training_info']['save_frequency'])
# except: save_frequency=None

model_path = os.path.join(model_save_dir, path_info['model_info']['weight'])
hist_path = os.path.join(model_save_dir, path_info['model_info']['history'])

### Reader ###########################################################################################
log.info('-----------------------------------------------------------------')
reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
reader = reader_class(log, verbose=True)

data_path = path_info['data_info']['data_path']
count_path = path_info['data_info']['count_path']
x_list = np.array(pd.read_csv(path_info['data_info']['count_list_path'], header=None)[0])
y_path = '%s/%s'%(data_path, path_info['data_info']['y_path'])
idxs = np.array(pd.read_csv(path_info['data_info']['idx_path'])-1, dtype=np.int)

### Simulations #################################################################################
history = []
evaluation = []
# train_tot_idxs = []
# test_tot_idxs = []
starttime = time.time()
for fold in range(20):
    log.info('-------%d simulation start!----------------------------------' % (fold+1))
    foldstarttime = time.time()
    
    ### Read datasets ####################################################################################
    reader.read_dataset('%s/%s'%(count_path, x_list[fold]), y_path, fold)
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
    history.append(hist.history)
    sys.stdout.flush()
        
    network.save_weights(file_path_fold(model_path, fold))
    log.debug('Save weight at {}'.format(file_path_fold(model_path, fold)))
    network.save_history(file_path_fold(hist_path, fold), history)
    log.debug('Save history at {}'.format(file_path_fold(hist_path, fold)))
    sys.stdout.flush()
    
    # Evaluation
    eval_res = network.evaluate(x_test, y_test)
    evaluation.append(eval_res)
    
    k.clear_session()
    log.info('Compute time : {}'.format(time.time()-foldstarttime))
    log.info('%d fold computing end!---------------------------------------------' % (fold+1))

### Summary #########################################################################################
evaluation = np.vstack(evaluation)
log.info('Evaluation : %s' % np.array(['loss']+[metric.strip() for metric in network_info['model_info']['metrics'].split(',')]))
mean = np.mean(evaluation, axis=0)
std = np.std(evaluation, axis=0)
log.info('      mean : %s',mean)
log.info('       std : %s',std)

### Save #########################################################################################
np.save(os.path.join(model_save_dir, path_info['model_info']['evaluation']),evaluation)
# np.save(os.path.join(model_save_dir, path_info['model_info']['train_tot_idxs']), np.array(train_tot_idxs))
# np.save(os.path.join(model_save_dir, path_info['model_info']['test_tot_idxs']), np.array(test_tot_idxs))

### Exit #########################################################################################
log.info('Total Computing Ended')
log.info('-----------------------------------------------------------------')
gc.collect()
sys.exit(0)