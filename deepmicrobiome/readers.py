######################################################################
## DeepBiome
## - Reader
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
import timeit
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import scale, minmax_scale, robust_scale
from sklearn.preprocessing import MinMaxScaler

import logging_daily
from utils import file_size, convert_bytes, print_sysinfo
from keras.utils import to_categorical
# from keras.utils import to_categorical

########################################################################################################
# Base reader
########################################################################################################
class BaseReader(object):
    """Inherit from this class when implementing new readers."""
    def __init__(self, log, verbose=True):
    # def __init__(self, log, path_info, network_info, verbose=True):
        self.log = log
        self.verbose = verbose
        
    def read_dataset(self, data_path):
        raise NotImplementedError()
        
    def get_dataset(self, train_index):
        return NotImplementedError()
    
    def get_training_validation_index(self, idx, validation_size=0.2):
        return train_test_split(idx, test_size = validation_size)

########################################################################################################
### MicroBiome Reader
########################################################################################################
class MicroBiomeReader(BaseReader):
    # def __init__(self, log, path_info, network_info, verbose=True):
    def __init__(self, log, verbose=True):
        super(MicroBiomeReader,self).__init__(log, verbose)
        # self.path_info = path_info
        # self.network_info = network_info
        
    def read_dataset(self, x_path, y_path, sim): # TODO fix without sim...
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-----------------------------------------------------------------------')
        self.log.info('Load data')
        
        x = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        
        # Normalize X with min-max normalization
        mat = np.matrix(x)
        prepro = MinMaxScaler()
        prepro.fit(mat)
        self.x_label = np.array(x.columns)
        x = pd.DataFrame(prepro.transform(mat), columns = list(x.columns))
        self.x = np.array(x, dtype=np.float32)
        
        y = pd.DataFrame(y.iloc[:, sim]) #.merge(pd.DataFrame(1-y.iloc[:, sim]), left_index = True, right_index = True)
        # self.y_label = ['y1','y2']
        if np.array(y).dtype == np.int and np.max(np.array(y)) > 1:
            self.num_classes = np.max(np.array(y))+1
            self.y = to_categorical(y, num_classes=np.max(np.array(y))+1, dtype=np.int32)
        elif np.array(y).dtype == np.int and np.max(np.array(y)) <= 1:
            self.num_classes = 1
            self.y = np.array(y, dtype=np.int32)[:,0]
        else:
            self.num_classes = 0
            self.y = np.array(y, dtype=np.float32)[:,0]
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_dataset(self, idxs):
        tot_idxs = np.arange(self.x.shape[0])
        remain_idxs = np.setdiff1d(tot_idxs, idxs)
        
        x_train = self.x[idxs]
        x_test = self.x[remain_idxs]
        
        y_train = self.y[idxs]
        y_test = self.y[remain_idxs]
        return x_train, x_test, y_train, y_test

########################################################################################################
########################################################################################################
# TODO: fix
if __name__ == "__main__":
    # Argument
    batch_size = 50
    
    # Logger
    logger = logging_daily.logging_daily('simulation_s2_deepbiome/config/log_info.yaml')
    logger.reset_logging()
    log = logger.get_logging()
    
    #############################################################################################################
    # Test Reader
    #############################################################################################################
    reader = BaseReader(log)
    sim = 0
    
    log.info('test')