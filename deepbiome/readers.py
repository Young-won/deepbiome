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
from keras.utils import to_categorical
# from keras.utils import to_categorical

from . import logging_daily

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
    def __init__(self, log, path_info, verbose=True):
        super(MicroBiomeReader,self).__init__(log, verbose)
        self.path_info = path_info
        # self.network_info = network_info
        
    def read_dataset(self, x_path, y_path, sim): # TODO fix without sim...
        if self.verbose:
            self.log.info('-----------------------------------------------------------------------')
            self.log.info('Construct Dataset')
            self.log.info('-----------------------------------------------------------------------')
            self.log.info('Load data')

        x = pd.read_csv(x_path)
        if y_path != None: y = pd.read_csv(y_path)
        
        # Normalize X with min-max normalization
        mat = np.matrix(x)
        prepro = MinMaxScaler()
        prepro.fit(mat)
        self.x_label = np.array(x.columns)
        x = pd.DataFrame(prepro.transform(mat), columns = list(x.columns))
        self.x = np.array(x, dtype=np.float32)
        
        if y_path != None: 
            try:
                y = pd.DataFrame(y.iloc[:, sim]) #.merge(pd.DataFrame(1-y.iloc[:, sim]), left_index = True, right_index = True)
            except:
                y = pd.DataFrame(y) #.merge(pd.DataFrame(1-y.iloc[:, sim]), left_index = True, right_index = True)
            self.num_classes, self.y = self._set_problem(y)
    
        # Covarates
        try: self.continuous_variabl_paths = [cov.strip() for cov in self.path_info['covariates_info']['continuous_variables'].split(',') if '.csv' in cov]
        except: self.continuous_variabl_paths = []
        try: self.categorical_variable_paths = [cov.strip() for cov in self.path_info['covariates_info']['categorical_variables'].split(',') if '.csv' in cov]
        except: self.categorical_variable_paths = []
        
        if len(self.continuous_variabl_paths) > 0:
            self.cov_conti = pd.DataFrame()
            for i in range(len(self.continuous_variabl_paths)):
                cov = pd.read_csv(self.continuous_variabl_paths[i])
                cov.columns = [name.title() for name in cov.columns]
                self.cov_conti = self.cov_conti.join(cov, how='right')
            self.cov_conti = self.cov_conti.astype(np.float32)
            
        if len(self.categorical_variable_paths) > 0:
            self.cov_categorical = pd.DataFrame()
            for i in range(len(self.categorical_variable_paths)):
                cov = pd.read_csv(self.categorical_variable_paths[i])
                cov_name = cov.columns[0]
                cov = to_categorical(cov, dtype='int32')
                cov = pd.DataFrame(cov)
                cov = cov.iloc[:,1:] #except first class as reference
                cov.columns = ['%s_%s' % (cov_name.title(), name) for name in cov.columns]
                self.cov_categorical = self.cov_categorical.join(cov, how='right')
            self.cov_categorical = self.cov_categorical.astype(np.float32)
        
        if len(self.continuous_variabl_paths) + len(self.categorical_variable_paths) > 0:
            self.covariates = pd.DataFrame()
            try: self.covariates = self.covariates.join(self.cov_conti, how='right')
            except: pass
            try: self.covariates = self.covariates.join(self.cov_categorical, how='right')
            except: pass
            self.covariate_names = self.covariates.columns
            self.covariates = np.array(self.covariates, dtype=np.float32)
            self.is_covariates = True
            self.covariate_shape = self.covariates.shape[1:]
        else:
            self.covariate_names = None
            self.is_covariates = False
            self.covariate_shape = None
   
    def _set_problem(self, y):
        raise NotImplementedError()
        
    def get_num_classes(self):
        return self.num_classes
    
    def get_dataset(self, idxs = None):
        if not np.all(idxs == None):
            tot_idxs = np.arange(self.x.shape[0])
            remain_idxs = np.setdiff1d(tot_idxs, idxs)

            x_train = self.x[idxs]
            x_test = self.x[remain_idxs]
            if self.is_covariates:
                cov_train = self.covariates[idxs]
                cov_test = self.covariates[remain_idxs]

            y_train = self.y[idxs]
            y_test = self.y[remain_idxs]
            
            if self.is_covariates:
                return [x_train,cov_train], [x_test,cov_test], y_train, y_test
            else:
                return x_train, x_test, y_train, y_test
        else:
            if self.is_covariates:
                return [self.x, self.covariates], None, self.y, None
            else:
                return self.x, None, self.y, None
        
    
    def get_input(self, idxs = None):
        if not np.all(idxs == None):
            tot_idxs = np.arange(self.x.shape[0])
            remain_idxs = np.setdiff1d(tot_idxs, idxs)

            x_train = self.x[idxs]
            x_test = self.x[remain_idxs]
            if self.is_covariates:
                cov_train = self.covariates[idxs]
                cov_test = self.covariates[remain_idxs]
            if self.is_covariates:
                return [x_train,cov_train], [x_test,cov_test]
            else:
                return x_train, x_test
        else:
            if self.is_covariates:
                return [self.x, self.covariates], None
            else:
                return self.x, None
            
########################################################################################################
# Regression
class MicroBiomeRegressionReader(MicroBiomeReader):
    def __init__(self, log, path_info, verbose=True):
        super(MicroBiomeRegressionReader, self).__init__(log, path_info, verbose)
    def _set_problem(self, y):
        num_classes = 0
        y = np.array(y, dtype=np.float32)[:,0]
        return num_classes, y
    
########################################################################################################
# Classification
class MicroBiomeClassificationReader(MicroBiomeReader):
    def __init__(self, log, path_info, verbose=True):
        super(MicroBiomeClassificationReader, self).__init__(log, path_info, verbose)
    def _set_problem(self, y):
        if np.min(np.array(y)) > 0: y = y - 1
        if np.max(np.array(y)) <= 1:
            # binary
            num_classes = 1
            try: y = np.array(y, dtype=np.int32)[:,:1]
            except: y = np.array(y, dtype=np.int32)
        else:
            # multiclass
            num_classes = np.max(np.array(y))+1
            y = to_categorical(y, num_classes=np.max(np.array(y))+1, dtype=np.int32)
        return num_classes, y
        
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