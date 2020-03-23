######################################################################
## DeepBiome
## - Build network
##
## July 10. 2019
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################

import time
import json
import sys
import abc
import copy
import numpy as np
import pandas as pd

import keras
import keras.callbacks

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Flatten, Lambda, Reshape, LeakyReLU
from keras.layers import Concatenate
from keras.layers import BatchNormalization, Dropout
from keras.initializers import VarianceScaling

from . import loss_and_metric
# from .utils import TensorBoardWrapper

pd.set_option('display.float_format', lambda x: '%.03f' % x)
np.set_printoptions(formatter={'float_kind':lambda x: '%.03f' % x})
     
#####################################################################################################################
# Base Network
#####################################################################################################################
class Base_Network(abc.ABC):
    """Inherit from this class when implementing new networks."""
    def __init__(self, network_info, log):
        # Build Network
        self.network_info = network_info
        self.log = log
        self.best_model_save = False
        # self.TB = TensorBoardWrapper
        self.model = None
    
    @abc.abstractmethod
    def build_model(self, verbose=True):
        # define self.model
        pass
        
    def model_compile(self):
        self.model.compile(loss=getattr(loss_and_metric, self.network_info['model_info']['loss']),
                           optimizer=getattr(keras.optimizers, 
                                             self.network_info['model_info']['optimizer'])(lr=float(self.network_info['model_info']['lr']),
                                                                                           decay=float(self.network_info['model_info']['decay'])),
                           metrics=[getattr(loss_and_metric, metric.strip()) for metric in self.network_info['model_info']['metrics'].split(',')])
        self.log.info('Build Network')
        self.log.info('Optimizer = {}'.format(self.network_info['model_info']['optimizer']))
        self.log.info('Loss = {}'.format(self.network_info['model_info']['loss']))
        self.log.info('Metrics = {}'.format(self.network_info['model_info']['metrics']))
        
    # TODO: Load and save model and weight togather
    def load_model(self, model_yaml_path, verbose=0):
        # load model from YAML
        yaml_file = open(model_yaml_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = model_from_yaml(loaded_model_yaml)
        if verbose:
            self.log.info(self.model.summary())
    
    def save_model(self, model_yaml_path):
        model_yaml = self.model.to_yaml()
        with open(model_yaml_path, "w") as yaml_file:
            yaml_file.write(model_yaml)
    
    def save_weights(self, model_path, verbose=True):
        self.model.save(model_path)
        if verbose: self.log.info('Saved trained model weight at {} '.format(model_path))
        
    def load_weights(self, model_path, verbose=True):
        self.model.load_weights(model_path)
        if verbose: self.log.info('Load trained model weight at {} '.format(model_path))

#     def save_evaluation(self, eval_path, evaluation):
#         np.save(eval_path, evaluation)
        
#     def save_prediction(self, pred_path, prediction):
#         np.save(pred_path, prediction)
            
    def save_history(self, hist_path, history):
        try:
            with open(hist_path, 'w+') as f:
                json.dump(history, f)
        except:
            with open(hist_path, 'w+') as f:
                hist = dict([(ky, np.array(val).astype(np.float).tolist()) for (ky, val) in history.items()])
                json.dump(hist, f)     
            
    def get_callbacks(self, validation_data=None, model_path=None):
        # Callback
        if 'callbacks' in self.network_info['training_info']:
            callback_names = [cb.strip() for cb in self.network_info['training_info']['callbacks'].split(',')]
            callbacks = []
            for idx, callback in enumerate(callback_names):
                if 'EarlyStopping' in callback:
                    callbacks.append(getattr(keras.callbacks, callback)(monitor=self.network_info['training_info']['monitor'],
                                                                        mode=self.network_info['training_info']['mode'],
                                                                        patience=int(self.network_info['training_info']['patience']),
                                                                        min_delta=float(self.network_info['training_info']['min_delta']),
                                                                        verbose=1))
                elif 'ModelCheckpoint' in callback:
                    self.best_model_save = True
                    callbacks.append(getattr(keras.callbacks, callback)(filepath=model_path,
                                                                        monitor=self.network_info['training_info']['monitor'],
                                                                        mode=self.network_info['training_info']['mode'],
                                                                        save_best_only=True, save_weights_only=False,
                                                                        verbose=0))
                else:
                    try: callbacks.append(getattr(keras.callbacks, callback)())
                    except: pass
                        
        else:
            callbacks = []
        try: batch_size = int(self.network_info['validation_info']['batch_size'])
        except: batch_size = None
        return callbacks
    
    def fit(self, x, y, max_queue_size=50, workers=1, use_multiprocessing=False, model_path = None):
        callbacks = self.get_callbacks(None, model_path)
        try: batch_size = int(self.network_info['training_info']['batch_size'])
        except: batch_size = len(y)
        self.log.info('Training start!')
        trainingtime = time.time()
        hist = self.model.fit(x, y, batch_size=batch_size,
                              epochs=int(self.network_info['training_info']['epochs']), 
                              verbose=1, 
                              callbacks=callbacks, 
                              validation_split=float(self.network_info['validation_info']['validation_size'].strip()))
        
        if self.best_model_save:
            self.load_weights(model_path)
        self.log.info('Training end with time {}!'.format(time.time()-trainingtime))
        return hist
    
#     def fit_generator(self, train_sampler, validation_sampler=None, 
#                       max_queue_size=50, workers=1, use_multiprocessing=False,
#                       model_path = None):
#         callbacks = self.get_callbacks(validation_sampler, model_path)
        
#         self.log.info('Training start!')
#         trainingtime = time.time()
        
#         hist = self.model.fit_generator(train_sampler,
#                                         steps_per_epoch=len(train_sampler),
#                                         epochs=int(self.network_info['training_info']['epochs']), 
#                                         verbose=1, 
#                                         callbacks=callbacks, 
#                                         validation_data=validation_sampler,
#                                         validation_steps=len(validation_sampler),
#                                         max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        
#         if self.best_model_save:
#             self.load_weights(model_path)
#         self.log.info('Training end with time {}!'.format(time.time()-trainingtime))
#         return hist
    
    def evaluate(self, x, y):
        self.log.info('Evaluation start!')
        trainingtime = time.time()
        try: batch_size = int(self.network_info['test_info']['batch_size'])
        except: batch_size = len(y)
        evaluation = self.model.evaluate(x, y, batch_size = batch_size, verbose=1)
        self.log.info('Evaluation end with time {}!'.format(time.time()-trainingtime))
        self.log.info('Evaluation: {}'.format(evaluation))
        return evaluation
    
    # def evaluate_generator(self, test_sampler, 
    #                        max_queue_size=50, workers=1, use_multiprocessing=False):
    #     self.log.info('Evaluation start!')
    #     trainingtime = time.time()
    #     evaluation = self.model.evaluate_generator(test_sampler, steps=len(test_sampler), 
    #                                                max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
    #     self.log.info('Evaluation end with time {}!'.format(time.time()-trainingtime))
    #     self.log.info('Evaluation: {}'.format(evaluation))
    #     return evaluation
    
    def predict(self, x):
        self.log.info('Prediction start!')
        trainingtime = time.time()
        try: batch_size = int(self.network_info['test_info']['batch_size'])
        except: batch_size = len(x)
        prediction = self.model.predict(x, batch_size = batch_size, verbose=1)
        self.log.info('Prediction end with time {}!'.format(time.time()-trainingtime))
        return prediction
    
#     def predict_generator(self, test_sampler,
#                           max_queue_size=50, workers=1, use_multiprocessing=False):
#         self.log.info('Prediction start!')
#         trainingtime = time.time()
        
#         prediction = self.model.predict_generator(test_sampler, steps=len(test_sampler), 
#                                             max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
#                                             verbose=1)
#         self.log.info('Prediction end with time {}!'.format(time.time()-trainingtime))
#         return prediction
    

#####################################################################################################################
# Deep MicroBiome
#####################################################################################################################
#     Dense with phylogenetic tree information class
#####################################################################################################################
class Dense_with_tree(Dense):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tree_weight=None,
                 **kwargs):
        super(Dense_with_tree, self).__init__(units, 
                                              activation=activation,
                                              use_bias=use_bias,
                                              kernel_initializer=kernel_initializer,
                                              bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              bias_constraint=bias_constraint,
                                              **kwargs)
        self.tree_weight = K.constant(tree_weight)
        # self.tree_weight = tree_weight
    
    def call(self, inputs):
        output = K.dot(inputs, tf.multiply(self.kernel, self.tree_weight))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_weights(self):
        # ref: https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/keras/engine/base_layer.py#L21
        params = self.weights
        weights = K.batch_get_value(params)
        return weights[0]*K.get_value(self.tree_weight), weights[1]
    
# class Dense_with_tree_schedule(Dense):
#     def __init__(self, units,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  tree_weight=None,
#                  tree_noise_weight=None,
#                  **kwargs):
#         super(Dense_with_tree_schedule, self).__init__(units, 
#                                               activation=activation,
#                                               use_bias=use_bias,
#                                               kernel_initializer=kernel_initializer,
#                                               bias_initializer=bias_initializer,
#                                               kernel_regularizer=kernel_regularizer,
#                                               bias_regularizer=bias_regularizer,
#                                               activity_regularizer=activity_regularizer,
#                                               kernel_constraint=kernel_constraint,
#                                               bias_constraint=bias_constraint,
#                                               **kwargs)
#         self.tree_weight = K.constant(tree_weight)
#         self.tree_noise_weight = K.constant(tree_noise_weight)
#         self.alpha = K.variable(1.)
    
#     def call(self, inputs):
#         self.scheduled_tree_weight = self.alpha * self.tree_noise_weight + (1.-self.alpha) * self.tree_weight 
#         output = K.dot(inputs, tf.multiply(self.kernel, self.scheduled_tree_weight))
#         self.alpha = self.alpha * 0.99
        
#         if self.use_bias:
#             output = K.bias_add(output, self.bias, data_format='channels_last')
#         if self.activation is not None:
#             output = self.activation(output)
#         return output
    
#     def get_weights(self):
#         # ref: https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/keras/engine/base_layer.py#L21
#         params = self.weights
#         weights = K.batch_get_value(params)
#         return weights[0]*K.get_value(self.scheduled_tree_weight), weights[1]

#####################################################################################################################
class Dense_with_new_tree(Dense):
    ## Ref: https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L796
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tree_weight=None,
                 tree_thrd = False,
                 **kwargs):
        super(Dense_with_new_tree, self).__init__(units, 
                                              activation=activation,
                                              use_bias=use_bias,
                                              kernel_initializer=kernel_initializer,
                                              bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              bias_constraint=bias_constraint,
                                              **kwargs)
        self.tree_weight = tree_weight
        self.tree_thrd = tree_thrd
        dtype = K.floatx()
        self.dtype = dtype
        
        self.mask_list = []
        self.np_mask_list = []
        for i in range(self.tree_weight.shape[1]):
            mask = np.zeros([self.tree_weight[:,i].shape[0], np.sum(self.tree_weight[:,i]==1)])
            for j, axk in enumerate(np.where(self.tree_weight[:,i] == 1)[0]):
                mask[axk,j] = 1.
            self.np_mask_list.append(mask)
            self.mask_list.append(K.constant(mask))
        
    def add_weight(self,
                   name,
                   shape,
                   tree_thrd=False,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        # ref: https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/keras/engine/base_layer.py#L216
        from keras import initializers
        
        initializer = initializers.get(initializer)
        if dtype is None:
            dtype = self.dtype
        
        
        weight = K.variable(initializer(shape, dtype=dtype),
                            dtype=dtype,
                            name=name,
                            constraint=constraint)
        if tree_thrd:
            weight_list = []
            for i in range(self.tree_weight.shape[1]):
                new_weight = K.variable(initializer((np.sum(self.tree_weight[:,i]==1),1), dtype=dtype), 
                                        dtype=dtype, name='%s_%d'%(name,i), constraint=constraint)
                weight_list.append(new_weight)
                if regularizer is not None:
                    with K.name_scope('weight_regularizer'):
                        self.add_loss(regularizer(new_weight))
                if trainable:
                    self._trainable_weights.append(new_weight)
                else:
                    self._non_trainable_weights.append(new_weight)
            weight = weight_list
        else:
            weight = K.variable(initializer(shape, dtype=dtype),
                            dtype=dtype,
                            name=name,
                            constraint=constraint)
            if regularizer is not None:
                with K.name_scope('weight_regularizer'):
                    self.add_loss(regularizer(weight))
            if trainable:
                self._trainable_weights.append(weight)
            else:
                self._non_trainable_weights.append(weight)
        return weight
    
    def build(self, input_shape):
        from keras.engine.base_layer import InputSpec
        
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      tree_thrd=self.tree_thrd,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs):
        if self.tree_thrd:
            output = [K.squeeze(K.dot(K.dot(inputs, self.mask_list[i]), self.kernel[i]), axis=-1) for i in range(self.tree_weight.shape[1])]
            output = K.stack(output, axis=1)
        else: output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_weights(self):
        # ref: https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/keras/engine/base_layer.py#L21
        params = self.weights
        weights = K.batch_get_value(params)
        
        if self.tree_thrd:
            kernel_weights = np.copy(self.tree_weight)
            for upper, (kernel, mask) in enumerate(zip(weights[:-1], self.np_mask_list)):
                for where_k, where_v in np.argwhere(mask):
                    kernel_weights[where_k,upper] = kernel[where_v]
            return kernel_weights, weights[-1]
        else:
            return weights[0], weights[-1]
    
#####################################################################################################################
#     initializer with phylogenetic tree information class
#####################################################################################################################
class VarianceScaling_with_tree(VarianceScaling):
    ## ref : https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/keras/initializers.py#L155
    def __init__(self, 
                 tree_weight,
                 scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None):
        super(VarianceScaling_with_tree, self).__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)
        self.tree_weight = tree_weight
        
    def __call__(self, shape, dtype=None):
        from keras.initializers import _compute_fans
        
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .87962566103423978
            init = K.truncated_normal(shape, 0., stddev,
                                      dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3. * scale)
            init = K.random_uniform(shape, -limit, limit,
                                    dtype=dtype, seed=self.seed)
        
        return init * self.tree_weight

    def get_config(self):
        return {
#             'tree_weight': self.tree_weight,
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed
        }
    
def glorot_uniform_with_tree(tree_weight, seed=None):
    return VarianceScaling_with_tree(tree_weight, scale=1., mode='fan_avg', distribution='uniform', seed=seed)

def he_normal_with_tree(tree_weight, seed=None):
    return VarianceScaling_with_tree(tree_weight, scale=2., mode='fan_in', distribution='normal', seed=seed)

#####################################################################################################################
#     DeepBiome Networks
#####################################################################################################################
class DeepBiomeNetwork(Base_Network):
    def __init__(self, network_info, path_info, log, fold=None, num_classes = 1, 
                 tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                 is_covariates=False, covariate_names = None,
                 lvl_category_dict = None,
                 verbose=True):
        super(DeepBiomeNetwork,self).__init__(network_info, log)
        if fold != None: self.fold = fold
        else: self.fold = ''
        # self.TB = TensorBoardWrapper_DeepBiome
        # self.TB = TensorBoardWrapper
        self.num_classes = num_classes
        self.build_model(path_info=path_info,
                         tree_level_list = tree_level_list,
                         is_covariates=is_covariates, covariate_names=covariate_names, 
                         lvl_category_dict = lvl_category_dict,
                         verbose=verbose)
    
    def set_phylogenetic_tree_info(self, tree_path, tree_level_list = ['Genus', 'Family', 'Order', 'Class', 'Phylum'], 
                                   is_covariates=False, covariate_names = None,
                                   lvl_category_dict = None,
                                   verbose=True):
        if verbose: 
            self.log.info('------------------------------------------------------------------------------------------')
            self.log.info('Read phylogenetic tree information from %s' % tree_path)
        self.phylogenetic_tree_info = pd.read_csv('%s' % tree_path)
        self.tree_level_list = [lvl_name for lvl_name in tree_level_list if lvl_name in self.phylogenetic_tree_info.columns.tolist()]
        self.phylogenetic_tree_info = self.phylogenetic_tree_info[self.tree_level_list]
        # self.tree_level_list = self.phylogenetic_tree_info.columns.tolist()
        if verbose: 
            self.log.info('Phylogenetic tree level list: %s' % self.tree_level_list)
            self.log.info('------------------------------------------------------------------------------------------')
        self.phylogenetic_tree_dict = {'Number':{}}
        for i, tree_lvl in enumerate(self.tree_level_list):
            if lvl_category_dict == None: 
                lvl_category = self.phylogenetic_tree_info[tree_lvl].unique()
            else: 
                lvl_category = lvl_category_dict[i]
            lvl_num = lvl_category.shape[0]
            if verbose: self.log.info('    %6s: %d' % (tree_lvl, lvl_num))
            self.phylogenetic_tree_dict[tree_lvl] = dict(zip(lvl_category, np.arange(lvl_num)))
            self.phylogenetic_tree_dict['Number'][tree_lvl]=lvl_num
            if is_covariates and i==len(self.tree_level_list)-1:
                lvl_category = np.append(lvl_category, covariate_names)
                lvl_num = lvl_category.shape[0]
                if verbose: self.log.info('    %6s: %d' % ('%s_with_covariates'%tree_lvl, lvl_num))
                self.phylogenetic_tree_dict['%s_with_covariates' % tree_lvl] = dict(zip(lvl_category, np.arange(lvl_num)))
        if verbose: 
            self.log.info('------------------------------------------------------------------------------------------')
            self.log.info('Phylogenetic_tree_dict info: %s' % list(self.phylogenetic_tree_dict.keys()))
            self.log.info('------------------------------------------------------------------------------------------')
        self.phylogenetic_tree = copy.deepcopy(self.phylogenetic_tree_info.iloc[:,:])
        for tree_lvl in self.tree_level_list:
            self.phylogenetic_tree[tree_lvl] = self.phylogenetic_tree[tree_lvl].map(self.phylogenetic_tree_dict[tree_lvl])
        self.phylogenetic_tree = np.array(self.phylogenetic_tree)
        
        self.tree_weight_list = []
        self.tree_weight_noise_list = []
        num_dict = self.phylogenetic_tree_dict['Number']
        for i in range(len(self.tree_level_list)-1):
            if verbose: self.log.info('Build edge weights between [%6s, %6s]'%(self.tree_level_list[i],self.tree_level_list[i+1]))
            lower = self.phylogenetic_tree[:,i]
            upper = self.phylogenetic_tree[:,i+1]
            n_lower = num_dict[self.tree_level_list[i]]
            n_upper = num_dict[self.tree_level_list[i+1]]

            tree_w = np.zeros((n_lower,n_upper))
            tree_w_n = np.zeros_like(tree_w) + 0.01
            for j in range(n_upper):
                tree_w[lower[j==upper],j] = 1.
                tree_w_n[lower[j==upper],j] = 1.
            self.tree_weight_list.append(tree_w)
            self.tree_weight_noise_list.append(tree_w_n)
        if verbose: self.log.info('------------------------------------------------------------------------------------------')
            
    def build_model(self, path_info, tree_level_list=['Genus', 'Family', 'Order', 'Class', 'Phylum'],
                    is_covariates=False, covariate_names = None, 
                    lvl_category_dict = None,
                    verbose=True):
        # Load Tree Weights
        self.set_phylogenetic_tree_info(path_info['data_info']['tree_info_path'], 
                                        tree_level_list = tree_level_list,
                                        is_covariates=is_covariates, covariate_names=covariate_names, 
                                        lvl_category_dict = lvl_category_dict,
                                        verbose=verbose)
        
        # Build model
        if verbose: 
            self.log.info('------------------------------------------------------------------------------------------')
            self.log.info('Build network based on phylogenetic tree information')
            self.log.info('------------------------------------------------------------------------------------------')

        weight_initial = self.network_info['architecture_info']['weight_initial'].strip()
        bn = self.network_info['architecture_info']['batch_normalization'].strip()=='True'
        dropout_p = float(self.network_info['architecture_info']['drop_out'].strip())
        
        l1_panelty = self.network_info['architecture_info'].get('weight_l1_panelty', None)
        l2_panelty = self.network_info['architecture_info'].get('weight_l2_panelty', None)
        if l1_panelty != None: kernel_regularizer = keras.regularizers.l1(int(l1_panelty))
        elif l2_panelty != None: kernel_regularizer = keras.regularizers.l2(int(l2_panelty))
        else: kernel_regularizer = None
        
        if self.network_info['architecture_info'].get('tree_thrd', 'False').strip() == 'True': tree_thrd = True
        else: tree_thrd = False
        
        # if weight_initial == 'true_disease_weight': 
        #     true_disease_weight_list, true_disease_bias_list = self.load_true_disease_weight_list(tree_info_path['disease_weight_path'])
            
        weight_decay = self.network_info['architecture_info'].get('weight_decay', None)
        if weight_decay != None: weight_decay = weight_decay.strip()
        if weight_decay == 'None': weight_decay = None
        
        x_input = Input(shape=(self.tree_weight_list[0].shape[0],), name='input')
        if is_covariates: covariates_input = Input(shape=covariate_names.shape[0:], name='covariates_input')
        l = x_input
        for i, (tree_w, tree_wn) in enumerate(zip(self.tree_weight_list, self.tree_weight_noise_list)):
            bias_initializer='zeros'
            if weight_initial == 'phylogenetic_tree_glorot_uniform': kernel_initializer = glorot_uniform_with_tree(tree_w, seed=123)
            elif weight_initial == 'phylogenetic_tree_he_normal': kernel_initializer = he_normal_with_tree(tree_w, seed=123)
            # elif weight_initial == 'true_disease_weight': 
            #     kernel_initializer = keras.initializers.Constant(true_disease_weight_list[i])
            #     bias_initializer = keras.initializers.Constant(true_disease_bias_list[i])
            else: kernel_initializer = weight_initial
                
            if weight_decay == 'phylogenetic_tree': 
                l = Dense_with_tree(tree_w.shape[1], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer, tree_weight=tree_wn, name='l%d_dense'%(i+1))(l)
            elif weight_decay == 'phylogenetic_tree_wo_noise': 
                l = Dense_with_tree(tree_w.shape[1], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer, tree_weight=tree_w, name='l%d_dense'%(i+1))(l)
            # elif weight_decay == 'phylogenetic_tree_schedule': 
            #     l = Dense_with_tree_schedule(tree_w.shape[1], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            #                         kernel_regularizer=kernel_regularizer, tree_weight=tree_w, tree_noise_weight=tree_wn, name='l%d_dense'%(i+1))(l)
            else:
                l = Dense_with_new_tree(tree_w.shape[1], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer, tree_weight=tree_w, tree_thrd=tree_thrd, name='l%d_dense'%(i+1))(l)
            
            if bn: l = BatchNormalization(name='l%d_bn'%(i+1))(l)
            
            l = Activation('relu', name='l%d_activation'%(i+1))(l)
            
            if dropout_p: l = Dropout(dropout_p, name='l%d_dropout'%(i+1))(l)
                
        # l = Dense(self.tree_weight_list[-1].shape[-1], kernel_initializer='he_normal', name='pre_last_h')(l)
        # l = BatchNormalization(name='pre_last_bn')(l)
        
        # if weight_initial == 'true_disease_weight':
        #     kernel_initializer = keras.initializers.Constant(true_disease_weight_list[len(self.tree_weight_list)])
        #     bias_initializer = keras.initializers.Constant(true_disease_bias_list[len(self.tree_weight_list)])
        # else: 
        #     kernel_initializer = 'he_normal'
        #     bias_initializer = 'zeros'
        
        kernel_initializer = 'he_normal'
        bias_initializer = 'zeros'
        
        if is_covariates:
            l = Concatenate(name='biome_covariates_concat')([l,covariates_input])
            
        last_h = Dense(max(1,self.num_classes),
                       kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='last_dense_h')(l)
        
        if self.num_classes == 0:
            p_hat = Activation('linear', name='p_hat')(last_h)
        elif self.num_classes == 1:
            p_hat = Activation('sigmoid', name='p_hat')(last_h)
        else: 
            p_hat = Activation('softmax', name='p_hat')(last_h)
        if is_covariates: self.model = Model(inputs=[x_input, covariates_input], outputs=p_hat)
        else: self.model = Model(inputs=x_input, outputs=p_hat)
        if verbose: 
            self.model.summary()
            self.log.info('------------------------------------------------------------------------------------------')
    
    def get_trained_weight(self):
        kernel_lists =  [l.get_weights()[0] for l in self.model.layers if 'dense' in l.name]
        kernel_lists_with_name = []
        for i in range(len(kernel_lists)):
            try:
                lower_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i]].items()])
                lower_colname = [lower_dict[ky] for ky in range(len(lower_dict))]
                if len(lower_colname) < kernel_lists[i].shape[0]:
                    lower_colname = lower_colname + list(np.arange(kernel_lists[i].shape[0] - len(lower_colname)))
            except:
                lower_colname = np.arange(kernel_lists[i].shape[0])
            try:
                upper_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i+1]].items()])
                upper_colname = [upper_dict[ky] for ky in range(len(upper_dict))]
                if len(upper_colname) < kernel_lists[i].shape[-1]:
                    upper_colname = upper_colname + list(np.arange(kernel_lists[i].shape[-1] - len(upper_colname)))
            except:
                upper_colname = np.arange(kernel_lists[i].shape[-1])
            kernel_lists_with_name.append(pd.DataFrame(kernel_lists[i], columns=upper_colname, index=lower_colname))
        return kernel_lists_with_name
    
    def get_trained_bias(self):
        kernel_lists =  [l.get_weights()[1] for l in self.model.layers if 'dense' in l.name]
        return kernel_lists
    
    def get_tree_weight(self):
        kernel_lists_with_name = []
        for i in range(len(self.tree_weight_list)):
            try:
                lower_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i]].items()])
                lower_colname = [lower_dict[ky] for ky in range(len(lower_dict))]
                if len(lower_colname) < self.tree_weight_list[i].shape[0]:
                    lower_colname = lower_colname + list(range(self.tree_weight_list[i].shape[0] - len(lower_colname)))
            except:
                lower_colname = np.arange(self.tree_weight_list[i].shape[0])
            try:
                upper_dict = dict([(y,x) for x,y in self.phylogenetic_tree_dict[self.tree_level_list[i+1]].items()])
                upper_colname = [upper_dict[ky] for ky in range(len(upper_dict))]
                if len(upper_colname) < self.tree_weight_list[i].shape[-1]:
                    upper_colname = upper_colname + list(range(self.tree_weight_list[i].shape[-1] - len(upper_colname)))
            except:
                upper_colname = np.arange(self.tree_weight_list[i].shape[-1])
            kernel_lists_with_name.append(pd.DataFrame(self.tree_weight_list[i], columns=upper_colname, index=lower_colname))
        return kernel_lists_with_name
    
#     def load_true_tree_weight_list(self, data_path):
#         true_tree_weight_list = [np.load('%s/tw_%d.npy'%(data_path,i))[self.fold] for i in range(1,len(self.tree_level_list))]
#         return true_tree_weight_list
    
#     def load_true_disease_weight_list(self, data_path):
#         true_disease_weight_list = [np.load('%s/solw_%d.npy'%(data_path,i))[self.fold] for i in range(len(self.tree_level_list))]
#         true_disease_bias_list = [np.load('%s/solb_%d.npy'%(data_path,i))[self.fold] for i in range(len(self.tree_level_list))]
#         return true_disease_weight_list, true_disease_bias_list
        