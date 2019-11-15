#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for `deepbiome` package.

export CUDA_VISIBLE_DEVICES=0; pytest -s --log-cli-level=10
"""

import pytest

import os

import random
import tensorflow as tf
import numpy as np
from pkg_resources import resource_filename

from deepbiome import deepbiome
from deepbiome import loss_and_metric
from deepbiome import utils

def test_utils():
    print(utils.argv_parse(['main.py','--log_info=log_info.yaml']))

def test_loss_numpy():
    y_true_set = np.array([[0,1],
                            [1,0],
                            [0,1],
                            [1,0],
                            [1,0]])
    y_pred_set = np.array([[0,1],
                            [0.1,0.9],
                            [0.4,0.6],
                            [0,1],
                            [0.7,0.3]])
    assert np.all(np.isclose(np.array([0.6, 0.6, 0.6, 0.6, 0.5833333333333333]), loss_and_metric.metric_test(y_true_set, y_pred_set)))
    
####################################################################################################################
# Classification
####################################################################################################################
def test_deepbiome_classification_training(input_value_classification_training, output_value_classification):
    '''
    Test deepbiome by classification problem with simulated data
    '''
    log, network_info, path_info = input_value_classification_training
    real_train_evaluation, real_test_evaluation = output_value_classification
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=2)
    # np.save('data/classification_real_train_evaluation.npy', train_evaluation)
    # np.save('data/classification_real_test_evaluation.npy', test_evaluation)
    
    log.info('test')
    log.info(real_test_evaluation)
    log.info(test_evaluation)
    log.info(np.all(np.isclose(real_test_evaluation, test_evaluation)))
    
    log.info('training')
    log.info(real_train_evaluation)
    log.info(train_evaluation)
    log.info(np.all(np.isclose(real_train_evaluation, train_evaluation)))
    # assert np.all(np.isclose(real_test_evaluation, test_evaluation)) & np.all(np.isclose(real_train_evaluation, train_evaluation))
    
def test_deepbiome_classification_deep_training(input_value_classification_deep_training):
    '''
    Test deepbiome by classification problem with simulated data
    '''
    log, network_info, path_info = input_value_classification_deep_training
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=2)
    
    log.info('test')
    log.info(test_evaluation)
    
    log.info('training')
    log.info(train_evaluation)
    
def test_deepbiome_classification_deep_l1_training(input_value_classification_deep_l1_training):
    '''
    Test deepbiome by classification problem with simulated data
    '''
    log, network_info, path_info = input_value_classification_deep_l1_training
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=2)
    
    log.info('test')
    log.info(test_evaluation)
    
    log.info('training')
    log.info(train_evaluation)
    
def test_deepbiome_classification_warmstart(input_value_classification_warmstart):
    '''
    Test deepbiome by classification problem with simulated data
    '''
    log, network_info, path_info = input_value_classification_warmstart
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=2)
    # np.save('data/classification_real_train_evaluation.npy', train_evaluation)
    # np.save('data/classification_real_test_evaluation.npy', test_evaluation)
    
    log.info('test')
    log.info(test_evaluation)
    
    log.info('training')
    log.info(train_evaluation)

def test_deepbiome_classification_test(input_value_classification_test):
    '''
    Test deepbiome by classification problem with simulated data
    '''
    log, network_info, path_info = input_value_classification_test
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    evaluation = deepbiome.deepbiome_test(log, network_info, path_info, number_of_fold=2)
    
    log.info('test evaluation')
    log.info(evaluation)
    
def test_deepbiome_classification_predition(input_value_classification_prediction):
    '''
    Test deepbiome by classification problem with simulated data
    '''
    log, network_info, path_info = input_value_classification_prediction
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    prediction = deepbiome.deepbiome_prediction(log, network_info, path_info, num_classes = 1, number_of_fold=2)
    
    log.info('prediction result')
    log.info(prediction.shape)
    log.info(prediction[0,:10])
    
####################################################################################################################
# Regression
####################################################################################################################
def test_deepbiome_regression_training(input_value_regression_training, output_value_regression):
    '''
    Test deepbiome by regression problem with simulated data
    '''
    log, network_info, path_info = input_value_regression_training
    real_train_evaluation, real_test_evaluation = output_value_regression
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=2)
    # np.save('data/regression_real_train_evaluation.npy', train_evaluation)
    # np.save('data/regression_real_test_evaluation.npy', test_evaluation)
    
    log.info('test')
    log.info(real_test_evaluation)
    log.info(test_evaluation)
    log.info(np.all(np.isclose(real_test_evaluation, test_evaluation)))
    
    log.info('training')
    log.info(real_train_evaluation)
    log.info(train_evaluation)
    log.info(np.all(np.isclose(real_train_evaluation, train_evaluation)))
    # assert np.all(np.isclose(real_test_evaluation, test_evaluation)) & np.all(np.isclose(real_train_evaluation, train_evaluation))
    
def test_deepbiome_regression_deep_training(input_value_regression_deep_training, output_value_regression):
    '''
    Test deepbiome by regression problem with simulated data
    '''
    log, network_info, path_info = input_value_regression_deep_training
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=2)
    # np.save('data/regression_real_train_evaluation.npy', train_evaluation)
    # np.save('data/regression_real_test_evaluation.npy', test_evaluation)
    
    log.info('test')
    log.info(test_evaluation)
    
    log.info('training')
    log.info(train_evaluation)
    
def test_deepbiome_regression_deep_l1_training(input_value_regression_deep_l1_training, output_value_regression):
    '''
    Test deepbiome by regression problem with simulated data
    '''
    log, network_info, path_info = input_value_regression_deep_l1_training
    
    seed_value = 123
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf.__version__.startswith('2'): tf.random.set_seed(seed_value)
    else: tf.set_random_seed(seed_value)
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=2)
    # np.save('data/regression_real_train_evaluation.npy', train_evaluation)
    # np.save('data/regression_real_test_evaluation.npy', test_evaluation)
    
    log.info('test')
    log.info(test_evaluation)
    
    log.info('training')
    log.info(train_evaluation)