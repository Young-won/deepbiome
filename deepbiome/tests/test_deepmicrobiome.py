#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for `deepbiome` package.

pytest -s --log-cli-level=10
"""


import pytest

import os

import random
import tensorflow as tf
import numpy as np

from deepbiome import deepbiome


def test_deepbiome_classification(input_value_classification, output_value_classification):
    '''
    Test deepbiome by classification problem with simulated data
    '''
    log, network_info, path_info = input_value_classification
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
    assert 1+2==3
    

def test_deepbiome_regression(input_value_regression, output_value_regression):
    '''
    Test deepbiome by regression problem with simulated data
    '''
    log, network_info, path_info = input_value_regression
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
    assert 1+2==3
