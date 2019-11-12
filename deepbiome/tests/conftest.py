import pytest

import sys
import numpy as np
from pkg_resources import resource_filename

from deepbiome import configuration
from deepbiome import logging_daily

    
####################################################################################################################
# Classification
####################################################################################################################
@pytest.fixture
def input_value_classification_training():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/classification_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/classification_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

@pytest.fixture
def input_value_classification_deep_training():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/classification_deep_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/classification_deep_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info


@pytest.fixture
def input_value_classification_deep_l1_training():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/classification_deep_l1_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/classification_deep_l1_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

@pytest.fixture
def input_value_classification_warmstart():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/warmstart_classification_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/warmstart_classification_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

@pytest.fixture
def input_value_classification_test():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/test_classification_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/test_classification_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

@pytest.fixture
def input_value_classification_prediction():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/prediction_classification_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/prediction_classification_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

@pytest.fixture
def output_value_classification():
    training_evaluation = np.load(resource_filename('deepbiome', 'tests/data/classification_real_train_evaluation.npy'))
    test_evaluation = np.load(resource_filename('deepbiome', 'tests/data/classification_real_test_evaluation.npy'))
    return training_evaluation, test_evaluation

####################################################################################################################
# Regression
####################################################################################################################
@pytest.fixture
def input_value_regression_training():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/regression_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/regression_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

@pytest.fixture
def input_value_regression_deep_training():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/regression_deep_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/regression_deep_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

@pytest.fixture
def input_value_regression_deep_l1_training():
    logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/regression_deep_l1_path_info.cfg'), log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/regression_deep_l1_network_info.cfg'), log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    
    for k, v in path_info['data_info'].items():
        if 'data' in v:
            resource_filename('deepbiome', 'tests/%s' % v)
            path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
    return log, network_info, path_info

# @pytest.fixture
# def input_value_regression_warmstart():
#     logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
#     logger.reset_logging()
#     log = logger.get_logging()
#     log.setLevel(logging_daily.logging.INFO)
    
#     config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/warmstart_regression_path_info.cfg'), log)
#     config_data.set_config_map(config_data.get_section_map())
#     config_data.print_config_map()

#     config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/warmstart_regression_network_info.cfg'), log)
#     config_network.set_config_map(config_network.get_section_map())
#     config_network.print_config_map()

#     path_info = config_data.get_config_map()
#     network_info = config_network.get_config_map()
    
#     for k, v in path_info['data_info'].items():
#         if 'data' in v:
#             resource_filename('deepbiome', 'tests/%s' % v)
#             path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
#     return log, network_info, path_info

# @pytest.fixture
# def input_value_regression_test():
#     logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
#     logger.reset_logging()
#     log = logger.get_logging()
#     log.setLevel(logging_daily.logging.INFO)
    
#     config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/test_regression_path_info.cfg'), log)
#     config_data.set_config_map(config_data.get_section_map())
#     config_data.print_config_map()

#     config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/test_regression_network_info.cfg'), log)
#     config_network.set_config_map(config_network.get_section_map())
#     config_network.print_config_map()

#     path_info = config_data.get_config_map()
#     network_info = config_network.get_config_map()
    
#     for k, v in path_info['data_info'].items():
#         if 'data' in v:
#             resource_filename('deepbiome', 'tests/%s' % v)
#             path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
#     return log, network_info, path_info


# @pytest.fixture
# def input_value_regression_prediction():
#     logger = logging_daily.logging_daily(resource_filename('deepbiome', 'tests/data/log_info.yaml'))
#     logger.reset_logging()
#     log = logger.get_logging()
#     log.setLevel(logging_daily.logging.INFO)
    
#     config_data = configuration.Configurator(resource_filename('deepbiome', 'tests/data/prediction_regression_path_info.cfg'), log)
#     config_data.set_config_map(config_data.get_section_map())
#     config_data.print_config_map()

#     config_network = configuration.Configurator(resource_filename('deepbiome', 'tests/data/prediction_regression_network_info.cfg'), log)
#     config_network.set_config_map(config_network.get_section_map())
#     config_network.print_config_map()

#     path_info = config_data.get_config_map()
#     network_info = config_network.get_config_map()
    
#     for k, v in path_info['data_info'].items():
#         if 'data' in v:
#             resource_filename('deepbiome', 'tests/%s' % v)
#             path_info['data_info'][k] = resource_filename('deepbiome', 'tests/%s' % v)
#     return log, network_info, path_info


@pytest.fixture
def output_value_regression():
    training_evaluation = np.load(resource_filename('deepbiome', 'tests/data/regression_real_train_evaluation.npy'))
    test_evaluation = np.load(resource_filename('deepbiome', 'tests/data/regression_real_test_evaluation.npy'))
    return training_evaluation, test_evaluation

