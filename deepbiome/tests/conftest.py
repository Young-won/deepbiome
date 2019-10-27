import pytest

import sys
import logging
import numpy as np
from pkg_resources import resource_filename

from deepbiome import configuration

@pytest.fixture
def input_value_regression():
    formatter = logging.Formatter('[%(name)-8s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s')
    logging.basicConfig(format = formatter,
                    level=logging.DEBUG)
    log = logging.getLogger()
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # handler.setFormatter(formatter)
    # log.addHandler(handler)
    
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
def output_value_regression():
    training_evaluation = np.load(resource_filename('deepbiome', 'tests/data/regression_real_train_evaluation.npy'))
    test_evaluation = np.load(resource_filename('deepbiome', 'tests/data/regression_real_test_evaluation.npy'))
    return training_evaluation, test_evaluation

@pytest.fixture
def input_value_classification():
    formatter = logging.Formatter('[%(name)-8s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s')
    logging.basicConfig(format = formatter,
                    level=logging.DEBUG)
    log = logging.getLogger()
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # handler.setFormatter(formatter)
    # log.addHandler(handler)
    
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
def output_value_classification():
    training_evaluation = np.load(resource_filename('deepbiome', 'tests/data/classification_real_train_evaluation.npy'))
    test_evaluation = np.load(resource_filename('deepbiome', 'tests/data/classification_real_test_evaluation.npy'))
    return training_evaluation, test_evaluation
