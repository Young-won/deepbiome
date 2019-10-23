import pytest

import sys
import logging
import numpy as np
from deepbiome import configuration

@pytest.fixture
def input_value():
    formatter = logging.Formatter('[%(name)-8s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s')
    logging.basicConfig(format = formatter,
                    level=logging.DEBUG)
    log = logging.getLogger()
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # handler.setFormatter(formatter)
    # log.addHandler(handler)
    
    config_data = configuration.Configurator('./data/path_info.cfg', log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()

    config_network = configuration.Configurator('./data/network_info.cfg', log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()
    return log, network_info, path_info

@pytest.fixture
def output_value():
    training_evaluation = np.load('./data/real_train_evaluation.npy')
    test_evaluation = np.load('./data/real_test_evaluation.npy')
    return training_evaluation, test_evaluation