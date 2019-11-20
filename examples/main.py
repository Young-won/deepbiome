"""
DeepBiome
- Main code

July 10. 2019
Youngwon (youngwon08@gmail.com)

Reference
- Keras (https://github.com/keras-team/keras)
"""

import sys

from deepbiome import configuration
from deepbiome import logging_daily
from deepbiome import deepbiome
from deepbiome.utils import argv_parse

# Argument ##########################################################
argdict = argv_parse(sys.argv)
try: gpu_memory_fraction = float(argdict['gpu_memory_fraction'][0]) 
except: gpu_memory_fraction = None
try: max_queue_size=int(argdict['max_queue_size'][0])
except: max_queue_size=10
try: workers=int(argdict['workers'][0])
except: workers=1
try: use_multiprocessing=argdict['use_multiprocessing'][0]=='True'      
except: use_multiprocessing=False

# Logger ###########################################################
logger = logging_daily.logging_daily(argdict['log_info'][0])
logger.reset_logging()
log = logger.get_logging()
log.setLevel(logging_daily.logging.INFO)

log.info('Argument input')
for argname, arg in argdict.items():
    log.info('    {}:{}'.format(argname,arg))

# Configuration ####################################################
config_data = configuration.Configurator(argdict['path_info'][0], log)
config_data.set_config_map(config_data.get_section_map())
config_data.print_config_map()

config_network = configuration.Configurator(argdict['network_info'][0], log)
config_network.set_config_map(config_network.get_section_map())
config_network.print_config_map()

path_info = config_data.get_config_map()
network_info = config_network.get_config_map()
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info) #, number_of_fold=100)