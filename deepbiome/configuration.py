######################################################################
## Medical segmentation using CNN
## - Read config
##
## Nov 16. 2018
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################

import os
import configparser
 
class Configurator(object):
    def __init__(self, config_file, log, verbose=True):
        self.log = log
        self.verbose = verbose
        self.load_config(config_file)
        #self.flag = True
 
    def load_config(self,config_file):
        if os.path.exists(config_file)==False:
            raise Exception("%s file does not exist.\n" % config_file)    
        self.config = configparser.ConfigParser()
        if self.verbose: self.log.info('Configuration with {}'.format(config_file))
        self.config.read(config_file)

    def get_section_map(self):
        return self.config.sections()
        
    def set_config_map(self, section_map):
        if self.verbose: self.log.info('Set configuration map with section {}'.format(section_map))
        self.config_map = dict(zip(section_map, map(lambda section : dict(self.config.items(section)), section_map)))
    
    def print_config_map(self):
        self.log.info('    Configuration map')
        for section, item in self.config_map.items():
            self.log.info('        {} \t: {}'.format(section, item.keys()))
        
    def get_config_map(self):
        return self.config_map