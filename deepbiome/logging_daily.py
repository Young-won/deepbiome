######################################################################
## Medical segmentation using CNN
## - Custom logging
##
## Nov 16. 2018
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################


import datetime
import os
import logging
import logging.config
import logging.handlers
import yaml

class logging_daily(object):
    def __init__(self, filename):
        self.filename = filename
        self.config_yaml(self.filename)
        # Apply the configuration
        logging.config.dictConfig(self.config_dict)
    
    def config_yaml(self, yaml_filename):
        if os.path.exists(yaml_filename)==False:
            raise Exception("%s file does not exist.\n" % yaml_filename)
        with open(yaml_filename) as f:
            self.config_dict = yaml.load(f)
            # Append the date stamp to the file name
            log_filename = self.config_dict['handlers']['fileHandler']['filename']
            base, extension = os.path.splitext(log_filename)
            try:
                os.stat(os.path.dirname(base))
            except:
                os.mkdir(os.path.dirname(base))
            today = datetime.datetime.today()
            log_filename = '{}{}{}'.format(
                base,
                today.strftime('_%Y%m%d'),
                extension)
            self.config_dict['handlers']['fileHandler']['filename'] = log_filename
    
    def get_config_dict(self):
        return self.config_dict
    
    def get_log_file(self):
        return self.filename
    
    def get_logging(self, name='default'):
        return logging.getLogger(name)
        
    def reset_logging(self):
        with open(self.config_dict['handlers']['fileHandler']['filename'], 'w'):
            pass

if __name__ == '__main__':
    logger = logging_daily('./base_model/config/log_info.yaml')
    log = logger.get_logging('default')
    log.debug('debug message')
    log.info('info message')