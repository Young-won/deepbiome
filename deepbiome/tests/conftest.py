import pytest
import logging

@pytest.fixture
def input_value():
    logging.basicConfig(format = '[%(name)-8s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s',
                    level=logging.DEBUG)
    log = logging.getLogger()
    
    network_info = {
        'architecture_info': {
            'batch_normalization': 'False',
            'drop_out': '0',
            'weight_initial': 'glorot_uniform',
            'weight_l1_penalty':'0.01',
            'weight_decay': 'phylogenetic_tree',
        },
        'model_info': {
            'decay': '0.001',
            'loss': 'binary_crossentropy',
            'lr': '0.01',
            'metrics': 'binary_accuracy, sensitivity, specificity, gmeasure, auc',
            'network_class': 'DeepBiomeNetwork',
            'normalizer': 'normalize_minmax',
            'optimizer': 'adam',
            'reader_class': 'MicroBiomeReader',
            'texa_selection_metrics': 'accuracy, sensitivity, specificity, gmeasure'
        },
        'tensorboard_info': {
            'histogram_freq': '0',
            'tensorboard_dir': 'None',
            'write_grads': 'False',
            'write_graph': 'False',
            'write_image': 'False',
            'write_weights_histogram': 'False',
            'write_weights_images': 'False'},
        'test_info': {
            'batch_size': 'None'
        },
        'training_info': {
            'batch_size': '200', 'epochs': '10'
        },
        'validation_info': {
            'batch_size': 'None', 'validation_size': '0.2'
        }
    }
    
    path_info = {
        'data_info': {
            'count_list_path': 'data/simulation/gcount_list.csv',
            'count_path': 'data/simulation/count',
            'data_path': 'data/simulation/s2/',
            'idx_path': 'data/simulation/s2/idx.csv',
            'tree_info_path': 'data/genus48/genus48_dic.csv',
            'x_path': '',
            'y_path': 'y.csv'
        },
        'model_info': {
            'evaluation': 'eval.npy',
            'history': 'history/hist.json',
            'model_dir': './simulation_s2/simulation_s2_deepbiome/',
            'weight': 'weight/weight.h5'
        }
    }
    return log, network_info


