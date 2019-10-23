######################################################################
## DeepBiome
## - Loss and metrics (mse, cross-entropy)
##
## July 10. 2019
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################

import numpy as np
import sklearn.metrics as skmetrics

from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error, mean_absolute_error, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from keras.metrics import binary_accuracy, categorical_accuracy, sparse_categorical_accuracy
from sklearn.metrics import roc_auc_score, f1_score

###############################################################################################################################
# tf loss functions
def precision(y_true, y_pred):
    return K.sum(y_true*y_pred)/(K.sum(y_true*y_pred) + K.sum((1-y_true)*y_pred) + 1e-10)

def recall(y_true, y_pred):
    return K.sum(y_true*y_pred)/(K.sum(y_true*y_pred) + K.sum(y_true*(1-y_pred)) + 1e-10)

def sensitivity(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(y_true * y_pred)
    data_positives = K.sum(y_true)
    return true_positives / (data_positives + K.epsilon())

def specificity(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_negatives = K.sum((1-y_true) * (1-y_pred))
    data_positives = K.sum(1-y_true)
    return true_negatives / (data_positives + K.epsilon())

def gmeasure(y_true, y_pred):
    return (sensitivity(y_true, y_pred) * specificity(y_true, y_pred)) ** 0.5

def auc(y_true, y_pred):
    # https://stackoverflow.com/questions/43263111/defining-an-auc-metric-for-keras-to-support-evaluation-of-validation-dataset
    score = tf.py_func(lambda y_true, y_pred : roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                        [y_true, y_pred],
                        'float32',
                        stateful=False,
                        name='sklearnAUC')
    return score

def f1_score_with_nan(y_true, y_pred, average='macro', sample_weight=None):
    try:
        score = f1_score(y_true, y_pred, average=average, sample_weight=sample_weight)
    except:
        score = np.nan
    return score

def f1(y_true, y_pred):
    # https://stackoverflow.com/questions/43263111/defining-an-auc-metric-for-keras-to-support-evaluation-of-validation-dataset
    y_pred = K.round(y_pred)
    score = tf.py_func(lambda y_true, y_pred : f1_score_with_nan(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                        [y_true, y_pred],
                        'float32',
                        stateful=False,
                        name='sklearnF1')
    return score

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    # return 1 - K.square(r)
    return K.square(r)

# TODO
# https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
# def auc(y_true, y_pred):
#     return NotImplementedError()

###############################################################################################################################
# helper

def np_binary_accuracy(y_true, y_pred):
    return skmetrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

def np_precision(y_true, y_pred):
    return skmetrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    # return (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum((1-y_true)*y_pred) + 1e-7)

def np_recall(y_true, y_pred):
    return skmetrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    # return (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum(y_true*(1-y_pred)) + 1e-7)

def np_f1_score(y_true, y_pred):
    return skmetrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)

def np_roc_auc(y_true, y_pred):
    return skmetrics.roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)

def np_confusion_matrix(y_true, y_pred):
    return skmetrics.confusion_matrix(y_true, y_pred).ravel()

def np_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = np_confusion_matrix(y_true, y_pred)
    sensitivity = tp / (tp+fn)
    return sensitivity

def np_specificity(y_true, y_pred):
    tn, fp, fn, tp = np_confusion_matrix(y_true, y_pred)
    specificity = tn / (tn+fp)
    return specificity

def np_PPV(y_true, y_pred):
    tn, fp, fn, tp = np_confusion_matrix(y_true, y_pred)
    return tp/(tp+fp)

def np_gmeasure(y_true, y_pred):
    sensitivity = np_sensitivity(y_true, y_pred)
    specificity = np_specificity(y_true, y_pred)
    return (sensitivity*specificity)**0.5
    
def metric_test(y_true, y_pred):
    return (np_sensitivity(y_true, y_pred), np_specificity(y_true, y_pred),
            np_gmeasure(y_true, y_pred), np_binary_accuracy(y_true, y_pred),
            np_roc_auc(y_true, y_pred))

def metric_texa_test(y_true, y_pred):
    return (np_sensitivity(y_true, y_pred), np_specificity(y_true, y_pred),
            np_gmeasure(y_true, y_pred), np_binary_accuracy(y_true, y_pred))
###############################################################################################################################
        
# if __name__ == "__main__":
#     test_metrics = {'Accuracy':binary_accuracy, 'Precision':precision, 'Recall':recall}
    
#     print('Test loss functions %s' % test_metrics.keys())
#     y_true_set = np.array([[[0,0,0,0,0],
#                             [0,0,0,0,0],
#                             [0,1,1,0,0],
#                             [1,1,1,0,0],
#                             [0,1,0,0,0]]])
#     y_pred_set = np.array([[[0,0,0,0,1],
#                             [0,0,0,0,0],
#                             [0,1,0.6,0,0],
#                             [0,1,1,0,0],
#                             [0,0.3,0,0,0]]])
    
#     def test(acc, y_true_set, y_pred_set):
#         sess = tf.Session()
#         K.set_session(sess)
#         with sess.as_default():
#             return acc.eval(feed_dict={y_true: y_true_set, y_pred: y_pred_set})
    
#     # tf
#     y_true = tf.placeholder("float32", shape=(None,y_true_set.shape[1],y_true_set.shape[2])) 
#     y_pred = tf.placeholder("float32", shape=(None,y_pred_set.shape[1],y_pred_set.shape[2]))
#     metric_list = [binary_accuracy(y_true, y_pred), 
#                    precision(y_true, y_pred),
#                    recall(y_true, y_pred)]

#     # numpy
#     print('%15s %15s %15s' % tuple(test_metrics.keys()))
#     print('tf : {}'.format([test(acc, y_true_set, y_pred_set) for acc in metric_list]))
#     print('np : {}'.format(np.round(metric_test(y_true_set[0],y_pred_set[0]),8)))
