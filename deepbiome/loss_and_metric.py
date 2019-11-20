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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

###############################################################################################################################
# tf loss functions
def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    score = tf.py_function(lambda y_true, y_pred : recall_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                           [y_true, y_pred],
                           Tout=tf.float32,
                           name='sklearnRecall')
    return score
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # recall = true_positives / (predicted_positives + K.epsilon())
    # return recall
    # return K.sum(y_true==y_pred)/(K.sum(y_true==y_pred) + K.sum(y_true==(1-y_pred)) + 1e-10)

def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    score = tf.py_function(lambda y_true, y_pred : precision_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                           [y_true, y_pred],
                           Tout=tf.float32,
                           name='sklearnPrecision')
    return score
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    # return precision
    # return K.sum(y_true==y_pred)/(K.sum(y_true==y_pred) + K.sum((1-y_true)==y_pred) + 1e-10)
    
def sensitivity(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), K.floatx())
    neg_y_pred = 1 - y_pred
    true_positive = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
    false_negative = K.round(K.sum(K.clip(y_true * neg_y_pred, 0, 1)))
    return (true_positive) / (true_positive + false_negative + K.epsilon())

def specificity(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), K.floatx())
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    false_positive = K.round(K.sum(K.clip(neg_y_true * y_pred, 0, 1)))
    true_negative = K.round(K.sum(K.clip(neg_y_true * neg_y_pred, 0, 1)))
    return (true_negative) / (false_positive + true_negative + K.epsilon())

def gmeasure(y_true, y_pred):
    return (sensitivity(y_true, y_pred) * specificity(y_true, y_pred)) ** 0.5

def auc(y_true, y_pred):
    # https://stackoverflow.com/questions/43263111/defining-an-auc-metric-for-keras-to-support-evaluation-of-validation-dataset
    score = tf.py_function(lambda y_true, y_pred : roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                           [y_true, y_pred],
                           Tout=tf.float32,
                           name='sklearnAUC')
    return score

def f1_score_with_nan(y_true, y_pred, average='macro', sample_weight=None):
    try:
        score = f1_score(y_true, y_pred, average=average, sample_weight=sample_weight)
    except:
        score = np.nan
    return score

def f1(y_true, y_pred):
    # precision = precision(y_true, y_pred)
    # recall = recall(y_true, y_pred)
    # return 2 * (precision * recall) / (precision + recall + K.epsilon())

    # https://stackoverflow.com/questions/43263111/defining-an-auc-metric-for-keras-to-support-evaluation-of-validation-dataset
    y_pred = K.round(y_pred)
    score = tf.py_function(lambda y_true, y_pred : f1_score_with_nan(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                           [y_true, y_pred],
                           Tout=tf.float32,
                           name='sklearnF1')
    return score

def ss(a, axis=0):
    # a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)

def pearsonr(x,y):
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(ss(xm) * ss(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    return r

def correlation_coefficient(y_true, y_pred):
    score = tf.py_function(lambda y_true, y_pred : pearsonr(y_true, y_pred),
                           [y_true, y_pred],
                           Tout=tf.float32,
                           name='correlation_coefficient')
    return score

# TODO
# https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
# def auc(y_true, y_pred):
#     return NotImplementedError()

###############################################################################################################################
# helper

def np_binary_accuracy(y_true, y_pred):
    y_pred = (y_pred>=0.5).astype(np.int32)
    return skmetrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

def np_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    # return (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum((1-y_true)*y_pred) + 1e-7)

def np_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    # return (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum(y_true*(1-y_pred)) + 1e-7)

def np_f1_score(y_true, y_pred):
    return skmetrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)

def np_roc_auc(y_true, y_pred):
    return skmetrics.roc_auc_score(y_true, y_pred, average='macro', sample_weight=None)

def np_confusion_matrix(y_true, y_pred):
    return skmetrics.confusion_matrix(y_true, y_pred).ravel()

def np_sensitivity(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = (y_pred >= 0.5).astype(np.int32)
    neg_y_pred = 1 - y_pred
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * neg_y_pred)
    return tp / (tp+fn)

def np_specificity(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = (y_pred >= 0.5).astype(np.int32)
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = np.sum(neg_y_true * y_pred)
    tn = np.sum(neg_y_true * neg_y_pred)
    return tn / (tn+fp)
    
def np_PPV(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = (y_pred >= 0.5).astype(np.int32)
    neg_y_true = 1 - y_true
    tp = np.sum(y_true * y_pred)
    fp = np.sum(neg_y_true * y_pred)
    return tp/(tp+fp)

def np_gmeasure(y_true, y_pred):
    sensitivity = np_sensitivity(y_true, y_pred)
    specificity = np_specificity(y_true, y_pred)
    return (sensitivity*specificity)**0.5
    
def metric_test(y_true, y_pred):
    return (np_sensitivity(y_true, y_pred), np_specificity(y_true, y_pred),
            np_gmeasure(y_true, y_pred), np_binary_accuracy(y_true, y_pred),
            np_roc_auc(y_true, y_pred))

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
