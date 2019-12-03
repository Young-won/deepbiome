######################################################################
## Medical segmentation using CNN
## - Miscellaneous
##
## Nov 16. 2018
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################

import os
import platform
import timeit
import glob
import json
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import keras.backend as k
import tensorflow as tf

from . import loss_and_metric

import matplotlib
import matplotlib.cm

def argv_parse(argvs):
    arglist = [arg.strip() for args in argvs[1:] for arg in args.split('=') if not arg.strip()=='']
    arglist.reverse()
    argdict = dict()
    argname = arglist.pop()
    while len(arglist) > 0:
        if '--' not in argname:
            raise Exception('python argument error')
        argv = []
        while len(arglist) > 0:
            arg = arglist.pop()
            if '--' in arg:
                argdict[argname.split('--')[-1]] = argv
                argname = arg
                break
            argv.append(arg)
    argdict[argname.split('--')[-1]] = argv
    return argdict

def file_path_fold(path, fold):
    path = path.split('.')
    return '.'+''.join(path[:-1])+'_'+str(fold)+'.'+path[-1]

# def convert_bytes(num, x='MB'):
#     for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
#         if num < 1024.0:
#             return "%3.1f %s" % (num, x)
#         num /= 1024.0

# def file_size(file_path, scale='MB'):
#     if os.path.isfile(file_path):
#         file_info = os.stat(file_path)
#         return convert_bytes(file_info.st_size, scale)
    
# def print_sysinfo():
#     print('\nPython version  : {}'.format(platform.python_version()))
#     print('compiler        : {}'.format(platform.python_compiler()))
#     print('\nsystem     : {}'.format(platform.system()))
#     print('release    : {}'.format(platform.release()))
#     print('machine    : {}'.format(platform.machine()))
#     print('processor  : {}'.format(platform.processor()))
#     print('CPU count  : {}'.format(mp.cpu_count()))
#     print('interpreter: {}'.format(platform.architecture()[0]))
#     print('\n\n')

def metric_taxa_test(y_true, y_pred, taxa_metric=['sensitivity','specificity','gmeasure','accuracy']):
    test_ftn_dict = {'sensitivity':loss_and_metric.np_sensitivity,
                     'specificity':loss_and_metric.np_specificity,
                     'gmeasure':loss_and_metric.np_gmeasure,
                     'accuracy':loss_and_metric.np_binary_accuracy}
    
    y_true = y_true.astype(np.int32)
    y_pred = (y_pred>=0.5).astype(np.int32)
    
    results = [test_ftn_dict[ky](y_true, y_pred) for ky in taxa_metric]
    return results

# def taxa_selection_accuracy(tree_weight_list, true_tree_weight_list, taxa_metric=['sensitivity','specificity','gmeasure','accuracy']):
#     accuracy_list = []
#     for i in range(len(true_tree_weight_list)):
#         tree_tw = true_tree_weight_list[i].astype(np.int32)
#         tree_w = np.zeros_like(tree_tw, dtype=np.int32)
#         tree_w_abs = np.abs(tree_weight_list[i])
#         for row, maxcol in enumerate(np.argmax(tree_w_abs, axis=1)):
#             tree_w[row,maxcol] = tree_w_abs[row,maxcol]
# #         tree_w = (tree_w > 1e-2).astype(np.int32)
#         tree_w = (tree_w > 0).astype(np.int32)
#         num_selected_texa = np.sum(np.sum(tree_w, axis=1)>0)
#         taxa_results = metric_texa_test(tree_tw.flatten(), tree_w.flatten(), taxa_metric)
#         accuracy_list.append([num_selected_texa] + taxa_results)
#     return accuracy_list

def taxa_selection_accuracy(tree_weight_list, true_tree_weight_list, taxa_metric=['sensitivity','specificity','gmeasure','accuracy']):
    accuracy_list = []
    for i in range(len(true_tree_weight_list)):
        tree_tw = true_tree_weight_list[i].astype(np.int32)
        tree_w = np.zeros_like(tree_tw, dtype=np.int32)
        tree_w_abs = np.abs(np.array(tree_weight_list[i]))
#         tree_w = (tree_w_abs>1e-2).astype(np.int32)
        for row in range(tree_w_abs.shape[0]):
#             tree_w[row,:] = (tree_w_abs[row,:]> 0).astype(np.int32)
            tree_w[row,:] = (tree_w_abs[row,:]> 1e-2).astype(np.int32)
        num_selected_texa = np.sum(np.sum(tree_w, axis=1)>0)
        taxa_results = metric_taxa_test(tree_tw.flatten(), tree_w.flatten(), taxa_metric)
        accuracy_list.append([num_selected_texa] + taxa_results)
    return accuracy_list
    

# def set_ax_plot(ax, models=['base_model'], models_aka=['base_model'], y_name = 'val_dice', x_name = 'epochs', mode='summary', niters=None):
#     x_title = x_name.title()
#     if 'val' in y_name:
#         y_title = y_name.split('_')[-1].title()
#         title = 'Validation History - %s (per %s)' % (y_title, x_title)
#     else:
#         y_title = y_name.title()
#         title = 'History - %s (per %s)' % (y_title, x_title)
        
#     ax.set_title(title, fontdict={'fontsize':15})
#     for col, model in enumerate(models):
#         aka = models_aka[col]
#         hist_path = './%s/history/hist.json' % (model)
#         kfold = len(glob.glob(file_path_fold(hist_path,'*')))
#         with open(file_path_fold(hist_path,kfold-1), 'r') as f:
#             history = json.load(f)
#         max_epoch = np.max([len(hist['loss']) for hist in history])
        
#         if not niters == None: niter = niters[col]
#         else: niter = None
#         if 'epoch' in x_name: index = np.arange(1, max_epoch+1)
#         elif 'iter' in x_name: index = np.arange(1, max_epoch*niter+1, niter)
#         else: raise ValueError
            
#         value = np.zeros((len(history),max_epoch))
#         for i, hist in enumerate(history):
#             value[i,:len(hist[y_name])] = hist[y_name]
            
#         if mode == 'summary':
#             ax.plot(index, np.mean(value, axis=0), 'C%s.-' % (col+1), label='%s-%s'% (aka, y_title))
#             ax.fill_between(index, np.mean(value, axis=0)-np.std(value, axis=0), np.mean(value, axis=0)+np.std(value, axis=0),
#                             color='C%s' % (col+1), alpha=0.2)
#             if 'accuracy' in y_name : # total epoch max per each fold
#                 ax.plot(index[np.argmax(value, axis=1)], np.max(value, axis=1), 'C%s*' % (col+1), markersize=12)
#                 for j, (x,y) in enumerate(zip(index[np.argmax(value, axis=1)], np.max(value, axis=1))):
#                     ax.annotate(j+1, (x,y))
#             if 'loss' in y_name : # total epoch min per each fold
#                 ax.plot(index[np.argmax(value, axis=1)], np.min(value, axis=1), 'C%s*' % (col+1), markersize=12)
#                 for j, (x,y) in enumerate(zip(index[np.argmin(value, axis=1)], np.min(value, axis=1))):
#                     ax.annotate(j+1, (x,y)) 
#             # ax.plot(index, np.max(value, axis=0), 'C%s.' % (col+1), alpha=0.3) # total fold max per epoch
#             # ax.plot(index, np.min(value, axis=0), 'C%s.' % (col+1), alpha=0.3) # total fold min per epoch
#         elif mode == 'all':
#             for j, v in enumerate(value):
#                 if j == 0: ax.plot(index, v, 'C%s.-' % (col+1), label='%s-%s'% (aka, y_title), alpha=0.4)
#                 else: ax.plot(index, v, 'C%s.-' % (col+1), alpha=0.4)
            
#     ax.set_xlabel(x_title)
#     ax.set_ylabel(y_title)
#     ax.legend(loc='lower right', fontsize='small')

# def plot_history(models, models_aka,
#                  history_types = ['validation','train'], y_names = ['dice','precision','recall'], x_name='epochs',
#                  mode = 'summary', figsize=(20,20), niters=None):
#     fig, axes = plt.subplots(len(y_names), len(history_types), figsize=figsize)
#     if len(y_names) == 1: axes = np.expand_dims(axes, 0)
#     if len(history_types) == 1: axes = np.expand_dims(axes, -1)
#     for j in range(len(history_types)):
#         for i, y_name in enumerate(y_names):
#             if 'val' in history_types[j]: set_ax_plot(axes[i,j], models, models_aka, 'val_%s' % y_name, x_name, mode, niters)
#             else: set_ax_plot(axes[i,j], models, models_aka, y_name, x_name, mode, niters)
#     fig.tight_layout()
#     return fig

# class TensorBoardWrapper(TensorBoard):
#     '''
#     Sets the self.validation_data property for use with TensorBoard callback.
    
#     Image Summary with multi-modal medical 3D volumes:  
#         Thumbnail of nrow x ncol 2D images (of one person) 
#             nrow: number of slice (z-axis)
#             ncol: 
#                    input images: number of modals
#                    bottleneck images : number of filters
#                    output images: 2 (GT, predict)
#         TODO: fix one person as reference..
#     '''

#     def __init__(self, validation_data, write_weights_histogram = True, write_weights_images=False, 
#                  tb_data_steps=1,  **kwargs):
#         super(TensorBoardWrapper, self).__init__(**kwargs)
#         self.write_weights_histogram = write_weights_histogram
#         self.write_weights_images = write_weights_images
#         self.tb_data_steps = tb_data_steps
#         self.validation_data = validation_data
        
#         if self.embeddings_data is None and self.validation_data:
#             self.embeddings_data = self.validation_data
    
#     def set_model(self, model):
#         self.model = model
#         if k.backend() == 'tensorflow':
#             self.sess = k.get_session()
            
#         if self.histogram_freq and self.merged is None:
#             for layer in self.model.layers:
#                 for weight in layer.weights:
#                     mapped_weight_name = 'weight_%s' % weight.name.replace(':', '_')
#                     # histogram
#                     if self.write_weights_histogram: tf.summary.histogram(mapped_weight_name, weight)
#                     # gradient histogram
#                     if self.write_grads:
#                         grads = model.optimizer.get_gradients(model.total_loss,
#                                                               weight)

#                         def is_indexed_slices(grad):
#                             return type(grad).__name__ == 'IndexedSlices'
#                         grads = [
#                             grad.values if is_indexed_slices(grad) else grad
#                             for grad in grads]
#                         tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                        
#                     if self.write_weights_images:
#                         w_img = tf.squeeze(weight)
#                         shape = k.int_shape(w_img)
#                         if len(shape) == 2:  # dense layer kernel case
#                             if shape[0] > shape[1]:
#                                 w_img = tf.transpose(w_img)
#                                 shape = k.int_shape(w_img)
#                             w_img = tf.reshape(w_img, [1,
#                                                        shape[0],
#                                                        shape[1],
#                                                        1])
#                         elif len(shape) == 3:  # 1d convnet case
#                             if k.image_data_format() == 'channels_last':
#                                 # switch to channels_first to display
#                                 # every kernel as a separate image
#                                 w_img = tf.transpose(w_img, perm=[2, 0, 1])
#                                 shape = k.int_shape(w_img)
#                             w_img = tf.reshape(w_img, [shape[0],
#                                                        shape[1],
#                                                        shape[2],
#                                                        1])
#                         elif len(shape) == 4: # conv2D
#                             # input_dim * output_dim, width, hieght
#                             w_img = tf.transpose(w_img, perm=[2, 3, 0, 1])
#                             shape = k.int_shape(w_img)
#                             w_img = tf.reshape(w_img, [shape[0]*shape[1],
#                                                        shape[2],
#                                                        shape[3],
#                                                        1])
#                         elif len(shape) == 5: # conv3D
#                             # input_dim * output_dim*depth, width, hieght
#                             w_img = tf.transpose(w_img, perm=[3, 4, 0, 1, 2])
#                             shape = k.int_shape(w_img)
#                             w_img = tf.reshape(w_img, [shape[0]*shape[1]*shape[2],
#                                                        shape[3],
#                                                        shape[4],
#                                                        1])
#                         elif len(shape) == 1:  # bias case
#                             w_img = tf.reshape(w_img, [1,
#                                                        shape[0],
#                                                        1,
#                                                        1])
#                         tf.summary.image(mapped_weight_name, w_img)

#                 if hasattr(layer, 'output'):
#                     if isinstance(layer.output, list):
#                         for i, output in enumerate(layer.output):
#                             tf.summary.histogram('{}_out_{}'.format(layer.name, i), output)
#                     else:
#                         tf.summary.histogram('{}_out'.format(layer.name),
#                                              layer.output)
#             #################################################################################
#             # image summary
# #             if self.write_images:
# #                 input_shape = []
# #                 input_shape[:] = self.img_shape[:]
# #                 input_shape[-1] += 2
                
# #                 tot_pred_image = []
# #                 for i in range(self.batch_size):
# #                     # input images, GT, prediction
# #                     mri = model.inputs[0][i]
# #                     gt = model.targets[0][i]
# #                     pred = model.outputs[0][i]
# #                     pred_image = self.tile_patches_medical(k.concatenate([mri, gt, pred], axis=-1),
# #                                                            shape=input_shape, zcut=[10,6]) # output : [1,x*nrow, y*ncol, 1]
# #                     pred_image = colorize(pred_image, cmap='inferno') # output : [x*nrow, y*ncol, 3]
# #                     tot_pred_image.append(pred_image)
# #                 tot_pred_image = k.stack(tot_pred_image) # output : [batch, x*nrow, y*ncol, 3]
# #                 shape = k.int_shape(tot_pred_image)
# #                 assert len(shape) == 4 and shape[-1] in [1, 3, 4]
# #                 tf.summary.image('prediction', tot_pred_image, max_outputs=self.batch_size)
            
#             #################################################################################
            
#         self.merged = tf.summary.merge_all()
#         #################################################################################
#         # tensor graph & file write
#         if self.write_graph:
#             self.writer = tf.summary.FileWriter(self.log_dir,
#                                                 self.sess.graph)
#         else:
#             self.writer = tf.summary.FileWriter(self.log_dir)

#         #################################################################################
#         # embedding : TODO
#         if self.embeddings_freq:
#             embeddings_layer_names = self.embeddings_layer_names

#             if not embeddings_layer_names:
#                 embeddings_layer_names = [layer.name for layer in self.model.layers
#                                           if type(layer).__name__ == 'Embedding']
#             self.assign_embeddings = []
#             embeddings_vars = {}

#             self.batch_id = batch_id = tf.placeholder(tf.int32)
#             self.step = step = tf.placeholder(tf.int32)

#             for layer in self.model.layers:
#                 if layer.name in embeddings_layer_names:
#                     embedding_input = self.model.get_layer(layer.name).output
#                     embedding_size = int(np.prod(embedding_input.shape[1:]))
#                     embedding_input = tf.reshape(embedding_input,
#                                                  (step, embedding_size))
#                     shape = (self.embeddings_data[0].shape[0], embedding_size)
#                     embedding = tf.Variable(tf.zeros(shape),
#                                             name=layer.name + '_embedding')
#                     embeddings_vars[layer.name] = embedding
#                     batch = tf.assign(embedding[batch_id:batch_id + step],
#                                       embedding_input)
#                     self.assign_embeddings.append(batch)

#             self.saver = tf.train.Saver(list(embeddings_vars.values()))

#             embeddings_metadata = {}

#             if not isinstance(self.embeddings_metadata, str):
#                 embeddings_metadata = self.embeddings_metadata
#             else:
#                 embeddings_metadata = {layer_name: self.embeddings_metadata
#                                        for layer_name in embeddings_vars.keys()}

#             config = projector.ProjectorConfig()

#             for layer_name, tensor in embeddings_vars.items():
#                 embedding = config.embeddings.add()
#                 embedding.tensor_name = tensor.name

#                 if layer_name in embeddings_metadata:
#                     embedding.metadata_path = embeddings_metadata[layer_name]

#             projector.visualize_embeddings(self.writer, config)
    
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}

#         if not self.validation_data and self.histogram_freq:
#             raise ValueError("If printing histograms, validation_data must be "
#                              "provided, and cannot be a generator.")
#         if self.embeddings_data is None and self.embeddings_freq:
#             raise ValueError("To visualize embeddings, embeddings_data must "
#                              "be provided.")
            
#         if self.validation_data and self.histogram_freq:
#             if epoch % self.histogram_freq == 0: # TODO : last epoch..

#                 val_data = self.validation_data
#                 if self.batch_size == None:
#                     self.batch_size = val_data[0].shape[0]
                    
#                 tensors = (self.model.inputs +
#                            self.model.targets +
#                            self.model.sample_weights)

#                 if self.model.uses_learning_phase:
#                     tensors += [k.learning_phase()]

#                 try:
#                     for i in range(self.tb_data_steps):
#                         x, y = val_data[i]
#                         if self.model.uses_learning_phase:
#                             batch_val = np.array([x, y, np.ones(self.batch_size, dtype=np.float32), 0.0])
#                         else:
#                             batch_val = np.array([x, y, np.ones(self.batch_size, dtype=np.float32)])

#                         assert len(batch_val) == len(tensors)
#                         feed_dict = dict(zip(tensors, batch_val))
#                         result = self.sess.run([self.merged], feed_dict=feed_dict)
#                         summary_str = result[0]
#                         self.writer.add_summary(summary_str, epoch)
#                 except:
#                     val_size = val_data[0].shape[0]
#                     i = 0
#                     while i < val_size:
#                         step = min(self.batch_size, val_size - i)
#                         if self.model.uses_learning_phase:
#                             # do not slice the learning phase
#                             batch_val = [x[i:i + step] for x in val_data[:-1]]
#                             batch_val.append(val_data[-1])
#                         else:
#                             batch_val = [x[i:i + step] for x in val_data]
#                         assert len(batch_val) == len(tensors)
#                         feed_dict = dict(zip(tensors, batch_val))
#                         result = self.sess.run([self.merged], feed_dict=feed_dict)
#                         summary_str = result[0]
#                         self.writer.add_summary(summary_str, epoch)
#                         i += self.batch_size

#         if self.embeddings_freq and self.embeddings_data is not None:
#             if epoch % self.embeddings_freq == 0: ## TODO : Last epoch..
#                 embeddings_data = self.embeddings_data
#                 for i in range(self.tb_data_steps):
#                     if type(self.model.input) == list:
#                         feed_dict = {model_input: embeddings_data[i][idx]
#                                      for idx, model_input in enumerate(self.model.input)}
#                     else:
#                         feed_dict = {self.model.input: embeddings_data[i]}

#                     feed_dict.update({self.batch_id: i, self.step: self.batch_size})

#                     if self.model.uses_learning_phase:
#                         feed_dict[k.learning_phase()] = False

#                     self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
#                     self.saver.save(self.sess,
#                                     os.path.join(self.log_dir, 'keras_embedding.ckpt'),
#                                     epoch)
                    
#         for name, value in logs.items():
#             if name in ['batch', 'size']:
#                 continue
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_value.tag = name
#             self.writer.add_summary(summary, epoch)
#         self.writer.flush()