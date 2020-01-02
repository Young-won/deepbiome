.. highlight:: shell

=====
Usage
=====

Preparing the dataset
===================================

We assume that you have the dataset 

See :ref:`Prerequisites:Data preprocessing`.


Preparing the configufation
===================================

To provide configurations for the training process and path information, we can use either dictionaries as input to functions or configuration files.

Using the configuration dictionary
----------------------------------------

You can use the network configuration dictionary `network_info` and the path configuration dictionary `path_info` for configuration.

See :ref:`Prerequisites:Configuration`.



Using the configuration file (.cfg)
----------------------------------------

You can use the configuration format files `.cfg` for configuration (for example, `path_info.cfg` and `network_info.cfg`).

See :ref:`Prerequisites:Configuration`.

If then, you can get the dictionaries from the config files by:

.. code-block:: python

    # for python 3.x
    from deepbiome import configuration
    
    config_data = configuration.Configurator('./path_info.cfg', log)
    config_data.set_config_map(config_data.get_section_map())
    config_data.print_config_map()
    path_info = config_data.get_config_map()

    config_network = configuration.Configurator('./network_info.cfg', log)
    config_network.set_config_map(config_network.get_section_map())
    config_network.print_config_map()
    network_info = config_network.get_config_map()


Logging
--------------------------------

For logging, we can use the logging instance. For example:

.. code-block:: python

    # for python 3.x
    import logging
    
    logging.basicConfig(format = '[%(name)-8s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s',
                    level=logging.DEBUG)
                    
    log = logging.getLogger()

We can use the logging instance `log` as an input of the main training fuction below.

For more information about `logging` module, please check the documentation logging_.

Training
===================================

To use DeepBiome:

.. code-block:: python

    # for python 3.x
    from deepbiome import deepbiome
    
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info)


If you want to train the network with specific number `k` of cross-validation, you can set the `number_of_fold`. For example, if you want to run the 5-fold cross-validation:

.. code-block:: python

    # for python 3.x
    from deepbiome import deepbiome
    
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=5)

If you use one input file, it will run the 5-fold cross validation using all data in that file. If you use the list of the input files, it will run the training by the first `k` files.


By defaults, `number_of_fold=None`. Then, the code will run for the leave-one-out-cross-validation (LOOCV).

If you use one input file, it will run the LOOCV using all data in that file. If you use the list of the input files, it will repeat the training for every file.


Testing
===================================

If you want to test the pre-trained model, you can use the `deepbiome.deepbiome_test` function. 

.. code-block:: python

    # for python 3.x
    from deepbiome import deepbiome
    
    evaluation = deepbiome.deepbiome_test(log, network_info, path_info, number_of_fold=None)

If you use the index file, this function provides the evaluation using test index (index set not included in the index file) for each fold. If not, this function provides the evaluation using the whole sample. 
If `number_of_fold` is setted as `k`, the function will test the model only with first `k` folds.

This function provides the evaluation result as a numpy array with a shape of (number of folds, number of evaluation measures).


Prediction
===================================

If you want to predict the output using the pre-trained model, you can use the `deepbiome.deepbiome_prediction` function. 

.. code-block:: python

    # for python 3.x
    from deepbiome import deepbiome
    
    evaluation = deepbiome.deepbiome_prediction(log, network_info, path_info, num_classes = 1, number_of_fold=None, change_weight_for_each_fold=False)

If `number_of_fold` is set as `k`, the function will predict the output of the first `k` folds' samples.

If `change_weight_for_each_fold` is set as `False`, the function will predict the output of every repeatition by same weight from the given path.
If `change_weight_for_each_fold` is set as `True`, the function will predict the output of by each fold weight.

If 'get_y=True', the function will provide a list of tuples (prediction, true output) as a numpy array output with the shape of `(n_samples, 2, n_classes)`. If 'get_y=False', the function will provide a numpy array of predictions only. The numpy array output will have the shape of `(n_samples, n_classes)`.


Cheatsheet for running the project on console
=============================================
1. Preprocessing the data (convert raw data to the format readable for python): (See :ref:`Prerequisites:Data preprocessing`.)
    Example: TODO (Julia)
    
2. Set configuration file about training hyperparameter and path information:
    1. Set the training hyper-parameter (`network_info.cfg`): (See :ref:`Prerequisites:Configuration`.)
        Example: https://github.com/Young-won/deepbiome/tree/master/examples/simulation_s0/simulation_s0_deepbiome//config/network_info.cfg
        
    2. Set the path information (`path_info.cfg`): (See :ref:`Prerequisites:Configuration`.)
        Example: https://github.com/Young-won/deepbiome/tree/master/examples/simulation_s0/simulation_s0_deepbiome/config/path_info.cfg
        
3. Write the python script for running the function `deepbiome.deepbiome_train`. For example:
    Example: https://github.com/Young-won/deepbiome/tree/master/examples/main.py
    
    Example of the python script:
    
    .. code-block:: python

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
        test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info)
    

5. Check the available GPU (if you have no GPU, it will run on CPU):
    
    .. code-block:: console

        nvidia-smi
            
6. Select the number of GPUs and CPU cores from the bash file:
    Example: https://github.com/Young-won/deepbiome/tree/master/examples/simulation_s0/simulation_s0_deepbiome/run.sh
    
    Example of the bash file (.sh):

    .. code-block:: bash

        export CUDA_VISIBLE_DEVICES=2
        echo $CUDA_VISIBLE_DEVICES

        model=${PWD##*/}
        echo $model

        python3 ../../main.py --log_info=config/log_info.yaml --path_info=config/path_info.cfg --network_info=config/network_info.cfg  --max_queue_size=50 --workers=10 --use_multiprocessing=False

        
7. Run the bash file!

    .. code-block:: console

        ./run.sh


Summary
===================================

To use deepbiome in a project::

    from deepbiome import deepbiome

.. autofunction:: deepbiome.deepbiome.deepbiome_train

.. autofunction:: deepbiome.deepbiome.deepbiome_test

.. autofunction:: deepbiome.deepbiome.deepbiome_prediction

.. autofunction:: deepbiome.deepbiome.deepbiome_get_trained_weight

.. autofunction:: deepbiome.deepbiome.deepbiome_taxa_selection_performance

.. autofunction:: deepbiome.deepbiome.deepbiome_draw_phylogenetic_tree

.. autosummary::
   :toctree: generated/


.. _logging: https://docs.python.org/3/library/logging.html