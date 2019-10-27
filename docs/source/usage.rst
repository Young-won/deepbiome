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

Using the configuration dictionary
--------------------------------

You can use the network configuration dictionary `network_info` and the path configuration dictionary `path_info` for configuration.

See :ref:`Prerequisites:Configuration`.



Using the configuration file (.cfg)
--------------------------------

You can use the configuration format files `.cfg` for configuration (for example, `path_info.cfg` and `network_info.cfg`).

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

For logging, you have to use the logging instance. For example:

.. code-block:: python
    # for python 3.x
    import logging
    
    logging.basicConfig(format = '[%(name)-8s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s',
                    level=logging.DEBUG)
                    
    log = logging.getLogger()

You can use the logging instance `log` as an input of the main training fuction below.


Training
===================================

To use deepbiome::

.. code-block:: python
    # for python 3.x
    from deepbiome import deepbiome
    
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info)


If you want to train the network with specific number `k`of cross-validation repeatition, you can set the `number_of_fold`. For example, if you want to run the 5-fold cross-validation:

.. code-block:: python
    # for python 3.x
    from deepbiome import deepbiome
    test_evaluation, train_evaluation, network = deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=5)

If you use one input file, it will run the 5-fold cross validation using all data in that file. If you use the list of the input files, it will run the training by the first `k` files.


By defaults, `number_of_fold==None`. Then, the code will run for the leave-one-out-cross-validation (LOOCV).

If you use one input file, it will run the LOOCV using all data in that file. If you use the list of the input files, it will repeat the training for every files.


Cheatsheet for running the project on console
=============================================
1. Preprocessing the data (convert raw data to the format readable for python):
    Example: TODO (Julia)
    
2. Set configuration file about training hyperparameter and path information:
    1. Set the training hyper-parameter (`network_info.cfg`):
        Example: example/simulation_s0_v0/simulation_s0_deepbiome/config/network_info.cfg
        
    2. Set the path information (`path_info.cfg`):
        Example: example/simulation_s0_v0/simulation_s0_deepbiome/config/path_info.cfg
        
3. Check the available GPU (if you have no GPU, it will run on CPU):
    
.. code-block:: console
    nvidia-smi
            
4. Select the number of GPUs and CPU cores from the bash file:
    Example:example/simulation_s0_deepbiome/run.sh
        
5. Run!:
    Example:
        
.. code-block:: console
    example/simulation_s0_deepbiome/run.sh


Summary
===================================

To use deepbiome in a project::

    from deepbiome import deepbiome

.. autofunction:: deepbiome.deepbiome.deepbiome_train

.. autosummary::
   :toctree: generated/
