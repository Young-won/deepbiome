=========
Deepbiome
=========

.. image:: https://img.shields.io/travis/Young-won/deepbiome.svg
        :target: https://travis-ci.org/Young-won/deepbiome
        :alt: Build
        
.. image:: https://coveralls.io/repos/github/Young-won/deepbiome/badge.svg?branch=master
        :target: https://coveralls.io/github/Young-won/deepbiome?branch=master
        :alt: Coverage

.. image:: https://img.shields.io/pypi/v/deepbiome.svg
        :target: https://pypi.python.org/pypi/deepbiome
        :alt: Version
 
Deep Learning package using the phylogenetic tree information for microbiome data analysis.

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) https://Young-won.github.io/deepbiome.

Installation
--------

Prerequisites
^^^^^^^^^^^^^^^^
* python >= 3.5
* Tensorflow
* Keras

Install Deepbiome
^^^^^^^^^^^^^^^^

At the command line:

.. code-block:: bash

    # for python 3.x
    
    $ pip3 install git+https://github.com/Young-won/deepbiome.git

Features
--------

* deepbiome.deepbiome_train(log, network_info, path_info, number_of_fold=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    Function for training the deep neural network with phylogenetic tree weight regularizer.
    
    It uses microbiome abundance data as input and uses the phylogenetic taxonomy to guide the decision of the optimal number of layers and neurons in the deep learning architecture.

    See ref url (TODO: update)

Credits
--------
This package was created with Cookiecutter_ and the `NSLS-II/scientific-python-cookiecutter`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`NSLS-II/scientific-python-cookiecutter`: https://github.com/NSLS-II/scientific-python-cookiecutter
