.. highlight:: shell

=============
Introduction
=============


DeepBiome: a phylogenetic tree regularized deep neural network for microbiome data analysis
----------------------------------------------------------------------------------------------------------------

DeepBiome uncovers the microbiome-phenotype association network and visualizes its path to disease. DeepBiome takes microbiome abundance data as input and uses the phylogenetic taxonomy to guide the decision of the optimal number of layers and neurons in the deep learning architecture. 
A phylogeny regularized weight decay technique regularize the DeepBiome model to avoid overfitting. 


Microbiome data structure
------------------------------------

.. figure:: ./figures/tree.pdf
    :align: center
    :alt: microbiome data structure

A phylogenetic tree depicts the evolutionary relationship among microbes. 
The figure displays an example phylogenetic tree.
A tip node on the phylogenetic tree represents groups of descendent taxa and each internal node is a taxonomic unit representing a common ancestor of those descendent.

DeepBiome
----------------------------------------------

DeepBiome architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. figure:: ./figures/DeepBiome.pdf
    :align: center
    :alt: the deepbiome architecture

The figure illustrates a DeepBiome architecture.
DeepBiome prespecifies the network architecture according to the phylogenetic tree.
Each hidden layer represents one phylogenetic level (i.e., family, order, class, and phylum).
The number of taxonomic levels decides the number of hidden layers. The number of taxa at each taxonomic level decides the number of neurons in the corresponding layer. 


Phylogeny regularized weight decay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We constructed a weight decay matrix to regularize weights in the neural network using evolutionary relationship carried by the phylogenetic tree.


Selection of disease associated microbiome taxa 
------------------------------------------------------

.. figure:: ./figures/tree_fev1_deepbiome2.pdf
    :align: center
    :alt: selection of disease associated microbiome taxa 
    
DeepBiome analyzes the whole microbiome profile and its path to disease and identifies taxa associated with outcome at each taxonomic level.    
The red nodes indicate taxa have positive coefficient estimation and blue nodes indicate taxa have negative estimation.


