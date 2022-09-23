# Graph Learning-Convolutional Networks

This is a TensorFlow implementation of Re-weight Nodes and Graph Learning Convolutional Network with Mani-fold Regularization (ReNode-GLCNMR) for the task of (semi-supervised) classification of nodes in a graph, as described in our paper:
 
Fadi Dornaikaa, Jingjun Bi, Chongsheng Zhang, [A Unified Deep Semi-supervised Graph Learning Scheme Based on Nodes
Re-weighting and Manifold Regularization]


## Introduction

In this repo, we provide ReNode-GLCNMR's code with the Scene15 datasets as example. The GLCN method used in this code is provided by Bo Jiang, Ziyan Zhang, [Semi-supervised Learning with Graph Learning-Convolutional Networks](http://http://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf) (CVPR 2019)


## Requirements
The codebase is implemented in Python 3.6.8. package versions used for development are just below
* tensorflow-gpu (1.13.1)
* scipy (1.4.1)
* numpy (1.16.2)

## Run the demo

```bash
cd glcn
python run_scence.py
```

## Data

There are seven entries for the code.
* Feature matrix (feat.mat): An n * p sparse matrix, where n represents the number of nodes, and p represents the feature dimension of each node.
* Adjacency matrix (adj.mat): An n * n sparse matrix, where n represents the number of nodes.
* Label matrix (label.mat): An n * c matrix, where n represents the number of nodes, c represents the number of classes, and the label of the node is represented by onehot.
* ReNode_weigth matrix (scence1rnp15w5s30.mat): An 1 * n matrix, where n represents the number of nodes. This matrix come from [ReNode](http://http:)
* Train index matrix (scence1reid.mat): An 1 * n matrix, where n represents the number of nodes.
* Validation index matrix (scence1vaid.mat): An 1 * n matrix, where n represents the number of nodes.
* Test index matrix (scence1teid.mat): An 1 * n matrix, where n represents the number of nodes.

We provide the Scene15 datasets as example. In our paper, we perform experiments on eight benchmark datasets, including three widely used Plantoid Paper Citation Graphs (Citeseer, Cora, and Pubmed), a co-authorship graph (Coauthor CS) based on the Microsoft Academic Graph, and four image datasets (CIFAR10 , SVHN, MNIST, and Scene15) (see `data` folder). 


## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{jiang2019semi,
  title={A Unified Deep Semi-supervised Graph Learning Scheme Based on Nodes Re-weighting and Manifold Regularization},
  author={Fadi Dornaikaa, Jingjun Bi, Chongsheng Zhang},
  booktitle={},
  pages={},
  year={}
}
```
