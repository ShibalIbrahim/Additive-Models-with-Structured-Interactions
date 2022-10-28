# Additive-Models-with-Structured-Interactions

This is our implementation of Additive Models with interactions under L0 subset selection with/without strong hierarchy as described in our manuscript.

[Predicting Census Survey Response Rates via Interpretable Nonparametric Additive Models with Structured Interactions](http://arxiv.org/abs/2108.11328) by Shibal Ibrahim, Rahul Mazumder, Peter Radchenko, Emanuel Ben-David

## Installation
We provide a conda environment file named "sparse-am-with-interactions.yml" for straightforward installation with Anaconda, which can be used to setup a sparse-am-with-interactions environment with the commands:

conda env create --name sparse-am-with-interactions --file=sparse-am-with-interactions.yml

source activate sparse-am-with-interactions

Alternatively, the following packages can be downloaded to run the python scripts and jupyter notebooks.

## Requirements
* descartes                 1.1.0
* dill                      0.3.3 
* fiona                     1.8.18
* gurobi                    9.0.1 
* ipywidgets                7.5.1
* matplotlib                3.3.2 
* notebook                  6.1.5
* numpy                     1.19.4 
* pandas                    1.1.4
* patsy                     0.5.1
* pyproj                    2.6.1.post1
* python                    3.6.12 
* rtree                     0.9.4
* scikit-learn              0.23.2
* scipy                     1.5.3
* tqdm                      4.54.1
 
## Proposed Models
* `AM`: Additive Models under L0
* `AMI`: Additive Models with Interactions under L0
* `AMISH`: Additive Models with Interactions with Strong Hierarchy

## Running the code

```bash
cd SparseGAMsWithInteractions

The following 3 ipython notebooks can be used to run block cyclic coordinate descent algorithm for the three models
For `AM`: run src/AMsL0/AMsL0.ipynb [no interactions]
For `AMI`: run src/AMsWithInteractionsL0/AMsWithInteractionsL0.ipynb
For `AMISH`: run src/AMsWithInteractionsStrongHierarchy/AMsWithInteractionsStrongHierarchy.ipynb
```

## Citing Additive-Models-with-Structured-Interactions
If you find our repository useful in your research, please consider citing the following paper.

```
@misc{Ibrahim2021,
  title={Interpretable-Models-To-Identify-Low-Response-Populations-In-Census-Bureau-Surveys},
  author={Ibrahim, Shibal and Mazumder, Rahul and Radchenki, Peter and Ben-David, Emanuel},
  Eprint={arXiv:},
  year={2021}
}
```

