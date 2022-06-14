# CFFN  
## Installation

To use this code, PyTorch (>=1.6.0), PyTorch Geometric (>=1.6.0), and RDKit and OpenBabel are need to installed.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) (>=1.6.0)


2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) (>=1.6.0)

3. Install RDKit   
```shell script
conda install -c rdkit rdkit
``` 

4. Install OpenBabel  
```shell script
conda install -c openbabel openbabel
```


## Usage

run   
```shell script
python qm9.py
``` 
to train and test QM9 dataset. All the .mol files in folder mol are the result of OpenBabel from .xyz files. You can get them by xyz.py



