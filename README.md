# KinTree-Kinetic-Interaction-Tree-
KinTree is a decision-tree-like kineti model that can automatically identify kinetically essnetial interactions from protein simulation data. The repository provides all-in-one guidance for the usage of the model, including installation, inputs generation, preprocessing, model training, model validation, and visualization. For detailed introduction of the methodology and citation, please refer to
{M. Xu et al., XXXXX}

## Installation
We recommend using conda to install the package. 
```bash
conda env create -f KinTree.yml
conda activate KinTree
```

You can also manually intall dependencies if the ```conda create -f``` fails
```bash
conda create -n KinTree python==3.10.0
conda acivate KinTree
conda install mdtraj mdanalysis matplotlib networkx plotly pytorch numpy -c conda-forge -c pytorch
```

## Inputs Generation
KinTree requires contact map and interaction information. Six types of interactions are calculated: hydrogen bonds, salt bridges, $\pi$-$\pi$ stacking, T-shape interactions, cation-$\pi$ interactions, and chalcogen interactions. The calculation of the interactions depends on defined atom masks. We have provided atom masks suitable for common MD engines, but for ligands or DNA, user has to define their own atom masks. We have provides various of functions to validate and update atom masks. Please see the example notebooks for more detailed information.

## Preprocessing
