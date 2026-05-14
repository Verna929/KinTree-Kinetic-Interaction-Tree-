# KinTree-Kinetic-Interaction-Tree-
KinTree is a decision-tree-like kineti model that can automatically identify kinetically essnetial interactions from protein simulation data. The repository provides all-in-one guidance for the usage of the model, including installation, inputs generation, preprocessing, model training, model validation, and visualization. For detailed introduction of the methodology and citation, please refer to \
{M. Xu et al., XXXXX}
<img width="1963" height="1033" alt="图片1" src="https://github.com/user-attachments/assets/03f3d521-5253-4426-8efa-3671e4cb5c4f" />

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
KinTree requires contact map and interaction information. Six types of interactions are calculated: hydrogen bonds, salt bridges, $\pi$ - $\pi$ stacking, T-shape interactions, cation - $\pi$ interactions, and chalcogen interactions. The calculation of the interactions depends on defined atom masks. We have provided atom masks suitable for common MD engines, but for ligands or DNA, user has to define their own atom masks. We have provides various of functions to validate and update atom masks. Please see the example notebooks for more detailed information.

## Preprocessing
As we binarize the interaction information using strcit distance or angle thresholds. A tiny alteration in distance or angle would cuase the data fluctuate between 0 and 1. This is harmful for the extraction of protein slow dynamics. Thus we meed to do two things before passing the data to KinTree. 1. A temporal smoothing 2. A sanity check to check whether we have introduced to much bias to the smoothed data. \
Built in function has been provided to advise a propriate temporal smoothing thresholds. 
```bash
min_on, min_off, on_lengths, off_lengths = suggest_global_hysteresis_thresholds(
    X, traj_lengths=traj_lengths, on_percentile=95, off_percentile=95
)
```
After hysteresis, some interaction traces become cleaner and more stable, which is useful for kinetic modeling. However, excessive smoothing can also distort the original signal by inflating rare interactions, suppressing real transient events, or reshaping long-lived ON/OFF periods. The filtering logic therefore evaluates each feature using both occupancy-based and run-length-based criteria. The parameters below define what counts as a short fluctuation, how much modification is acceptable, and when a feature should be trusted, partially restored, or fully reverted to its raw form.
```bash
short_on_thr            = 2     # Max ON-run for short fluctuation
short_off_thr           = 4     # Max OFF-run for short fluctuation
pct_modified_thr        = 0.15  # Min modification fraction
min_raw_support         = 0.02  # Min raw occupancy for support
max_occupancy_shift     = 0.15  # Max allowed shift in global occupancy
safe_occupancy_shift    = 0.05  # Shift threshold for common interactions
safe_helpful_ratio      = 0.70  # Min ratio of helpful cleanup edits
max_long_run_damage     = 0.20  # Max edits affecting long raw runs
common_feature_raw_frac = 0.10  # Min occupancy to be "common"
```
You may find difficult at first in alterting these parameters, it is enough to start with the default parameters as showing above.

## Model Training
To train the KinTree, you need to provide 1. Contact Map (filtering recommended). 2. Intreaction information. Parameters for KinTree is shown as below.
```bash
lags                  = [5, 10, 15]  # Lag times used to build transition pairs
alpha_vamp            = 0.50         # Weight for VAMP-2 relative to NMI
min_pairs             = 3000         # Min valid lagged transition pairs required to split a node
min_leaf              = 3000         # Min frames required in each child leaf
max_depth             = 5            # Max tree depth
max_features_to_try   = None         # Number of candidate features tried per split (None = all)
feature_types         = feature_types # Feature-type labels for optional type weighting
random_state          = seed         # RNG seed

lag_mode              = "uniform"    # How multiple lags are combined: uniform / max / power
tau_power             = 1.0          # Exponent for lag weighting when lag_mode="power"
lambda_row            = 0.70         # Penalty for kinetically similar child transition behavior

contact_frames        = filtered_contacts # Soft contact map frames for contact-change scoring
local_window          = 2            # Exclude contacts near the split residues
delta_mode            = "quantile"   # How contact-change magnitude is binarized: quantile / threshold
delta_quantile        = 0.60         # Quantile cutoff when delta_mode="quantile"
#delta_thresh         = 0.20         # Absolute cutoff when delta_mode="threshold"
```

## Model Validation
If a you want to test Markovianity, our model incorporate IGME (Integrative Generalized Master Equation) to study the kinetics. For detailed guidance of this part, please refer to {https://doi.org/10.1063/5.0189429} and {https://github.com/xuhuihuang/GME_tutorials}

## Visualization
We have built in tools to visualize transtion flux, tree. We provides built in function to create a pml file that ready to be input in to Pymol. Moreover, we provide a script to create an interactive HTML interface so that user can visualize the tree and the protein structure at the same time. To use this script
```bash
python viewer.py --tree_json tree.json --pdb pdb_file --macrostate ./data/macrostates_test.json
```
