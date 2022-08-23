# Workflows for Noise2Void training
These are workflows used to train [Noise2Void](https://github.
com/juglab/n2v) denoising networks.

# Installation
We recommend installing the requirements into a fresh conda environment.
```shell
conda create -n n2v python=3.8 cudatoolkit=11.0 cudnn=8.0 -c conda-forge
conda activate n2v
pip install tensorflow==2.4
```
Then install the required packages into the active environment with:
```shell
pip install -r requirements.txt
```

