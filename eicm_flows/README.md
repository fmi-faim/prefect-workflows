# Workflows to estimated illumination correction matrices
These flows can be used to estimated illumination correction matrices (EICM) 
with 
[eicm](http://github.com/fmi-faim/eicm).

# Installation
We recommend installing the requirements into a fresh conda environment.
```shell
conda create -n eicm python=3.9
conda activate eicm
pip install -r requirements.txt
```

# Create Shading Reference \[Yokogawa\]
Creates a single 2D shading reference from a Yokogawa experiment. 
The raw data are multiple Z-Stacks at different positions. 
From these Z-Stacks the indicated z-plane is extracted. 
From the extracted z-planes the corresponding dark-image (background) is subtracted. 
Then the median projection is computed over all extracted and dark-image subtracted z-planes.
The result is saved as shading reference.

# EICM with Median Filter
Simply applies a median filter, normalizes to the maximum and saves the result as illumination matrix.

## Parameters
* `shading_reference`: 2D shading reference
* `filter_size`: Size of the median filter 

# EICM with Gaussian Fit
Fits a 2D arbitrarily rotated Gaussian to the provided shading references, normalizes to the maximum and saves the estimated illumination matrix.

## Parameters
* `shading_reference`: 2D shading reference

# EICM with Polynomial Fit
Fits a 2D polynomial to the provided shading references, normalizes to the maximum and saves the estimated illumination matrix.

## Parameters
* `shading_reference`: 2D shading reference
* `polynomial_degree`
* `order`


  
# Outputs
Both methods save the estimated illumination correction matrix with the 
corresponding suffixes `eicm-blur` or `eicm-fit`. 
Additionally, an info markdown file is generated.