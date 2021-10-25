#!/bin/sh

# This is a hack to get conda shell commands within the script, see
# https://github.com/conda/conda/issues/7126
source $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh

ENV=$1
conda create --name ${ENV} python=3.8

# Conda installations go here
conda install -n ${ENV} numpy pandas scikit-learn 
conda install -n ${ENV} matplotlib seaborn
conda install -n ${ENV} jupyter
conda install -n ${ENV} pytorch==1.7.1 cudatoolkit=10.1 -c pytorch

# Activate before using pip
conda activate ${ENV}
# Pip installation goes here

pip install --upgrade pip

# Install EconML, used in appendix baselines
python -m pip install Cython
git clone https://github.com/microsoft/EconML.git
python EconML/setup.py develop
