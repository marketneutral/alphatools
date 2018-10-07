#!/usr/bin/env bash

conda create -n env_alphatools_latest -y python=$PYTHON_VERSION numpy=1.14.1 pandas=0.22.0 scipy=1.0.0 libgfortran=3.0 pip
source activate env_alphatools_latest
python -m pip install -r requirements_latest.txt --no-cache-dir
python -m pip install -r requirements_blaze_latest.txt --no-cache-dir
pip install zipline==1.3.0 --no-cache-dir
pip install statsmodels==0.9.0 --upgrade --no-cache-dir
pip install ipykernel --no-cache-dir
cd ..
pip install -e alphatools --no-cache-dir
python -m ipykernel install --user --name env_alphatools_latest --display-name "env_alphatools_latest"
