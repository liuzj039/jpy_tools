# Install
0. clone this package
```bash
git clone https://github.com/ZhaiLab-SUSTech/Liuzj_allScripts.git
cd ./Liuzj_allScripts/jModule
```
1. install reliable packages
1.1 create a new environment
```bash
conda create -n sc_py
```
1.2 install conda packages
```bash
conda install --file requirements_conda.txt -c conda-forge -c r -c bioconda -n sc_py # or mamba

conda activate sc_py
python -m ipykernel install --user --display-name 'sc_py' # add jupyter kernel
```
1.3 install pip packages
```bash
pip install -r requirements_pip.txt
```
2. install this package
```bash
python setup.py install
```

# Tips
The following code snippets need to be added to the python script if the global environment variables are not set.
```python
import os
os.environ['PATH'] = '{conda_path}/envs/sc_py/bin:' + os.environ['PATH']
import rpy2.robjects as ro
ro.r(".libPaths")("{conda_path}/envs/sc_py/lib/R/library")
```
{conda_path} is the folder where conda is installed.
