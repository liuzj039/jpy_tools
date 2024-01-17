# Install
0. clone this package
```bash
git clone https://github.com/liuzj039/jpy_tools.git
cd ./jpy_tools/jModule
```
1. install reliable packages
- create a new environment
```bash
conda create -n sc_py
```
- install conda packages
```bash
conda install --file requirements_conda.txt -c conda-forge -c r -c bioconda -n sc_py # or mamba

conda activate sc_py
python -m ipykernel install --name sc_py --user # add jupyter kernel
```
- install pip packages
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
from joblib import Parallel, delayed
import os
os.environ['R_HOME'] = '{conda_path}/anaconda3/envs/sc_py/lib/R'
```
{conda_path} is the folder where conda is installed.
