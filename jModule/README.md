# Install
0. clone this package
```bash
git clone https://github.com/ZhaiLab-SUSTech/Liuzj_allScripts.git
cd ./Liuzj_allScripts/jModule
```
1. install reliable packages
1.1 create new environments
```bash
conda create -n sc_py
conda create -n sc_r
```
1.2 install conda packages
```bash
conda install --file requirements_py.txt -c conda-forge -c r -c bioconda -n sc_py # or mamba
conda install --file requirements_r.txt -c conda-forge -c r -c bioconda -n sc_r # or mamba

conda activate sc_py
ln -s `realpath $CONDA_PREFIX/../test_jModule_R/bin/R` $CONDA_PREFIX/bin/R # link R envirment
```
1.3 install pip packages
```bash
pip install -r requirements_pip.txt
```
2. install this package
```bash
python setup.py install
```
