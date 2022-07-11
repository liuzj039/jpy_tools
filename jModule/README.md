# Install
0. clone this package
```bash
git clone https://github.com/ZhaiLab-SUSTech/Liuzj_allScripts.git
cd ./Liuzj_allScripts/jModule
```
1. install reliable packages
`pip install -r requirements.txt`
2. install anndata2ri and pySCTransform
  - `pip install git+https://github.com/liuzj039/pysctransform.git`
3. install this package
```bash
python setup.py install
```
4. install R package
- You can install these pacakges through conda or mamba (Recommend using a discrete environment)
  - Seurat
  - Monocle3
  - SingleCellExperiments
- Or add Liu Zhijian's R directly to the environment variable
  - /public1/software/liuzj/softwares/anaconda3/envs/R/bin/R
