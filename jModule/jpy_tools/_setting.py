class jtConfig:
    """
    config:
        seuratDisk_rPath: "/public/home/liuzj/softwares/anaconda3/envs/seurat_disk/bin/R"
        seuratDisk_rLibPath: "/public/home/liuzj/softwares/anaconda3/envs/seurat_disk/lib/R/library"
    """

    def __init__(self):
        self.seuratDisk_rPath = (
            "/public/home/liuzj/softwares/anaconda3/envs/seurat_disk/bin/R"
        )
        self.seuratDisk_rLibPath = (
            "/public/home/liuzj/softwares/anaconda3/envs/seurat_disk/lib/R/library"
        )
        self.dt_dirPkl = {
            "ipf": "/public/home/liuzj/tmp/python_pkl/",
            "scem": "/scem/work/liuzj/tmp/python_pkl/",
        }
    def __str__(self):
        dt_config = {
            "seuratDisk_rPath": self.seuratDisk_rPath,
            "seuratDisk_rLibPath": self.seuratDisk_rLibPath,
            "dt_dirPkl":self.dt_dirPkl
        }
        self._dt_config = dt_config
        return str(self._dt_config)

    def __repr__(self):
        return str(self)


settings = jtConfig()