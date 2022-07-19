from setuptools import setup

with open("requirements_pip.txt") as req_file:
    requirements = [req.strip() for req in req_file.readlines()]

setup(
    name='jpy_tools',
    author="Zhijian Liu",
    author_email="11930685@mail.sustech.edu.cn",
    description="All scirepts writed by Zhijian Liu",
    install_requires=requirements,
    license="BSD license",
    python_requires=">=3.8",
    setup_requires=requirements,
    url="https://github.com/ZhaiLab-SUSTech/Liuzj_allScripts/tree/master/jModule",
    version="1.0",
    zip_safe=False,
    packages=['jpy_tools', 'jpy_tools.singleCellTools']
)