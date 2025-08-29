from setuptools import setup
from Cython.Build import cythonize
import numpy

# Tentukan file .pyx yang ingin Anda compile
ext_modules = [
    "agents/c_ppo.pyx",
    "envs/c_baseline_trading_env.pyx"
]

setup(
    ext_modules=cythonize(ext_modules, language_level="3"),
    include_dirs=[numpy.get_include()] # Diperlukan untuk mengakses C-API NumPy
)

# command run terminal : python setup.py build_ext --inplace