import os 
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.0.1'
sources = ['pybind.cpp','cpu_func.cu','gpu_func.cu','Tensor.cu','tensor_func.cu']

setup(
    name='mytensor',
    version=__version__,
    author='liu',
    author_email='2100012953@stu.pku.edu.cn',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['torch'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[CUDAExtension(name='raw_tensor',sources=sources,) ],
    cmdclass={'build_ext': BuildExtension },
    classifiers=['License :: OSI Approved :: MIT License', ],
    )

# main.cu Tensor.cu cpu_func.cu gpu_func.cu tensor_func.cu -lcublas