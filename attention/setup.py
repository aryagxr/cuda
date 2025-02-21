from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='causal_attention',  # Must match the name in PYBIND11_MODULE
    ext_modules=[
        CUDAExtension('causal_attention', ['causal-attn.cu'], include_dirs=["/home/ari/anaconda3/envs/torch-keras/lib/python3.12/site-packages/torch/include"],) # Must match the name in PYBIND11_MODULE
    ],
    cmdclass={
        'build_ext': BuildExtension
    })