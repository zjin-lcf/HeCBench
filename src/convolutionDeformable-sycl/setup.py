import os
import glob
from setuptools import find_packages, setup

# https://docs.pytorch.org/docs/stable/cpp_extension.html
from torch.utils.cpp_extension import BuildExtension, SyclExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_sycl = glob.glob(os.path.join(extensions_dir, "sycl", "*.sycl"))
    sources = main_file + source_sycl
    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        SyclExtension(
            name='_ext', sources=sources,
        )
    ]

    return ext_modules

setup(
    name="convolution",
    version="0.1",
    author="",
    url="https://github.com/lucasjinreal/DCNv2_latest",
    description="deformable convolution",
    packages=find_packages(exclude=("configs", "tests")),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)

