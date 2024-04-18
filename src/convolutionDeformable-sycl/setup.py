import os
import glob
from setuptools import find_packages, setup
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_sycl = glob.glob(os.path.join(extensions_dir, "sycl", "*.cpp"))
    sources = main_file + source_sycl
    sources = [os.path.join(extensions_dir, s) for s in sources]

    extension = DPCPPExtension

    ext_modules = [
        extension(
            name='_ext',
            sources=sources,
            include_dirs=ipex.xpu.cpp_extension.include_paths(),
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
    cmdclass={"build_ext": DpcppBuildExtension},
)

