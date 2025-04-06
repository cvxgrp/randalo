#!/usr/bin/python
import subprocess
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# get all the git tags from the cmd line that follow our versioning pattern
# git_tags = subprocess.Popen(
#     ["git", "tag", "--list", "v*[0-9]", "--sort=version:refname"],
#     stdout=subprocess.PIPE,
# )
# tags = git_tags.stdout.read()
# git_tags.stdout.close()
# tags = tags.decode("utf-8").split("\n")
# tags.sort()
# print("-----")
# print(tags)


# PEP 440 won't accept the v in front, so here we remove it, strip the new line and decode the byte stream
# VERSION_FROM_GIT_TAG = tags[-1][1:] if len(tags) > 0 else "v0.0"
from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import sysconfig
import os
import platform
import subprocess


def run_cmd(cmd):
    try:
        output = subprocess.check_output(
            cmd.split(" "), stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        raise RuntimeError(output)
    return output.rstrip()


# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += [
    "-g0",
    "-Wall", 
    "-Wextra", 
    "-DNDEBUG", 
    "-O3",
]
include_dirs = [
    "randalo/src",
    "randalo/src/include",
]
libraries = []
library_dirs = []
runtime_library_dirs = []

eigen_include_path = 'src/eigen/'
adelie_path = '/home/parthnobel/.pyenv/versions/alo/lib/python3.12/site-packages/adelie/'
adelie_include_path = os.path.join(adelie_path, "src/include")
include_dirs += [
    eigen_include_path,
    adelie_include_path,
]

system_name = platform.system()
"""
if system_name == "Darwin":
    # if user provides OpenMP install prefix (containing lib/ and include/)
    if "OPENMP_PREFIX" in os.environ and os.environ["OPENMP_PREFIX"] != "":
        omp_prefix = os.environ["OPENMP_PREFIX"]

    # else if conda environment is activated
    elif not (conda_prefix is None):
        omp_prefix = conda_prefix
    
    # otherwise check brew installation
    else:
        # check if OpenMP is installed
        no_omp_msg = (
            "OpenMP is not detected. "
            "MacOS users should install Homebrew and run 'brew install libomp' "
            "to install OpenMP. "
        )
        try:
            libomp_info = run_cmd("brew info libomp")
        except:
            raise RuntimeError(no_omp_msg)
        if "Not installed" in libomp_info:
            raise RuntimeError(no_omp_msg)

        # grab include and lib directory
        omp_prefix = run_cmd("brew --prefix libomp")

    omp_include = os.path.join(omp_prefix, "include")
    omp_lib = os.path.join(omp_prefix, "lib")

    # augment arguments
    include_dirs += [omp_include]
    extra_compile_args += [
        "-Xpreprocessor",
        "-fopenmp",
    ]
    runtime_library_dirs += [omp_lib]
    library_dirs += [omp_lib]
    libraries += ['omp']
"""
    
if system_name == "Linux":
    extra_compile_args += [
        "-fopenmp", 
        "-march=native",
   ]
    libraries = ['gomp']

ext_modules = [
    Pybind11Extension(
        "randalo.randalo_core",
        sorted(glob("randalo/src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=17,
    ),
]
setup(
    name="randalo",
    # version=VERSION_FROM_GIT_TAG,  # Required
    version="0.1.0",
    build_requires=[
        "pybind11",
    ],
    setup_requires=[
        "setuptools>=18.0",
    ],
    packages=["randalo"], 
    install_requires=[
        "numpy >= 1.17.5",
        "scipy",
        "torch",
        "torch-linops",
    ],
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cvxgrp/randalo",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Parth Nobel",
    author_email="ptnobel@stanford.edu",

    package_data={
        "randalo": [
            "src/**/*.hpp",
            "src/**/*.cpp",
            "randalo_core.cpython*",
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)

