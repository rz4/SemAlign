#!/usr/bin/env python3

__author__ = 'Rafael Zamora-Resendiz, rz4@hood.edu'

from setuptools import setup, find_packages

setup(
    name="SemAlign",
    version="0.0.0",
    description="Semantic Alignment of Text Using Fuzzy Kernels",
    license="MIT",
    keywords="NLP Alignment",
    packages=find_packages(exclude=["data", "notebooks"]),
    install_requires = ["numpy", "scipy"],
)
