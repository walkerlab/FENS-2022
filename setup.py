#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="fens",
    version="0.1",
    description="A collection of code for FENS 2022 summer school workshop on Deep Learning in Neuroscience",
    author="Edgar Y. Walker, Zhuokun Ding, Suhas Shrinivasan",
    author_email="eywalker@uw.edu",
    packages=find_packages(exclude=[]),
    install_requires=["neuralpredictors~=0.0.1", "torch", "numpy"],
)
