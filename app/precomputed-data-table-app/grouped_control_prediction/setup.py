#!/usr/bin/env python

# pylint: skip-file

from setuptools import setup, find_packages

install_requires = ["pandas"]

setup(
    name="grouped_control_prediction",
    version="0.1",
    description="Python module to predict the output signal of a strain based upon its FCS data trained on grouped control data",
    url="https://gitlab.sd2e.org/sd2program/precomputed-data-table/tree/grouped_control_prediction",
    author="Ryan Mahtab",
    author_email="rmahtab@sift.net",
    license="MIT",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={'': ['data/*.pkl']},
    install_requires=install_requires,
    zip_safe=False,
)