#!/usr/bin/env python

# pylint: skip-file

from setuptools import setup, find_packages

install_requires = [
]

setup(
    name="fcs_signal_prediction",
    version="0.1",
    description="Python module to predict the output signal of a strain based upon its FCS data.",
    url="https://gitlab.sd2e.org/dbryce/fcs_signal_prediction",
    author="Daniel Bryce",
    author_email="dbryce@sift.net",
    license="MIT",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
    zip_safe=False,
    include_package_data=True
)
