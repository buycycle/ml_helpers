# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="buycycle",
    version="0.0.1",
    description="Common functions to interact with buycycle.",
    license="MIT",
    author="Sebastian Rauner",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.4.0',
        'numpy>=1.21.0',
        'sqlalchemy>=1.4.0',
        'scikit-learn>=1.0.0',
        'python-json-logger>=2.0.7',
        'kafka-python>=2.0.2',
    ],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
    ]
)
