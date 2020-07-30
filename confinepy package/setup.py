# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="confinepy", # Replace with your own username
    version="0.1.dev",
    author="Samuel S.Y. Wong",
    author_email="samswong@stanford.edu",
    description="A package for studying confining string",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samuelwong100/Confining-String",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'numba','sympy','matplotlib','pickle',
                      'dill']
)

