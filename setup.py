#!/usr/bin/env python3

import os
from setuptools import setup

# Package folder name (check also classifiers and keyworks below)
ai_name = 'her2bdl'

# get key package details from <ai_name>/__version__.py
about = {}  # type: ignore
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, ai_name, '__version__.py')) as f:
    exec(f.read(), about)

# load the README file and use it as the long_description for PyPI
with open('README.md', 'r') as f:
    readme = f.read()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(
    name=about['__title__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    packages=[ai_name, 'deploy'],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=[
        'numpy', 'pandas', 'xlrd', 'scikit-learn', 'scipy', 'tqdm', 'openslide-python'
        'h5py', 'matplotlib', 'jupyter', 'nose', 'wandb',
        'tensorflow>=2.1.0', 'tensorflow-probability'
    ],
    license=about['__license__'],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            f'predict={ai_name}.deploy.cli.predict:predict'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        # Add here more topics
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'

    ],
    # Add new keyworks
    keywords='bayesian deep learning her2 scoring uncertainty'
)
