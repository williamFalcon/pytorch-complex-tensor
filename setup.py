#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
      name='pytorch-complex-tensor',
      version='0.0.134',
      description='Pytorch complex tensor',
      author='William Falcon',
      author_email='waf2107@columbia.edu',
      url='https://github.com/williamFalcon/pytorch-complex-tensor',
      install_requires=[
            'numpy>=1.15.4',
            'torch>=1.0',
            'torchvision>=0.2.1'
      ],
      packages=find_packages()
)
