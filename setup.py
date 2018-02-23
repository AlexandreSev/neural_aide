# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='neural_aide',
    version='0.1.0',
    description='Neural Active Learning',
    long_description=readme,
    author='Alexandre Sevin',
    url='https://github.com/alexsev/neural_aide',
    license=license,
    packages=find_packages()
)
