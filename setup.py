#!/usr/bin/env python

from setuptools import setup

packages = {
    'hw2': 'hw2/',
    'hw2.lib': 'hw2/lib',
}

setup(
    name='hw2',
    version='1.0',
    description='HW2',
    author='Ilariia Belova',
    author_email='ilariia@yandex-team.ru',
    requires=['pytest', 'matplotlib'],
    packages=packages,
    package_dir=packages
)
