"""Installation script, which makes this project pip installable.

Install with: `pip install --editable .`
"""

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='LULC/LPIS classification of Sentinel 1/2 imagery using \
        incremental algorithms.',
    author='JP',
    license='MIT',
)
