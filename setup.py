"""Install package."""
from setuptools import setup, find_packages

setup(
    name='paccmann_predictor',
    version='0.0.1',
    description=(
        'PyTorch implementation of PaccMann'
    ),
    long_description=open('README.md').read(),
    url='https://github.com/PaccMann/paccmann_predictor',
    author='Ali Oskooei, Jannis Born, Matteo Manica, Joris Cadow',
    author_email=(
        'ali.oskooei@gmail.com, jab@zurich.ibm.com, '
        'drugilsberg@gmail.com, joriscadow@gmail.com'
    ),
    install_requires=[
        'numpy',
        'scipy',
        'tensorflow>=1.10.0,<2.0',
        'torch>=1.0.0'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)
