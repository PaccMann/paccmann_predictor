"""Install package."""
import codecs
import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


setup(
    name='paccmann_predictor',
    version=get_version('paccmann_predictor/__init__.py'),
    description=('PyTorch implementation of PaccMann'),
    long_description=open('README.md').read(),
    url='https://github.com/PaccMann/paccmann_predictor',
    author='Ali Oskooei, Jannis Born, Matteo Manica, Joris Cadow',
    author_email=(
        'ali.oskooei@gmail.com, jab@zurich.ibm.com, '
        'drugilsberg@gmail.com, joriscadow@gmail.com'
    ),
    install_requires=['numpy', 'scipy', 'torch>=1.0.0'],
    packages=find_packages('.'),
    zip_safe=False,
)
