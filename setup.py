"""
setup.py — ratri installation script.

Installs all dependencies required to run the ratri HRCCS observation
simulator, including ratri.py, stellar.py, and generate_runner.py.

Usage:
    pip install -e .          # editable install (recommended for development)
    pip install .             # standard install
    pip install -r requirements.txt  # install dependencies only
"""

from setuptools import setup, find_packages

setup(
    name             = 'ratri',
    version          = '0.1.0',
    description      = 'High-Resolution Cross-Correlation Spectroscopy observation simulator for exoplanet atmospheres',
    long_description = open('README.md').read() if __import__('os').path.exists('README.md') else '',
    long_description_content_type = 'text/markdown',
    python_requires  = '>=3.10',

    py_modules = [
        'ratri',
        'stellar',
        'generate_runner',
    ],

    install_requires = [
        'numpy>=1.24',
        'scipy>=1.10',
        'astropy>=5.3',
        'astroplan>=0.9',
        'astroquery>=0.4.6',
        'matplotlib>=3.7',
        'pandas>=1.5',
        'h5py>=3.8',
        'requests>=2.28',
    ],

    extras_require = {
        # Optional planet spectrum generator
        'petitradtrans': ['petitRADTRANS>=2.4'],
        # Development tools
        'dev': [
            'jupyter',
            'ipywidgets',
            'tqdm',
        ],
    },

    classifiers = [
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
