# Setup script for the pycvc package
#
# Usage: python setup.py install
#
import os
from setuptools import setup, find_packages

DESCRIPTION = "pycvc: Analyze CVC stormwater data"
LONG_DESCRIPTION = DESCRIPTION
NAME = "pycvc"
VERSION = "0.3.0"
AUTHOR = "Paul Hobson (Geosyntec Consultants)"
AUTHOR_EMAIL = "phobson@geosyntec.com"
URL = ""
DOWNLOAD_URL = ""
LICENSE = "BSD 3-clause"
PACKAGES = find_packages(exclude=[])
PLATFORMS = "Python 3.4 and later."
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Formats and Protocols :: Data Formats",
    "Topic :: Scientific/Engineering :: Earth Sciences",
    "Topic :: Software Development :: Libraries :: Python Modules",
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
]
INSTALL_REQUIRES = ['wqio', 'pybmpdb', 'pynsqd']
PACKAGE_DATA = {
    'pycvc.tex': ['*.tex'],
    'pycvc.tests.testdata': ['*.csv', '*.accdb'],
    'pycvc.tests.baseline_images.viz_tests': ['*.png'],
}

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    platforms=PLATFORMS,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    zip_safe=False
)
