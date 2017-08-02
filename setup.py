import os

from bmcs.scripts.bmcs_version import CURRENT_VERSION
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="bmcs",
    version=CURRENT_VERSION,
    author="Rostislav Chudoba, Yionxiung Li, Abedulgader Baktheer, Alexander Scholzen",
    author_email="rostislav.chudoba@rwth-aachen.de",
    description=("Suite of educational packages for to the course "
                 "on Brittle-Matrix Composite Structures."),
    license="BSD",
    keywords="brittle matrix composite structures",
    url="http://packages.python.org/bmcs",
    packages=find_packages(),
    # If any package contains *.txt or *.rst files, include them:
    package_data={
        '': ['*.txt', '*.md', '*.rst', '*.png'],
        'reporter': ['texfiles/example.tex',
                     'texfiles/packages.tex',
                     'texfiles/scidoc.cls',
                     ]
    },
    include_package_data=True,
    long_description='',
    entry_points={
        'gui_scripts': [
            'bmcs = bmcs.scripts.bmcs_app:run_bmcs_launcher',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
