"""
TAPnx setup script.

See license in LICENSE.txt.
"""

import os

from setuptools import setup

# provide a long description using reStructuredText
LONG_DESCRIPTION = r"""
**TAPnx** is a Python package that lets you...


Read the `docs`_ or see usage examples and demos on `GitHub`_.

.. _GitHub: https://github.com/tapnx/tapnx-examples
.. _docs: https://tapnx.readthedocs.io
.. _OSMnx\: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks: http://geoffboeing.com/publications/osmnx-complex-street-networks/
"""

# list of classifiers from the PyPI classifiers trove
CLASSIFIERS = [
    #"Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Network Assignment",
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Scientific/Engineering :: Traffic",
    "Topic :: Scientific/Engineering :: Mathematics"
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]

DESC = (
    "Short Description"
)

# only specify install_requires if not in RTD environment
if os.getenv("READTHEDOCS") == "True":
    INSTALL_REQUIRES = []
else:
    with open("requirements.txt") as f:
        INSTALL_REQUIRES = [line.strip() for line in f.readlines()]

# now call setup
setup(
    name="tapnx",
    version="1.0.0",
    description=DESC,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    url="https://github.com/tapnx/tapnx",
    author="Sam O'Neill",
    author_email="sam.t.oneill@googlemail.com",
    license="MIT",
    platforms="any",
    packages=["tapnx"],
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    # extras_require={
    #     "folium": ["folium>=0.11"],
    #     "kdtree": ["scipy>=1.5"],
    #     "balltree": ["scikit-learn>=0.24"],
    # },
)