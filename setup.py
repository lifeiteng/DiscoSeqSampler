#!/usr/bin/env python

"""Setup script for DiscoSeqSampler."""

import os

from setuptools import find_packages, setup


# Read the README file
def read_readme():
    """Read README.md file."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        return f.read()


# Read version from __init__.py
def get_version():
    """Get version from discoss/__init__.py."""
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, "discoss", "__init__.py")

    with open(version_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")


setup(
    name="discoss",
    version=get_version(),
    description="Distributed Coordinated Sequence Sampler",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Feiteng Li",
    author_email="lifeiteng0422@gmail.com",
    url="https://github.com/lifeiteng/DiscoSeqSampler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "lhotse==1.30.3",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "torchcodec==0.5",
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-xdist>=3.0",
            # Code formatting and linting
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.5",
            # Pre-commit hooks
            "pre-commit>=3.0",
            # Documentation
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # "discoss=discoss.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="distributed sampling sequence coordination",
)
