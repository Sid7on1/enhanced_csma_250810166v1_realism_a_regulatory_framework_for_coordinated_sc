import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List

# Define constants
PROJECT_NAME = "enhanced_cs.MA_2508.10166v1_REALISM_A_Regulatory_Framework_for_Coordinated_Sc"
VERSION = "1.0.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your@email.com"
DESCRIPTION = "A package for the REALISM: A Regulatory Framework for Coordinated Scheduling in Multi-Operator Shared Micromobility Services project"
LICENSE = "MIT"
URL = "https://github.com/your-username/your-repo-name"

# Define dependencies
INSTALL_REQUIRES: List[str] = [
    "torch",
    "numpy",
    "pandas",
]

# Define development dependencies
DEV_REQUIRES: List[str] = [
    "pytest",
    "flake8",
    "mypy",
]

# Define test dependencies
TEST_REQUIRES: List[str] = [
    "pytest",
]

# Define the setup function
def setup_package():
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        extras_require={
            "dev": DEV_REQUIRES,
            "test": TEST_REQUIRES,
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        keywords="REALISM regulatory framework coordinated scheduling micromobility services",
        project_urls={
            "Documentation": "https://your-repo-name.readthedocs.io/en/latest/",
            "Source Code": "https://github.com/your-username/your-repo-name",
        },
    )

# Define a custom install command
class CustomInstallCommand(install):
    def run(self):
        try:
            install.run(self)
        except Exception as e:
            print(f"Error during installation: {e}")

# Define a custom develop command
class CustomDevelopCommand(develop):
    def run(self):
        try:
            develop.run(self)
        except Exception as e:
            print(f"Error during development installation: {e}")

# Define a custom egg info command
class CustomEggInfoCommand(egg_info):
    def run(self):
        try:
            egg_info.run(self)
        except Exception as e:
            print(f"Error during egg info generation: {e}")

# Define the main function
def main():
    setup_package()

# Run the main function
if __name__ == "__main__":
    main()