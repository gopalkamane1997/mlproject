from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
requirements = []

def get_requirements(file_path:str) -> list:
    '''
    this function will return list of requirements
    '''
    with open ('requirements.txt') as file_obj:
        requirements = file_obj.readline()
    # requirements = [req for req in requirements ]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="ml_project",
    version="0.0.1",
    author="Gopal",
    author_email="Gopalkamane93@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)

