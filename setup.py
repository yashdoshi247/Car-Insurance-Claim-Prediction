from setuptools import find_packages,setup
from typing import List

#Function to return list of requirements
def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() #Reading each package
        requirements = [requirement.replace('\n',"") for requirement in requirements] #For replacing \n at end of line with empty string

        if '-e .' in requirements:  #Condition to ignore last line that runs setup.py
            requirements.remove('-e .')
    
    return requirements

setup(
    name='Car_Insurance_Claim',
    version = '0.0.1',
    author = 'Yash Doshi',
    author_email='yashjdoshi99@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)