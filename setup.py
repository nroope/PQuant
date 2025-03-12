import os
from setuptools import setup, find_packages


requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = []
with open(requirements_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        install_requires.append(line)

setup(name="pquant",
      packages=find_packages(),
      python_requires='>=3.10',
      install_requires=install_requires
      )
