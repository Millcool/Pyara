"""Metadata of package"""
from setuptools import setup, find_packages


requirements = []

# Создаст библиотеку для загрузки на PyPI
setup(name='Pyara',
      version='0.1.3',
      url='https://github.com/Millcool/Pyara.git',
      license='MIT',
      author='Ilya Mironov',
      author_email='ilyamironov210202@gmail.com',
      description='Library for audio classification',
      long_description=open('README.md').read(),
      long_description_content_type = "text/markdown",
      packages=find_packages(exclude=['tests']),
      zip_safe=False,
      install_requires=requirements,
      platform = 'Any',
      classifiers=[
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent"
      ],
      )