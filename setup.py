import os
from setuptools import setup

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()

setup(
    name='clear-cut',
    version='1.0.3',
    description='Number Crunching Box to Extract Edges from a Provided Image',
    url='https://github.com/chrispdharman/clear-cut',
    author='Christopher Harman',
    author_email='christopher.p.d.harman@gmail.com',
    license='Unlicensed',
    packages=['clear_cut'],
    install_requires=requirements,
    zip_safe=False
)
