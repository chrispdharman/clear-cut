import os
import setuptools

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()

setuptools.setup(
    name='clear-cut-quanta-chris',
    version='1.0.4',
    description='Number Crunching Backend to Extract Edges from a Provided Image',
    url='https://github.com/chrispdharman/clear-cut',
    author='Christopher Harman',
    author_email='christopher.p.d.harman@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Information Technology",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
    zip_safe=False
)
