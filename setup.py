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
    name='clear-cut',
    version='1.2.4',
    description='Number Crunching Backend to Extract Edges from a Provided Image',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/chrispdharman/clear-cut',
    author='Christopher Harman',
    author_email='christopher.p.d.harman@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(exclude=[
        'documentation',
        'results',
        'tests'
    ]),
    install_requires=requirements,
    package_data={
        "clear_cut": [
            'images/*.jpeg',
            'images/*.JPG',
            'images/*.jpg'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Information Technology',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.6',
    zip_safe=False
)
