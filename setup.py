from setuptools import setup

name = 'grids'
version = '0.15'
description = 'Tools for extracting time series subsets from n-dimensional arrays in several storage formats.'
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: GIS',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
]
with open('README.md', 'r') as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

setup(
    name=name,
    packages=[name, ],
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    license='BSD 3-Clause Clear',
    python_requires='>=3',
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=dict(pygrib=['pyproj', 'pygrib', ]),
)
