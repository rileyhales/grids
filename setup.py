from setuptools import setup

version = '0.3'

with open('README.md', 'r') as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

setup(
    name='grids',
    packages=['grids'],
    version=version,
    description='Tools for extracting time series subsets from n-dimensional arrays in several storage formats.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    license='BSD 3-Clause Clear',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    install_requires=install_requires,
    extras_require=dict(pygrib=['pyproj', 'pygrib', ]),
)
