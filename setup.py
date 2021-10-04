# --- built in ---
import os
import re

from setuptools import find_packages, setup

def get_version():
    with open(os.path.join('toy_gradlogp', '__init__.py'), 'r') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(
    name='toy_gradlogp',
    version=get_version(),
    description='Some toy examples of score matching algorithms written in PyTorch',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Ending2015a/toy_gradlogp',
    author='JoeHsiao',
    author_email='joehsiao@gapp.nthu.edu.tw',
    license='MIT',
    python_requires=">=3.6",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='score-matching playform pytorch',
    packages=[
        # exclude deprecated module
        package for package in find_packages()
        if package.startswith('toy_gradlogp')
    ],
    install_requires=[
        'tensorflow>=2.3.0',
        'numpy',
        'torch>=1.8.0',
        'matplotlib'
    ],
)
