from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

VERSION = '0.7.6'
try:
    import pypandoc
    README = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    README = open('README.md').read()

setup(
    name='aocr',
    url='https://github.com/emedvedev/attention-ocr',
    download_url='https://github.com/emedvedev/attention-ocr/archive/{}.tar.gz'.format(VERSION),
    author='Ed Medvedev',
    author_email='edward.medvedev@gmail.com',
    version=VERSION,
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    description=('''Optical character recognition model '''
                 '''for Tensorflow based on Visual Attention.'''),
    long_description=README,
    entry_points={
        'console_scripts': ['aocr=aocr.__main__:main'],
    }
)