import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())

requirements = [
    # use environment.yml
]

setup(
    name="object_detector",
    version="0.0.1",
    url="https://github.com/SamBahrami/lightweight-object-detector-lightning",
    license='MIT',
    author="Sam Bahrami",
    author_email="bahrami.sam@gmail.com",
    description="Lightweight object detector using pytorch lightning",
    long_description=read("README.rst"),
    packages=find_packages(exclude=('tests',)),
    entry_points={
        'console_scripts': [
            'object_detector=object_detector.cli:cli'
        ]
    },
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ],
)
