import pathlib
import setuptools
from setuptools import find_namespace_packages

setuptools.setup(
    name="scalify",
    version="0.0.1",
    author="Jovan Sardinha",
    author_email="jovan.sardinha@gmail.com",
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
)
