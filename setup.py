from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='fast1dkmeans',
    version='0.1.2',
    packages=['fast1dkmeans', 'fast1dkmeans.tests'],
    author='Felix Stamm',
    author_email='felix.stamm@cssh.rwth-aachen.de',
    description='A python package for optimal 1d k-means clustering.',
    install_requires=[
              'numpy',
              'numba',
          ],
    python_requires='>=3.8',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
