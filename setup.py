from setuptools import setup, find_packages

sys.dont_write_bytecode = True

setup(
    name="kernelturbo",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
)