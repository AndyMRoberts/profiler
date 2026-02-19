from setuptools import setup, find_packages

setup(
    name='profiler',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "psutil>=5.9.0",
        "matplotlib>=3.5.0",
        "nvidia-ml-py>=12.0.0",
        "pyJoules>=0.2.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
)