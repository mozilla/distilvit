from setuptools import setup, find_packages

setup(
    name="distilvit",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train=distilvit.train:main",  # "main" is a function in "train_model.py"
        ],
    },
)
