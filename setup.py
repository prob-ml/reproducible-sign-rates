from setuptools import setup

setup(
    name='reproduciblesignrates',
    author='Jackson Loper, Robert Barton, Meena Subramaniam, Maxime Dhainaut, Jeffrey Regier',
    version='0.0.1',
    description='Tools for evaluating cross-replicate reproducibility',
    packages=[
        'reproduciblesignrates',
    ],
    install_requires=[
        'tqdm',
        'matplotlib',
        'scipy',
        'pandas',
    ]
)
