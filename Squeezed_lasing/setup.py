# Setup file for the squeezed_lasing package.

import setuptools

# setup details
setuptools.setup(
    name="squeezed_lasing",
    version="0",
    author="Rodrigo Grande de Diego",
    description="A package to study squeezed lasing for my masters thesis",
    packages=setuptools.find_packages(where='src'),
    install_requires=[
        "matplotlib",
        "numpy",
        "jupyter",
        "qutip",
        "tqdm",
        "scipy",
    ],
    python_requires=">=3.9",
)