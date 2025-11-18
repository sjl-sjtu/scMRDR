from setuptools import setup, find_packages

setup(
    name="scMRDR", 
    version="1.1.0",
    author="Jianle Sun",
    description="A package to integrate unpaired multi-omics single-cell data via single-cell Multi-omics Regularized Disentangled Representations.",
    # url="https://github.com/sjl-sjtu/perturb_vae", 
    packages=find_packages(), 
    install_requires=[ 
        "anndata",
        "matplotlib",
        "numpy",
        "pandas",
        "scanpy",
        "scikit_learn",
        "scipy",
        "torch",
        "POT"
    ],
    classifiers=[ 
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)