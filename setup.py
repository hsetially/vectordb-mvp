from setuptools import setup, find_packages

setup(
    name="vectordb_mvp",
    version="0.1.0",
    description="MVP for VectorDB",
    author="Davanapally Itesh",
    author_email="iteshrao@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy==1.24.3",
        "grpcio==1.56.0",
        "grpcio-tools==1.56.0",
        "h5py==3.9.0",
        "faiss-cpu==1.7.4",
        "pyarrow==12.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)