from setuptools import setup, find_packages

setup(
    name="auto_ml_regression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
    ],
    author="Iurkin Andrei",
    description="AutoML library for regression tasks",
    python_requires=">=3.8",
)