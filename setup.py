from setuptools import setup, find_packages

setup(
    name="el",
    version="0.1.0",
    description="",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy==2.3.3",
        "pandas==2.3.2",
        "matplotlib==3.10.6",
    ],
    extras_require={
        "dev": [
            "black==25.1.0",
            "flake8==7.3.0",
        ],
    },
    classifiers=[],
)
