from setuptools import find_packages, setup


setup(
    name="smio-clrp",
    version="0.1.0",
    description="Base architecture for the SMIO-Hexaly CLRP challenge.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24"],
    entry_points={"console_scripts": ["clrp=smio_clrp.cli:main"]},
    extras_require={
        "exact": ["gurobipy"],
        "alns": ["alns"],
        "clustering": [],
        "dev": ["pytest", "ruff", "mypy"],
    },
)
