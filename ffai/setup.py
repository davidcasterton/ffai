from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ffai",
    version="0.1.0",
    author="David Casterton",
    author_email="david.casterton@gmail.com",
    description="Fantasy Football draft simulator and season simulator, to train a RL auction draft bidding model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "ffai": ["config/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    scripts=['scripts/auction.py']
)
