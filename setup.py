from setuptools import setup, find_packages

setup(
    name="stat_games",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "discopy", "numpyro"]
)
