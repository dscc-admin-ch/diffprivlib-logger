import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

this_directory = pathlib.Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="diffprivlib_logger",
    packages=find_packages(),
    version="0.0.2",
    description="A logger wrapper for DiffPrivLib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dscc-admin/lomas/",
    author="Data Science Competence Center, Swiss Federal Statistical Office",
    author_email="dscc@bfs.admin.ch",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=["diffprivliv", "logger", "serialiser", "deserialiser"],
    python_requires=">=3.10, <4",
    install_requires=[
        "diffprivlib>=0.6.4",
        "scikit-learn==1.5.0",
    ],
)
