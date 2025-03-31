import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mindtorch",
    version="0.0.1",
    author="lvyufeng",
    author_email="lvyufeng@cqu.edu.cn",
    description="mindtorch project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lvyufeng/mindtorch",
    project_urls={
        "Bug Tracker": "https://github.com/lvyufeng/mindtorch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=("example", "tests")),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    python_requires=">=3.9",
    install_requires=[
        "mindspore>=2.4",
        "requests",
    ],
)
