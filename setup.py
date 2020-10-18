import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='AE-LQ-comp5328_a1',
    author="Alex Elias - Luca Quaglia",
    author_email="aeli0392@uni.sydney.edu.au",
    description="Advanced Machine Learning Assignment 1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hixan/comp5328-a1",
    packages=['NMF_Implementation'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
