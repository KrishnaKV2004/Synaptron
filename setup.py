from setuptools import setup, find_packages

setup(
    name="Synaptron",
    version="1.0.0",
    description="A Neural Network framework built from scratch using NumPy !",
    author="Krishna Verma",
    author_email="krishnaverma.0227@gmail.com",
    url="https://github.com/KrishnaKV2004/Synaptron",
    packages=find_packages(),  # finds the Synaptron package folder automatically
    install_requires=[
        "numpy",  # numpy will be installed automatically if not present
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)